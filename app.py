# app.py
# Bizboost - Πρόβλεψη Ρυθμίσεων Εξωδικαστικού (GR UI)
# - XGBoost προαιρετικό (fallback σε RandomForest)
# - Ανά οφειλή: πρόταση δόσης + διαγραφή + οροφή μηνών (420 τράπεζες/servicers, 240 ΑΑΔΕ/ΕΦΚΑ)
# - Συνοφειλέτες με δομημένα πεδία (ετήσιο εισόδημα -> μηνιαίο, αφαίρεση ΕΔΔ)
# - Supabase Postgres μέσω SQLAlchemy + psycopg v3
# - PDF export (βασικό, με logo αν υπάρχει)

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# ML imports: XGBoost optional → fallback σε RandomForest
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestRegressor

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ────────────────────────────── UI CONFIG ──────────────────────────────
st.set_page_config(page_title="Bizboost - Πρόβλεψη Ρυθμίσεων", page_icon="💠", layout="wide")

LOGO_PATH = "logo.png"   # το logo σου στη ρίζα
DATA_CSV  = "cases.csv"  # προαιρετικό αρχικό import αν DB άδεια

# ─────────────────────────── ΣΤΑΘΕΡΕΣ / ΛΙΣΤΕΣ ───────────────────────────
CREDITORS = [
    # Servicers / Τράπεζες
    "DoValue","Intrum","Cepal","Qquant","APS","EOS","Veralitis",
    "Πειραιώς","Εθνική","Eurobank","Alpha",
    # Δημόσιο
    "ΑΑΔΕ","ΕΦΚΑ",
]
LOAN_TYPES = ["Στεγαστικό","Καταναλωτικό","Επαγγελματικό"]

PUBLIC_CREDITORS = {"ΑΑΔΕ", "ΕΦΚΑ"}
BANK_SERVICERS = {
    "DoValue","Intrum","Cepal","Qquant","APS","EOS","Veralitis",
    "Πειραιώς","Εθνική","Eurobank","Alpha"
}

def term_cap_for_single_debt(creditor_name: str, age_cap_months: int) -> int:
    """Οροφή μηνών για ΜΙΑ οφειλή ανά πιστωτή, με κόφτη ηλικίας."""
    c = (creditor_name or "").strip()
    if c in BANK_SERVICERS:
        policy_cap = 420
    elif c in PUBLIC_CREDITORS:
        policy_cap = 240
    else:
        policy_cap = 240
    return max(1, min(policy_cap, age_cap_months))

# ───────────────────── ΕΛΑΧΙΣΤΕΣ ΔΑΠΑΝΕΣ ΔΙΑΒΙΩΣΗΣ (ΕΔΔ) ─────────────────────
# Βασική κλίμακα (προσαρμόσιμη): 1 ενήλικας 537€, +269€ κάθε επιπλέον ενήλικας, +211€ ανά ανήλικο
def compute_edd(adults:int, children:int)->float:
    if adults <= 0 and children <= 0:
        return 0.0
    base_adult = 537
    add_adult  = 269
    per_child  = 211
    if adults <= 0:
        adults = 1
    total = base_adult + max(adults-1,0)*add_adult + children*per_child
    return float(total)

def months_cap_from_age(age:int)->int:
    """Απλός κόφτης μηνών βάσει ηλικίας (προσαρμόσιμο)."""
    try:
        a = int(age)
    except:
        return 120
    if a <= 35:  return 240
    if a <= 50:  return 180
    if a <= 65:  return 120
    return 60

def available_income(total_income:float, edd_household:float, extra_medical:float, extra_students:float, extra_legal:float)->float:
    extras = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
    return max(0.0, float(total_income or 0) - float(edd_household or 0) - extras)

# ───────────────────────────── ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ─────────────────────────────
def get_db_engine():
    # Με προτεραιότητα στα secrets (Streamlit/τοπικό)
    try:
        db_url = st.secrets["DATABASE_URL"]
    except Exception:
        db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        st.error("Δεν έχει οριστεί DATABASE_URL στα Secrets ή στα Environment variables.")
        st.stop()
    # SQLAlchemy driver name για psycopg v3
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def init_db(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS cases (
      case_id TEXT PRIMARY KEY,
      borrower TEXT,
      debtor_age INT,
      adults INT,
      children INT,
      monthly_income NUMERIC,
      property_value NUMERIC,
      annual_rate_pct NUMERIC,
      edd_use_manual INT,
      edd_manual NUMERIC,
      extra_medical NUMERIC,
      extra_students NUMERIC,
      extra_legal NUMERIC,
      age_cap INT,
      debts_json JSONB,
      co_debtors_json JSONB,
      term_months INT,
      predicted_at TEXT,
      predicted_monthly NUMERIC,
      predicted_haircut_pct NUMERIC,
      prob_accept NUMERIC,
      real_monthly NUMERIC,
      real_haircut_pct NUMERIC,
      accepted INT,
      real_term_months INT,
      real_writeoff_amount NUMERIC,
      real_residual_balance NUMERIC
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def load_data_db()->pd.DataFrame:
    engine = get_db_engine()
    init_db(engine)
    try:
        df = pd.read_sql("SELECT * FROM cases", con=engine)
        return df
    except Exception as e:
        st.error(f"Σφάλμα ανάγνωσης DB: {e}")
        return pd.DataFrame()

def _nan_to_none(x):
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x

def upsert_cases_db(df: pd.DataFrame):
    if df.empty:
        return
    engine = get_db_engine()
    init_db(engine)

    cols = [
        "case_id","borrower","debtor_age","adults","children","monthly_income","property_value",
        "annual_rate_pct","edd_use_manual","edd_manual","extra_medical","extra_students","extra_legal",
        "age_cap","debts_json","co_debtors_json","term_months","predicted_at",
        "predicted_monthly","predicted_haircut_pct","prob_accept",
        "real_monthly","real_haircut_pct","accepted","real_term_months",
        "real_writeoff_amount","real_residual_balance"
    ]
    df2 = df.copy()

    # Εγγύηση ότι τα JSON είναι strings έγκυρα
    for c in ["debts_json","co_debtors_json"]:
        if c in df2.columns:
            def _to_json_str(v):
                if isinstance(v, str):
                    return v
                try:
                    return json.dumps(v, ensure_ascii=False)
                except:
                    return "[]"
            df2[c] = df2[c].apply(_to_json_str)

    # NaN -> None
    df2 = df2.where(pd.notnull(df2), None)
    for c in df2.columns:
        df2[c] = df2[c].map(_nan_to_none)

    df2 = df2.reindex(columns=cols, fill_value=None)

    sql = f"""
    INSERT INTO cases ({",".join(cols)})
    VALUES ({",".join([f":{c}" for c in cols])})
    ON CONFLICT (case_id) DO UPDATE SET
      {",".join([f"{c}=EXCLUDED.{c}" for c in cols if c!="case_id"])};
    """
    with engine.begin() as conn:
        conn.execute(text(sql), df2.to_dict(orient="records"))

def csv_to_db_once_if_empty():
    engine = get_db_engine()
    init_db(engine)
    with engine.begin() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM cases")).scalar()
    if cnt == 0 and os.path.exists(DATA_CSV):
        try:
            dfcsv = pd.read_csv(DATA_CSV)
            # Καθαρισμός NaN σε JSON στήλες αν υπάρχουν
            for c in ["debts_json","co_debtors_json"]:
                if c in dfcsv.columns:
                    dfcsv[c] = dfcsv[c].fillna("[]").astype(str)
            upsert_cases_db(dfcsv)
            st.success("Έγινε αρχικό import από cases.csv")
        except Exception as e:
            st.warning(f"Αποτυχία import από cases.csv: {e}")

def load_data():
    csv_to_db_once_if_empty()
    return load_data_db()

def save_data(df: pd.DataFrame):
    upsert_cases_db(df)

# ─────────────────────────── ML ΜΟΝΤΕΛΟ / FEATURES ───────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    """Αν υπάρχει XGBoost → XGBRegressor, αλλιώς RandomForest."""
    if HAS_XGB:
        return Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("xgb", XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42, n_jobs=2
            ))
        ])
    else:
        return Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("rf", RandomForestRegressor(
                n_estimators=300, random_state=42, n_jobs=-1
            ))
        ])

def build_features_row(total_income, edd_household, extras_sum, total_debt, secured_amt, property_value, rate_pct, term_cap):
    avail = max(0.0, (total_income or 0) - (edd_household or 0) - (extras_sum or 0))
    debt_to_income = (total_debt or 0) / (total_income+1e-6)
    secured_ratio   = (secured_amt or 0) / (total_debt+1e-6)
    ltv             = (total_debt or 0) / (property_value+1e-6)
    x = pd.DataFrame([{
        "avail":avail,
        "income": total_income or 0,
        "edd": edd_household or 0,
        "extras": extras_sum or 0,
        "total_debt": total_debt or 0,
        "secured_amt": secured_amt or 0,
        "property": property_value or 0,
        "rate": (rate_pct or 0)/100.0,
        "term_cap": term_cap or 0,
        "dti": debt_to_income,
        "secured_ratio": secured_ratio,
        "ltv": ltv
    }])
    return x

def predict_single_debt_monthly(model, monthly_income, edd_val, extras_sum,
                                debt_balance, debt_secured_amt, property_value,
                                annual_rate_pct, age_cap_months, creditor_name):
    """Επιστρέφει (pred_monthly, haircut_pct, term_cap) για ΜΙΑ οφειλή."""
    term_cap = term_cap_for_single_debt(creditor_name, age_cap_months)

    Xd = build_features_row(
        total_income=monthly_income,
        edd_household=edd_val,
        extras_sum=extras_sum,
        total_debt=debt_balance,
        secured_amt=debt_secured_amt,
        property_value=property_value,
        rate_pct=annual_rate_pct,
        term_cap=term_cap
    )

    pred = None
    if model is not None:
        try:
            pred = float(model.predict(Xd)[0])
        except NotFittedError:
            pred = None
        except Exception:
            pred = None

    if pred is None:
        # Fallback rule-of-thumb: 70% διαθέσιμου
        avail = max(0.0, monthly_income - edd_val - extras_sum)
        pred = max(0.0, round(avail * 0.7, 2))
    else:
        pred = max(0.0, pred)

    if debt_balance > 0:
        expected_repay = pred * term_cap
        haircut_pct = float(np.clip(1 - (expected_repay / (debt_balance + 1e-6)), 0, 1)) * 100.0
    else:
        haircut_pct = 0.0

    return float(pred), float(haircut_pct), int(term_cap)

def train_if_labels(df: pd.DataFrame):
    """Εκπαίδευση μοντέλου αν υπάρχουν πραγματικές δόσεις (real_monthly)."""
    if df is None or df.empty:
        return None, None
    labeled = df.dropna(subset=["real_monthly"])
    if labeled.empty:
        return get_model(), None  # επιστρέφουμε un-fitted pipeline → θα γίνει fallback στη predict
    X_rows = []
    y = []
    for _, r in labeled.iterrows():
        try:
            debts = json.loads(r.get("debts_json") or "[]")
        except Exception:
            debts = []
        total_debt = sum([float(d.get("balance",0) or 0) for d in debts])
        secured_amt = sum([float(d.get("collateral_value",0) or 0) for d in debts if str(d.get("secured")).lower() in ["true","1","yes","ναι"]])
        extras_sum = (r.get("extra_medical") or 0)+(r.get("extra_students") or 0)+(r.get("extra_legal") or 0)
        edd_hh = (r.get("edd_manual") or 0) if (r.get("edd_use_manual")==1 or str(r.get("edd_use_manual"))=="1") \
                 else compute_edd(int(r.get("adults") or 1), int(r.get("children") or 0))
        X_rows.append(build_features_row(
            r.get("monthly_income") or 0,
            edd_hh,
            extras_sum,
            total_debt, secured_amt,
            r.get("property_value") or 0,
            r.get("annual_rate_pct") or 0,
            r.get("age_cap") or 120
        ).iloc[0].to_dict())
        y.append(float(r.get("real_monthly") or 0))
    if not X_rows:
        return get_model(), None

    X = pd.DataFrame(X_rows)
    y = np.array(y)

    model = get_model()
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        mae = float(mean_absolute_error(yte, preds))
    except Exception:
        try:
            model.fit(X, y)
            mae = None
        except Exception:
            return get_model(), None
    return model, mae

# ────────────────────────────── PDF EXPORT ──────────────────────────────
def make_pdf(case_dict:dict)->bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 2*cm
    # Logo (αν υπάρχει)
    try:
        if os.path.exists(LOGO_PATH):
            img = ImageReader(LOGO_PATH)
            c.drawImage(img, width-6*cm, y-1.5*cm, 5.2*cm, 1.5*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, y, "Bizboost - Πρόβλεψη Ρύθμισης")
    y -= 1.0*cm

    c.setFont("Helvetica", 10)
    for k,v in [
        ("Υπόθεση", case_dict.get("case_id","")),
        ("Οφειλέτης", case_dict.get("borrower","")),
        ("Ηλικία", str(case_dict.get("debtor_age",""))),
        ("Ενήλικες/Ανήλικοι", f"{case_dict.get('adults',0)}/{case_dict.get('children',0)}"),
        ("Συνολικό εισόδημα", f"{case_dict.get('monthly_income',0):,.2f} €"),
        ("ΕΔΔ νοικοκυριού", f"{case_dict.get('edd_household',0):,.2f} €"),
        ("Επιπλέον δαπάνες", f"{case_dict.get('extras_sum',0):,.2f} €"),
        ("Διαθέσιμο εισόδημα", f"{case_dict.get('avail',0):,.2f} €"),
        ("Ακίνητη περιουσία", f"{case_dict.get('property_value',0):,.2f} €"),
        ("Συνολική οφειλή", f"{case_dict.get('total_debt',0):,.2f} €"),
        ("Εξασφαλισμένα", f"{case_dict.get('secured_amt',0):,.2f} €"),
        ("Επιτόκιο (ετ.)", f"{case_dict.get('annual_rate_pct',0):,.2f}%"),
        ("Ημ/νία", case_dict.get("predicted_at","")),
    ]:
        c.drawString(2*cm, y, f"{k}: {v}")
        y -= 0.6*cm
        if y < 3*cm:
            c.showPage(); y = height - 2*cm

    debts = case_dict.get("debts", [])
    if debts:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Αναλυτικά Οφειλές:")
        y -= 0.7*cm
        c.setFont("Helvetica", 10)
        for d in debts:
            # Υπόλοιπο προς ρύθμιση σε € + %
            balance = float(d.get("balance",0) or 0)
            pm = float(d.get("predicted_monthly",0) or 0)
            term_cap = int(d.get("term_cap",0) or 0)
            expected_repay = pm * max(1,term_cap)
            writeoff_amount = max(0.0, balance - expected_repay)
            writeoff_pct = (writeoff_amount / (balance+1e-6)) * 100.0 if balance>0 else 0.0

            line1 = f"- {d.get('creditor')} | {d.get('loan_type')} | Υπόλοιπο: {balance:,.2f} €"
            line2 = f"  → Δόση: {pm:,.2f} € • Μήνες: {term_cap} • Υπόλοιπο προς ρύθμιση: {max(0.0, balance-writeoff_amount):,.2f} € • Διαγραφή: {writeoff_amount:,.2f} € ({writeoff_pct:.1f}%)"
            c.drawString(2*cm, y, line1); y -= 0.55*cm
            c.drawString(2*cm, y, line2); y -= 0.55*cm
            if y < 3*cm:
                c.showPage(); y = height - 2*cm

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ────────────────────────────── UI PAGES ──────────────────────────────
st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost")
page = st.sidebar.radio("Μενού", ["Νέα Πρόβλεψη", "Διαχείριση Δεδομένων", "Εκπαίδευση Μοντέλου"], index=0)

df_all = load_data()

# ────────────────────────────── ΝΕΑ ΠΡΟΒΛΕΨΗ ──────────────────────────────
if page == "Νέα Πρόβλεψη":
    st.title("🧮 Πρόβλεψη Ρύθμισης (Εξωδικαστικός)")

    with st.form("case_form", clear_on_submit=False):
        colA, colB, colC, colD = st.columns(4)
        borrower   = colA.text_input("Ονοματεπώνυμο / Κωδ. Υπόθεσης", "")
        debtor_age = colB.number_input("Ηλικία οφειλέτη", 18, 99, 45)
        adults     = colC.number_input("Ενήλικες στο νοικοκυριό", 1, 6, 1)
        children   = colD.number_input("Ανήλικοι στο νοικοκυριό", 0, 6, 0)

        st.markdown("### Συνολικό μηνιαίο εισόδημα (οφειλέτη + συνοφειλετών)")
        calc_from_codes = st.checkbox("Υπολογισμός από πίνακα συνοφειλετών (ετήσιο→μηνιαίο & αφαίρεση ΕΔΔ)", value=True)
        monthly_income_input = st.number_input("Μηνιαίο εισόδημα (€) [αν δεν κάνεις υπολογισμό από τον πίνακα]", 0.0, 1e9, 0.0, step=50.0)

        col1, col2, col3 = st.columns(3)
        property_value = col1.number_input("Σύνολο αξίας ακίνητης περιουσίας (€)", 0.0, 1e9, 0.0, step=1000.0)
        annual_rate_pct= col2.number_input("Επιτόκιο ετησίως (%)", 0.0, 30.0, 6.0, step=0.1)

        st.subheader("Επιπλέον Δαπάνες (πέραν ΕΔΔ)")
        c1,c2,c3 = st.columns(3)
        extra_medical = c1.number_input("Ιατρικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)
        extra_students= c2.number_input("Φοιτητές / Σπουδές (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)
        extra_legal   = c3.number_input("Δικαστικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)

        st.markdown("---")
        st.subheader("Οφειλές (ανά οφειλή)")
        default_debts = pd.DataFrame([{
            "creditor": CREDITORS[0],
            "loan_type": LOAN_TYPES[0],
            "balance": 0.0,
            "secured": False,
            "collateral_value": 0.0
        }])
        debts_df = st.data_editor(
            default_debts, num_rows="dynamic",
            column_config={
                "creditor": st.column_config.SelectboxColumn("Πιστωτής", options=CREDITORS),
                "loan_type": st.column_config.SelectboxColumn("Είδος δανείου", options=LOAN_TYPES),
                "balance": st.column_config.NumberColumn("Υπόλοιπο (€)", step=500.0, format="%.2f"),
                "secured": st.column_config.CheckboxColumn("Εξασφαλισμένο"),
                "collateral_value": st.column_config.NumberColumn("Ποσό εξασφάλισης (€)", step=500.0, format="%.2f")
            },
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Συνοφειλέτες (δομημένα πεδία)")
        st.caption("Συμπλήρωσε: Όνομα, Ετήσιο εισόδημα (σε €), Ακίνητη περιουσία (σε €), Ηλικία")
        codef_df = st.data_editor(
            pd.DataFrame([{"name":"", "annual_income":0.0, "property_value":0.0, "age":40}]),
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Όνομα"),
                "annual_income": st.column_config.NumberColumn("Ετήσιο εισόδημα (€)", step=100.0, format="%.2f"),
                "property_value": st.column_config.NumberColumn("Ακίνητη περιουσία (€)", step=1000.0, format="%.2f"),
                "age": st.column_config.NumberColumn("Ηλικία", min_value=18, max_value=99, step=1),
            },
            use_container_width=True
        )

        st.subheader("Ελάχιστες Δαπάνες Διαβίωσης (ΕΔΔ)")
        use_manual = st.checkbox("Χειροκίνητη εισαγωγή ΕΔΔ;", value=False)
        if use_manual:
            edd_val = st.number_input("ΕΔΔ νοικοκυριού (€ / μήνα)", 0.0, 10000.0, 800.0, step=10.0)
        else:
            edd_val = compute_edd(int(adults), int(children))
            st.info(f"Αυτόματος υπολογισμός ΕΔΔ: **{edd_val:,.2f} €**")

        submitted = st.form_submit_button("Υπολογισμός Πρόβλεψης & Αποθήκευση", use_container_width=True)

    if submitted:
        # Συνυπολογισμός συνοφειλετών: μετατροπή ετήσιων εισοδημάτων σε καθαρά μηνιαία (μετά ΕΔΔ)
        codebtors = codef_df.fillna(0).to_dict(orient="records") if codef_df is not None else []
        # Για απλότητα: ΕΔΔ ανά συνοφειλέτη ως 1 ενήλικας χωρίς παιδιά
        def edd_single_adult():
            return compute_edd(1, 0)

        monthly_income_from_codes = 0.0
        for co in codebtors:
            ann = float(co.get("annual_income",0) or 0.0)
            mon_gross = ann/12.0
            mon_net = max(0.0, mon_gross - edd_single_adult())
            monthly_income_from_codes += mon_net

        monthly_income = monthly_income_from_codes if calc_from_codes else monthly_income_input

        # Συγκεντρωτικά από οφειλές (για header/PDF)
        debts = debts_df.fillna(0).to_dict(orient="records")
        total_debt = sum([float(d["balance"] or 0) for d in debts])
        secured_amt = sum([float(d["collateral_value"] or 0) for d in debts if d.get("secured")])
        extras_sum = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
        avail = available_income(monthly_income, edd_val, extra_medical, extra_students, extra_legal)
        age_cap_months = months_cap_from_age(int(debtor_age))

        # Εκπαίδευση μοντέλου (αν υπάρχουν labels)
        model, mae = train_if_labels(df_all)

        # ── Υπολογισμός ΑΝΑ ΟΦΕΙΛΗ ──
        per_debt_rows = []
        for d in debts:
            creditor = str(d.get("creditor", "")).strip()
            balance  = float(d.get("balance", 0) or 0)
            is_sec   = bool(d.get("secured"))
            coll_val = float(d.get("collateral_value", 0) or 0)

            pred_m, hair_pct, term_cap_single = predict_single_debt_monthly(
                model=model,
                monthly_income=monthly_income,
                edd_val=edd_val,
                extras_sum=extras_sum,
                debt_balance=balance,
                debt_secured_amt=(coll_val if is_sec else 0.0),
                property_value=property_value,
                annual_rate_pct=annual_rate_pct,
                age_cap_months=age_cap_months,
                creditor_name=creditor
            )

            d["predicted_monthly"] = round(pred_m, 2)
            d["predicted_haircut_pct"] = round(hair_pct, 2)
            d["term_cap"] = int(term_cap_single)

            # Επιπλέον πεδία για αναφορά: υπόλοιπο προς ρύθμιση & διαγραφή
            expected_repay = pred_m * term_cap_single
            writeoff_amount = max(0.0, balance - expected_repay)
            remaining_after_writeoff = max(0.0, balance - writeoff_amount)
            writeoff_pct = (writeoff_amount / (balance+1e-6)) * 100.0 if balance>0 else 0.0

            per_debt_rows.append({
                "Πιστωτής": creditor,
                "Είδος": d.get("loan_type",""),
                "Υπόλοιπο (€)": balance,
                "Εξασφαλισμένο": "Ναι" if is_sec else "Όχι",
                "Εξασφάλιση (€)": coll_val if is_sec else 0.0,
                "Οροφή μηνών": term_cap_single,
                "Πρόταση δόσης (€)": round(pred_m, 2),
                "Υπόλοιπο προς ρύθμιση (€)": round(remaining_after_writeoff, 2),
                "Ποσό διαγραφής (€)": round(writeoff_amount, 2),
                "Διαγραφή (%)": round(writeoff_pct, 2),
            })

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("Οι προτάσεις δίνονται **ανά οφειλή** (δεν γίνεται συνολική άθροιση).")

        # Αποθήκευση υπόθεσης (χωρίς συνολικό predicted στο header)
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        row = {
            "case_id": case_id,
            "borrower": borrower,
            "debtor_age": int(debtor_age),
            "adults": int(adults),
            "children": int(children),

            "monthly_income": float(monthly_income),
            "property_value": float(property_value),
            "annual_rate_pct": float(annual_rate_pct),

            "edd_use_manual": 1 if use_manual else 0,
            "edd_manual": float(edd_val),
            "extra_medical": float(extra_medical or 0),
            "extra_students": float(extra_students or 0),
            "extra_legal": float(extra_legal or 0),

            "age_cap": int(age_cap_months),

            "debts_json": json.dumps(debts, ensure_ascii=False),
            "co_debtors_json": json.dumps(codebtors, ensure_ascii=False),

            "term_months": None,
            "predicted_at": now_str,
            "predicted_monthly": None,
            "predicted_haircut_pct": None,
            "prob_accept": None,

            "real_monthly": None,
            "real_haircut_pct": None,
            "accepted": None,
            "real_term_months": None,
            "real_writeoff_amount": None,
            "real_residual_balance": None
        }
        save_data(pd.DataFrame([row]))
        st.success("✅ Αποθήκευση ολοκληρώθηκε.")

        # PDF export
        case_for_pdf = dict(row)
        case_for_pdf["edd_household"] = float(edd_val)
        case_for_pdf["extras_sum"] = float((extra_medical or 0) + (extra_students or 0) + (extra_legal or 0))
        case_for_pdf["avail"] = float(avail)
        case_for_pdf["total_debt"] = float(total_debt)
        case_for_pdf["secured_amt"] = float(secured_amt)
        case_for_pdf["debts"] = debts
        pdf_bytes = make_pdf(case_for_pdf)
        st.download_button("⬇️ Λήψη Πρόβλεψης (PDF)", data=pdf_bytes, file_name=f"{case_id}_prediction.pdf", mime="application/pdf", use_container_width=True)

        if mae is not None:
            st.caption(f"MAE μοντέλου (εκπαιδεύτηκε στα ιστορικά): ~{mae:,.2f} €/μήνα")

# ────────────────────────────── ΔΙΑΧΕΙΡΙΣΗ ΔΕΔΟΜΕΝΩΝ ──────────────────────────────
elif page == "Διαχείριση Δεδομένων":
    st.title("📚 Διαχείριση Δεδομένων")
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
    else:
        st.dataframe(df_all.sort_values("predicted_at", ascending=False), use_container_width=True)

        with st.expander("Ενημέρωση με πραγματική ρύθμιση (μαθαίνει το ML)"):
            case_ids = df_all["case_id"].tolist()
            case_pick = st.selectbox("Διάλεξε Υπόθεση", case_ids)
            row = df_all[df_all["case_id"]==case_pick].iloc[0].to_dict()

            c1,c2,c3 = st.columns(3)
            real_monthly = c1.number_input("Πραγματική μηνιαία δόση (€)", 0.0, 1e7, float(row.get("real_monthly") or 0.0), step=10.0)
            real_term    = c2.number_input("Πραγματικοί μήνες", 0, 1200, int(row.get("real_term_months") or row.get("term_months") or 0))
            real_writeoff= c3.number_input("Ποσό διαγραφής (€)", 0.0, 1e10, float(row.get("real_writeoff_amount") or 0.0), step=100.0)

            r1,r2 = st.columns(2)
            real_residual = r1.number_input("Υπόλοιπο προς ρύθμιση (€)", 0.0, 1e12, float(row.get("real_residual_balance") or 0.0), step=100.0)
            accepted      = r2.selectbox("Έγινε αποδεκτή;", ["Άγνωστο","Ναι","Όχι"], index=0)

            # Υπολογισμός πραγματικής % διαγραφής αν υπάρχει συνολική οφειλή
            try:
                debts = json.loads(row.get("debts_json") or "[]")
                total_debt = sum([float(d.get("balance",0) or 0) for d in debts])
            except Exception:
                total_debt = 0.0
            real_haircut_pct = 100.0 * (float(real_writeoff or 0) / (total_debt+1e-6)) if total_debt>0 else None

            if st.button("Αποθήκευση πραγματικής ρύθμισης", type="primary"):
                row_update = row.copy()
                row_update.update({
                    "real_monthly": float(real_monthly) if real_monthly else None,
                    "real_term_months": int(real_term) if real_term else None,
                    "real_writeoff_amount": float(real_writeoff) if real_writeoff else None,
                    "real_residual_balance": float(real_residual) if real_residual else None,
                    "real_haircut_pct": float(real_haircut_pct) if real_haircut_pct is not None else None,
                    "accepted": None if accepted=="Άγνωστο" else (1 if accepted=="Ναι" else 0)
                })
                save_data(pd.DataFrame([row_update]))
                st.success("✅ Ενημερώθηκε. Το μοντέλο θα μάθει από τα νέα δεδομένα στην επόμενη εκπαίδευση.")

# ────────────────────────────── ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ ──────────────────────────────
else:
    st.title("🤖 Εκπαίδευση & Απόδοση Μοντέλου")
    if df_all.empty or df_all.dropna(subset=["real_monthly"]).empty:
        st.info("Χρειάζονται αποθηκευμένες υποθέσεις με πραγματικές ρυθμίσεις για εκπαίδευση.")
    else:
        with st.spinner("Εκπαίδευση..."):
            model, mae = train_if_labels(df_all)
        if model is None:
            st.warning("Δεν επαρκούν δεδομένα για εκπαίδευση.")
        else:
            st.success("Το μοντέλο εκπαιδεύτηκε.")
            if mae is not None:
                st.metric("MAE (€/μήνα)", f"{mae:,.2f}")
            st.caption("Το μοντέλο χρησιμοποιείται αυτόματα στις νέες προβλέψεις.")
