# app.py
# Bizboost - Πρόβλεψη Ρυθμίσεων Εξωδικαστικού

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# Προαιρετικό ML: αν λείπει xgboost, τρέχουμε fallback
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ───────────────── UI / σταθερές ─────────────────
st.set_page_config(page_title="Bizboost - Πρόβλεψη Ρυθμίσεων", page_icon="💠", layout="wide")
LOGO_PATH = "logo.png"
DATA_CSV  = "cases.csv"

CREDITORS = [
    "DoValue","Intrum","Cepal","Qquant","APS","EOS","Veralitis",
    "Πειραιώς","Εθνική","Eurobank","Alpha",
    "ΑΑΔΕ","ΕΦΚΑ",
]
LOAN_TYPES = ["Στεγαστικό","Καταναλωτικό","Επαγγελματικό"]

PUBLIC_CREDITORS = {"ΑΑΔΕ","ΕΦΚΑ"}
BANK_SERVICERS = set(CREDITORS) - PUBLIC_CREDITORS

def term_cap_for_single_debt(creditor_name: str, age_cap_months: int) -> int:
    c = (creditor_name or "").strip()
    if c in BANK_SERVICERS:
        policy_cap = 420
    elif c in PUBLIC_CREDITORS:
        policy_cap = 240
    else:
        policy_cap = 240
    return max(1, min(policy_cap, int(age_cap_months or 120)))

# ───────────────── ΕΔΔ / βοηθητικά ─────────────────
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

# ───────────────── Βάση δεδομένων ─────────────────
def get_db_engine():
    db_url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL",""))
    if not db_url:
        st.error("Δεν έχει οριστεί DATABASE_URL στα Secrets.")
        st.stop()
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(db_url, pool_pre_ping=True)

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
    with get_db_engine().begin() as conn:
        conn.execute(text(ddl))

def load_data_db()->pd.DataFrame:
    init_db(get_db_engine())
    try:
        return pd.read_sql("SELECT * FROM cases", con=get_db_engine())
    except Exception as e:
        st.error(f"Σφάλμα ανάγνωσης DB: {e}")
        return pd.DataFrame()

def upsert_cases_db(df: pd.DataFrame):
    if df.empty:
        return
    init_db(get_db_engine())
    cols = [
        "case_id","borrower","debtor_age","adults","children","monthly_income","property_value",
        "annual_rate_pct","edd_use_manual","edd_manual","extra_medical","extra_students","extra_legal",
        "age_cap","debts_json","co_debtors_json","term_months","predicted_at",
        "predicted_monthly","predicted_haircut_pct","prob_accept",
        "real_monthly","real_haircut_pct","accepted","real_term_months",
        "real_writeoff_amount","real_residual_balance"
    ]
    df2 = df.copy()
    for c in ["debts_json","co_debtors_json"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: x if isinstance(x,str) else json.dumps(x, ensure_ascii=False))
    df2 = df2.reindex(columns=cols, fill_value=np.nan)
    sql = f"""
    INSERT INTO cases ({",".join(cols)})
    VALUES ({",".join([f":{c}" for c in cols])})
    ON CONFLICT (case_id) DO UPDATE SET
      {",".join([f"{c}=EXCLUDED.{c}" for c in cols if c!="case_id"])};
    """
    with get_db_engine().begin() as conn:
        conn.execute(text(sql), df2.to_dict(orient="records"))

def csv_to_db_once_if_empty():
    init_db(get_db_engine())
    with get_db_engine().begin() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM cases")).scalar()
    if cnt == 0 and os.path.exists(DATA_CSV):
        try:
            dfcsv = pd.read_csv(DATA_CSV)
            for col in ["debts_json","co_debtors_json"]:
                if col in dfcsv.columns:
                    dfcsv[col] = dfcsv[col].where(dfcsv[col].notna(), "[]")
            upsert_cases_db(dfcsv)
            st.success("Έγινε αρχικό import από cases.csv")
        except Exception as e:
            st.warning(f"Αποτυχία import από cases.csv: {e}")

def load_data():
    csv_to_db_once_if_empty()
    return load_data_db()

def save_data(df: pd.DataFrame):
    upsert_cases_db(df)

# ───────────────── ML ─────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    if XGB_OK:
        return Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("xgb", XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, objective="reg:squarederror",
                random_state=42, n_jobs=2
            ))
        ])
    else:
        return Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", Ridge(alpha=1.0, random_state=42))
        ])

def build_features_row(total_income, edd_household, extras_sum, total_debt, secured_amt, property_value, rate_pct, term_cap):
    avail = max(0.0, (total_income or 0) - (edd_household or 0) - (extras_sum or 0))
    debt_to_income = (total_debt or 0) / (total_income+1e-6)
    secured_ratio   = (secured_amt or 0) / (total_debt+1e-6)
    ltv             = (total_debt or 0) / (property_value+1e-6)
    return pd.DataFrame([{
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
    # Ασφαλής πρόβλεψη (fallback αν το μοντέλο δεν είναι fitted)
    try:
        pred = float(model.predict(Xd)[0])
        if not np.isfinite(pred) or pred < 0:
            raise ValueError("non-finite pred")
    except Exception:
        avail = max(0.0, monthly_income - edd_val - extras_sum)
        pred = round(avail * 0.7, 2)

    if debt_balance > 0:
        expected_repay = pred * term_cap
        haircut_pct = float(np.clip(1 - (expected_repay / (debt_balance + 1e-6)), 0, 1)) * 100.0
    else:
        haircut_pct = 0.0
    return pred, haircut_pct, term_cap

def train_if_labels(df: pd.DataFrame):
    labeled = df.dropna(subset=["real_monthly"])
    if labeled.empty:
        # ΠΟΛΥ ΣΗΜΑΝΤΙΚΟ: επιστρέφουμε None για να ενεργοποιηθεί το fallback
        return None, None
    X, y = [], []
    for _, r in labeled.iterrows():
        debts = json.loads(r.get("debts_json") or "[]")
        total_debt = sum(float(d.get("balance",0) or 0) for d in debts)
        secured_amt = sum(float(d.get("collateral_value",0) or 0) for d in debts if d.get("secured") in [True,"True","true",1])
        extras_sum = (r.get("extra_medical") or 0)+(r.get("extra_students") or 0)+(r.get("extra_legal") or 0)
        X.append(build_features_row(
            r.get("monthly_income") or 0,
            r.get("edd_manual") if r.get("edd_use_manual")==1 else compute_edd(int(r.get("adults") or 1), int(r.get("children") or 0)),
            extras_sum,
            total_debt, secured_amt,
            r.get("property_value") or 0,
            r.get("annual_rate_pct") or 0,
            r.get("age_cap") or 120
        ).iloc[0].to_dict())
        y.append(float(r.get("real_monthly") or 0))
    X = pd.DataFrame(X); y = np.array(y)
    model = get_model()
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        mae = float(mean_absolute_error(yte, preds))
    except Exception:
        model.fit(X, y)
        mae = None
    return model, mae

# ───────────────── PDF ─────────────────
def make_pdf(case_dict:dict)->bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 2*cm
    try:
        if os.path.exists(LOGO_PATH):
            img = ImageReader(LOGO_PATH)
            c.drawImage(img, width-6*cm, y-1.5*cm, 5.2*cm, 1.5*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass
    c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, y, "Bizboost - Πρόβλεψη Ρύθμισης"); y -= 1.0*cm
    c.setFont("Helvetica", 10)
    for k,v in [
        ("Υπόθεση", case_dict.get("case_id","")),
        ("Οφειλέτης", case_dict.get("borrower","")),
        ("Ηλικία", str(case_dict.get("debtor_age",""))),
        ("Ενήλικες/Ανήλικοι", f"{case_dict.get('adults',0)}/{case_dict.get('children',0)}"),
        ("Συνολικό μηνιαίο εισόδημα", f"{case_dict.get('monthly_income',0):,.2f} €"),
        ("ΕΔΔ νοικοκυριού", f"{case_dict.get('edd_household',0):,.2f} €"),
        ("Επιπλέον δαπάνες", f"{case_dict.get('extras_sum',0):,.2f} €"),
        ("Διαθέσιμο εισόδημα", f"{case_dict.get('avail',0):,.2f} €"),
        ("Σύνολο περιουσίας", f"{case_dict.get('property_value',0):,.2f} €"),
        ("Επιτόκιο (ετ.)", f"{case_dict.get('annual_rate_pct',0):,.2f}%"),
        ("Ημ/νία", case_dict.get("predicted_at","")),
    ]:
        c.drawString(2*cm, y, f"{k}: {v}"); y -= 0.6*cm
        if y < 3*cm: c.showPage(); y = height - 2*cm
    debts = case_dict.get("debts", [])
    if debts:
        c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Αναλυτικά Οφειλές:"); y -= 0.7*cm
        c.setFont("Helvetica", 10)
        for d in debts:
            line = f"- {d.get('creditor')} | {d.get('loan_type')} | Υπόλοιπο: {float(d.get('balance',0)):,.2f} € | " \
                   f"Εξασφαλισμένο: {'Ναι' if d.get('secured') else 'Όχι'}"
            if d.get("secured"):
                line += f" (Εξασφάλιση: {float(d.get('collateral_value',0)):,.2f} €)"
            line += f" | Οροφή μηνών: {d.get('term_cap','-')} | Δόση: {d.get('predicted_monthly','-')} € | Διαγραφή: {d.get('predicted_haircut_pct','-')}%"
            c.drawString(2*cm, y, line); y -= 0.55*cm
            if y < 3*cm: c.showPage(); y = height - 2*cm
    c.showPage(); c.save(); buf.seek(0); return buf.read()

# ───────────────── UI ─────────────────
st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost")
page = st.sidebar.radio("Μενού", ["Νέα Πρόβλεψη", "Διαχείριση Δεδομένων", "Εκπαίδευση Μοντέλου"], index=0)

df_all = load_data()

if page == "Νέα Πρόβλεψη":
    st.title("🧮 Πρόβλεψη Ρύθμισης (Εξωδικαστικός)")

    with st.form("case_form", clear_on_submit=False):
        colA, colB, colC, colD = st.columns(4)
        borrower   = colA.text_input("Ονοματεπώνυμο / Κωδ. Υπόθεσης", "", key="borrower")
        debtor_age = colB.number_input("Ηλικία οφειλέτη", 18, 99, 45, key="debtor_age")
        adults     = colC.number_input("Ενήλικες στο νοικοκυριό", 1, 6, 1, key="adults")
        children   = colD.number_input("Ανήλικοι στο νοικοκυριό", 0, 6, 0, key="children")

        col1, col2, col3 = st.columns(3)
        debtor_monthly_income = col1.number_input("Μηνιαίο εισόδημα οφειλέτη (€ / μήνα)", 0.0, 100000.0, 1200.0, step=50.0, key="income_main")
        property_value_debtor = col2.number_input("Ακίνητη περιουσία οφειλέτη (€)", 0.0, 1e9, 0.0, step=1000.0, key="prop_main")
        annual_rate_pct = col3.number_input("Επιτόκιο ετησίως (%)", 0.0, 30.0, 6.0, step=0.1, key="rate")

        st.subheader("Συνοφειλέτες")
        st.caption("Συμπλήρωσε ετήσιο εισόδημα – το σύστημα το μετατρέπει σε μηνιαίο και αφαιρεί ΕΔΔ 537€/ενήλικα για να βρει το καθαρό διαθέσιμο.")
        co_df = st.data_editor(
            pd.DataFrame([{"name":"", "income_year":0.0, "property_value":0.0, "age":45}]),
            num_rows="dynamic",
            key="co_edit",
            column_config={
                "name": st.column_config.TextColumn("Όνομα"),
                "income_year": st.column_config.NumberColumn("Ετήσιο εισόδημα (€)", step=500.0, format="%.2f"),
                "property_value": st.column_config.NumberColumn("Ακίνητη περιουσία (€)", step=1000.0, format="%.2f"),
                "age": st.column_config.NumberColumn("Ηλικία", step=1)
            },
            use_container_width=True,
        )

        st.subheader("Επιπλέον Δαπάνες (πέραν ΕΔΔ νοικοκυριού)")
        c1,c2,c3 = st.columns(3)
        extra_medical  = c1.number_input("Ιατρικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0, key="x_med")
        extra_students = c2.number_input("Φοιτητές / Σπουδές (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0, key="x_stu")
        extra_legal    = c3.number_input("Δικαστικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0, key="x_leg")

        st.markdown("---")
        st.subheader("Οφειλές")
        debts_df = st.data_editor(
            pd.DataFrame([{
                "creditor": CREDITORS[0],
                "loan_type": LOAN_TYPES[0],
                "balance": 0.0,
                "secured": False,
                "collateral_value": 0.0
            }]),
            num_rows="dynamic",
            key="debts_editor",
            column_config={
                "creditor": st.column_config.SelectboxColumn("Πιστωτής", options=CREDITORS),
                "loan_type": st.column_config.SelectboxColumn("Είδος δανείου", options=LOAN_TYPES),
                "balance": st.column_config.NumberColumn("Υπόλοιπο (€)", step=500.0, format="%.2f"),
                "secured": st.column_config.CheckboxColumn("Εξασφαλισμένο"),
                "collateral_value": st.column_config.NumberColumn("Ποσό εξασφάλισης (€)", step=500.0, format="%.2f")
            },
            use_container_width=True
        )

        st.subheader("Ελάχιστες Δαπάνες Διαβίωσης (νοικοκυριού)")
        use_manual = st.checkbox("Χειροκίνητη εισαγωγή ΕΔΔ;", value=False, key="edd_manual_chk")
        if use_manual:
            edd_val = st.number_input("ΕΔΔ νοικοκυριού (€ / μήνα)", 0.0, 10000.0, 800.0, step=10.0, key="edd_val_manual")
        else:
            edd_val = compute_edd(int(adults), int(children))
            st.info(f"Αυτόματος υπολογισμός ΕΔΔ νοικοκυριού: **{edd_val:,.2f} €**")

        submitted = st.form_submit_button("Υπολογισμός Πρόβλεψης & Αποθήκευση", use_container_width=True)

    if submitted:
        # Συνοφειλέτες -> λίστα dict
        co_list = []
        if isinstance(co_df, pd.DataFrame):
            co_list = co_df.fillna(0).to_dict(orient="records")

        # Μηνιαία από συνοφειλέτες (μετατροπή από ετήσιο) + διαθέσιμο τους (αφαιρώντας 537€)
        co_monthly_income = sum((float(c.get("income_year",0) or 0)/12.0) for c in co_list)
        co_avail_total = sum(max(0.0, (float(c.get("income_year",0) or 0)/12.0) - 537.0) for c in co_list)
        co_property_sum = sum(float(c.get("property_value",0) or 0) for c in co_list)

        # Συγκεντρωτικά οφειλών
        debts = debts_df.fillna(0).to_dict(orient="records")
        total_debt = sum(float(d["balance"] or 0) for d in debts)
        secured_amt = sum(float(d["collateral_value"] or 0) for d in debts if d.get("secured"))

        # Συνολικό μηνιαίο εισόδημα: οφειλέτης + συνοφειλέτες
        monthly_income_total = float(debtor_monthly_income) + float(co_monthly_income)

        # Συνολική περιουσία: οφειλέτης + συνοφειλέτες
        property_value_total = float(property_value_debtor) + float(co_property_sum)

        extras_sum = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)

        # Διαθέσιμο: (οφειλέτης - ΕΔΔ νοικοκυριού - έξτρα) + διαθέσιμο συνοφειλετών
        debtor_avail = max(0.0, float(debtor_monthly_income) - float(edd_val) - float(extras_sum))
        avail_total = debtor_avail + co_avail_total

        age_cap_months = months_cap_from_age(int(debtor_age))

        # Εκπαίδευση μοντέλου (αν υπάρχουν labels)
        model, mae = train_if_labels(df_all)
        if model is None:
            model = get_model()  # άφτιαχτο, αλλά το predict_single_debt_monthly έχει fallback

        # ── Πρόβλεψη ΑΝΑ οφειλή ──
        per_debt_rows = []
        for d in debts:
            creditor = str(d.get("creditor", "")).strip()
            balance  = float(d.get("balance", 0) or 0)
            is_sec   = bool(d.get("secured"))
            coll_val = float(d.get("collateral_value", 0) or 0)

            pred_m, hair_pct, term_cap_single = predict_single_debt_monthly(
                model=model,
                monthly_income=monthly_income_total,
                edd_val=edd_val,
                extras_sum=extras_sum,
                debt_balance=balance,
                debt_secured_amt=(coll_val if is_sec else 0.0),
                property_value=property_value_total,
                annual_rate_pct=annual_rate_pct,
                age_cap_months=age_cap_months,
                creditor_name=creditor
            )

            d["predicted_monthly"] = round(pred_m, 2)
            d["predicted_haircut_pct"] = round(hair_pct, 2)
            d["term_cap"] = int(term_cap_single)

            per_debt_rows.append({
                "Πιστωτής": creditor,
                "Είδος": d.get("loan_type",""),
                "Υπόλοιπο (€)": balance,
                "Εξασφαλισμένο": "Ναι" if is_sec else "Όχι",
                "Εξασφάλιση (€)": coll_val if is_sec else 0.0,
                "Οροφή μηνών": term_cap_single,
                "Πρόταση δόσης (€)": round(pred_m, 2),
                "Πρόταση διαγραφής (%)": round(hair_pct, 2),
            })

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("Οι προτάσεις δίνονται **ανά οφειλή** (δεν γίνεται άθροιση).")

        # Αποθήκευση
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        row = {
            "case_id": case_id,
            "borrower": borrower,
            "debtor_age": int(debtor_age),
            "adults": int(adults),
            "children": int(children),

            # Σώζουμε το ΣΥΝΟΛΙΚΟ μηνιαίο εισόδημα (οφειλέτη + συνοφειλετών)
            "monthly_income": float(monthly_income_total),

            # Σώζουμε τη ΣΥΝΟΛΙΚΗ περιουσία (οφειλέτη + συνοφειλετών)
            "property_value": float(property_value_total),

            "annual_rate_pct": float(annual_rate_pct),
            "edd_use_manual": 1 if use_manual else 0,
            "edd_manual": float(edd_val),
            "extra_medical": float(extra_medical or 0),
            "extra_students": float(extra_students or 0),
            "extra_legal": float(extra_legal or 0),
            "age_cap": int(age_cap_months),

            "debts_json": json.dumps(debts, ensure_ascii=False),
            "co_debtors_json": json.dumps(co_list, ensure_ascii=False),

            # Χωρίς συνολικές προτάσεις στην υπόθεση
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

        # PDF
        case_for_pdf = dict(row)
        case_for_pdf["edd_household"] = float(edd_val)
        case_for_pdf["extras_sum"] = float(extras_sum)
        case_for_pdf["avail"] = float(avail_total)
        case_for_pdf["debts"] = debts
        pdf_bytes = make_pdf(case_for_pdf)
        st.download_button("⬇️ Λήψη Πρόβλεψης (PDF)", data=pdf_bytes,
                           file_name=f"{case_id}_prediction.pdf", mime="application/pdf",
                           use_container_width=True)

        if mae is not None:
            st.caption(f"MAE μοντέλου (εκπαιδεύτηκε στα ιστορικά): ~{mae:,.2f} €/μήνα")

elif page == "Διαχείριση Δεδομένων":
    st.title("📚 Διαχείριση Δεδομένων")
    df = df_all.copy()
    if df.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
    else:
        st.dataframe(df.sort_values("predicted_at", ascending=False, na_position="last"),
                     use_container_width=True)
        with st.expander("Ενημέρωση με πραγματική ρύθμιση (μαθαίνει το ML)"):
            case_ids = df["case_id"].tolist()
            if case_ids:
                case_pick = st.selectbox("Διάλεξε Υπόθεση", case_ids)
                row = df[df["case_id"]==case_pick].iloc[0].to_dict()
                c1,c2,c3 = st.columns(3)
                real_monthly = c1.number_input("Πραγματική μηνιαία δόση (€)", 0.0, 1e7, float(row.get("real_monthly") or 0.0), step=10.0)
                real_term    = c2.number_input("Πραγματικοί μήνες", 0, 1200, int(row.get("real_term_months") or row.get("term_months") or 0))
                real_writeoff= c3.number_input("Ποσό διαγραφής (€)", 0.0, 1e10, float(row.get("real_writeoff_amount") or 0.0), step=100.0)
                r1,r2 = st.columns(2)
                real_residual = r1.number_input("Υπόλοιπο προς ρύθμιση (€)", 0.0, 1e12, float(row.get("real_residual_balance") or 0.0), step=100.0)
                accepted      = r2.selectbox("Έγινε αποδεκτή;", ["Άγνωστο","Ναι","Όχι"], index=0)

                try:
                    debts = json.loads(row.get("debts_json") or "[]")
                    total_debt = sum(float(d.get("balance",0) or 0) for d in debts)
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
                    st.success("✅ Ενημερώθηκε. Το μοντέλο θα μάθει στα επόμενα run.")
            else:
                st.info("Δεν βρέθηκαν υποθέσεις.")

else:
    st.title("🤖 Εκπαίδευση & Απόδοση Μοντέλου")
    if df_all.empty or df_all.dropna(subset=["real_monthly"]).empty:
        st.info("Χρειάζονται υποθέσεις με πραγματικές ρυθμίσεις για εκπαίδευση.")
    else:
        with st.spinner("Εκπαίδευση..."):
            model, mae = train_if_labels(df_all)
        if mae is None:
            st.warning("Το μοντέλο εκπαιδεύτηκε χωρίς test split.")
        else:
            st.success("Το μοντέλο εκπαιδεύτηκε.")
            st.metric("MAE (€/μήνα)", f"{mae:,.2f}")
        st.caption("Το μοντέλο χρησιμοποιείται αυτόματα στις νέες προβλέψεις.")
