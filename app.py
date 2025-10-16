# app.py
# Bizboost – Πρόβλεψη Ρυθμίσεων Εξωδικαστικού (ελληνικό UI)
# - Postgres (Supabase) μέσω SQLAlchemy + psycopg
# - Πρόβλεψη ΑΝΑ ΟΦΕΙΛΗ (χωρίς άθροιση), με κόφτες 420/240 ανά πιστωτή και ηλικιακό κόφτη
# - Συνοφειλέτες με ετήσιο εισόδημα -> μηνιαίο, αφαίρεση ΕΔΔ ανά άτομο
# - Αποθήκευση πρόβλεψης & καταχώρηση πραγματικής ρύθμισης ΑΝΑ ΟΦΕΙΛΗ
# - PDF με ελληνικά (DejaVuSans), logo.png, καθαρή εμφάνιση
# - ML (LinearRegression) αν υπάρχουν labels, αλλιώς ασφαλής κανόνας fallback

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# Προαιρετικό, ελαφρύ ML
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ────────────────────────────── UI/Paths ──────────────────────────────
st.set_page_config(page_title="Bizboost - Πρόβλεψη Ρυθμίσεων", page_icon="💠", layout="wide")

LOGO_PATH = "logo.png"
FONT_PATH = "assets/fonts/DejaVuSans.ttf"  # για ελληνικά
DATA_CSV  = "cases.csv"                    # προαιρετικό αρχικό import

# Εγγραφή γραμματοσειράς για ελληνικά στο PDF
if os.path.exists(FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))
        PDF_FONT = "DejaVu"
    except Exception:
        PDF_FONT = "Helvetica"
else:
    PDF_FONT = "Helvetica"

# ─────────────────────────── Κατάλογοι επιλογών ───────────────────────────
CREDITORS = [
    # Servicers/Τράπεζες (420 μήνες)
    "DoValue","Intrum","Cepal","Qquant","APS","EOS","Veralitis",
    "Πειραιώς","Εθνική","Eurobank","Alpha",
    # Δημόσιο (240 μήνες)
    "ΑΑΔΕ","ΕΦΚΑ"
]
LOAN_TYPES = ["Στεγαστικό","Καταναλωτικό","Επαγγελματικό"]

PUBLIC_CREDITORS = {"ΑΑΔΕ","ΕΦΚΑ"}
BANK_SERVICERS = set(CREDITORS) - PUBLIC_CREDITORS

# ─────────────────────────── Βοηθητικές συναρτήσεις ───────────────────────────
def term_cap_for_single_debt(creditor_name: str, age_cap_months: int) -> int:
    """Οροφή μηνών βάσει κατηγορίας πιστωτή + κόφτης ηλικίας."""
    c = (creditor_name or "").strip()
    if c in BANK_SERVICERS:
        policy_cap = 420
    elif c in PUBLIC_CREDITORS:
        policy_cap = 240
    else:
        policy_cap = 240
    return max(1, min(policy_cap, int(age_cap_months or 120)))

def compute_edd(adults:int, children:int)->float:
    """Απλή κλίμακα ΕΔΔ: 1 ενήλικας 537€, επιπλέον ενήλικας +269€, ανήλικος +211€."""
    try:
        adults = int(adults)
        children = int(children)
    except Exception:
        return 0.0
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
    """Ηλικιακός κόφτης μηνών (προσεγγιστικός, προσαρμόσιμος)."""
    try:
        a = int(age)
    except Exception:
        return 120
    if a <= 35:  return 240
    if a <= 50:  return 180
    if a <= 65:  return 120
    return 60

def available_income(total_monthly_income: float, edd_household: float, extras_sum: float)->float:
    """Διαθέσιμο = Συνολικό μηνιαίο εισόδημα - ΕΔΔ - (ιατρικά+φοιτητές+δικαστικά)."""
    return max(0.0, float(total_monthly_income or 0) - float(edd_household or 0) - float(extras_sum or 0))

# ─────────────────────────────── Βάση Δεδομένων ───────────────────────────────
def get_db_engine():
    db_url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL",""))
    if not db_url:
        st.error("Δεν βρέθηκε DATABASE_URL στα Secrets.")
        st.stop()
    # χρήση psycopg3 driver name
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
      debts_json JSONB,            -- λίστα οφειλών με προβλέψεις ανα οφειλή
      co_debtors_json JSONB,       -- λίστα συνοφειλετών
      real_debts_json JSONB,       -- λίστα πραγματικών ρυθμίσεων ανα οφειλή
      term_months INT,             -- (όχι σε χρήση για πρόβλεψη ανα οφειλή)
      predicted_at TEXT,
      predicted_monthly NUMERIC,   -- (όχι σε χρήση)
      predicted_haircut_pct NUMERIC, -- (όχι σε χρήση)
      prob_accept NUMERIC,         -- (όχι σε χρήση)
      real_monthly NUMERIC,        -- (όχι σε χρήση)
      real_haircut_pct NUMERIC,    -- (όχι σε χρήση)
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

def upsert_cases_db(df: pd.DataFrame):
    if df.empty:
        return
    engine = get_db_engine()
    init_db(engine)

    cols = [
        "case_id","borrower","debtor_age","adults","children",
        "monthly_income","property_value","annual_rate_pct",
        "edd_use_manual","edd_manual","extra_medical","extra_students","extra_legal",
        "age_cap","debts_json","co_debtors_json","real_debts_json",
        "term_months","predicted_at","predicted_monthly","predicted_haircut_pct","prob_accept",
        "real_monthly","real_haircut_pct","accepted","real_term_months",
        "real_writeoff_amount","real_residual_balance"
    ]
    df2 = df.copy()
    for c in ["debts_json","co_debtors_json","real_debts_json"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: x if isinstance(x,str) else json.dumps(x or [], ensure_ascii=False))
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
            # καθάρισε NaN JSON πεδία
            for jc in ["debts_json","co_debtors_json","real_debts_json"]:
                if jc in dfcsv.columns:
                    dfcsv[jc] = dfcsv[jc].fillna("[]")
            upsert_cases_db(dfcsv)
            st.success("Έγινε αρχικό import από cases.csv")
        except Exception as e:
            st.warning(f"Αποτυχία import από cases.csv: {e}")

def load_data():
    csv_to_db_once_if_empty()
    return load_data_db()

def save_data(df: pd.DataFrame):
    upsert_cases_db(df)

# ─────────────────────────────── «ML» (ελαφρύ/προαιρετικό) ───────────────────────────────
def simple_features_row(total_income, edd_household, extras_sum, debt_balance, secured_amt, property_value, rate_pct, term_cap):
    avail = max(0.0, (total_income or 0) - (edd_household or 0) - (extras_sum or 0))
    dti = (debt_balance or 0) / (total_income + 1e-6)
    secured_ratio = (secured_amt or 0) / (debt_balance + 1e-6)
    ltv = (debt_balance or 0) / (property_value + 1e-6)
    return np.array([[avail, total_income or 0, edd_household or 0, extras_sum or 0,
                      debt_balance or 0, secured_amt or 0, property_value or 0,
                      (rate_pct or 0)/100.0, term_cap or 0, dti, secured_ratio, ltv]])

@st.cache_resource(show_spinner=False)
def get_lr_model():
    # απλό LinearRegression για να αποφεύγουμε βαριές βιβλιοθήκες/σφάλματα
    return LinearRegression()

def train_if_labels(df: pd.DataFrame):
    """Εκπαιδεύει μοντέλο μόνο αν υπάρχουν πραγματικές ρυθμίσεις ανά οφειλή στο real_debts_json."""
    if df.empty:
        return None, None
    rows = []
    y = []
    for _, r in df.iterrows():
        try:
            debts = json.loads(r.get("debts_json") or "[]")
            real_debts = json.loads(r.get("real_debts_json") or "[]")
        except Exception:
            continue
        if not real_debts:
            continue
        # χρησιμοποιούμε τις πραγματικές δόσεις ανά οφειλή ως labels
        for rd in real_debts:
            # βρες το αντίστοιχο debt balance (fallback 0)
            bal = 0.0
            for d in debts:
                if str(d.get("creditor","")).strip() == str(rd.get("creditor","")).strip() and abs(float(d.get("balance",0))-float(rd.get("balance",0)))<1e-3:
                    bal = float(d.get("balance",0))
                    break
            total_income = float(r.get("monthly_income") or 0)
            adults = int(r.get("adults") or 1)
            children = int(r.get("children") or 0)
            edd_household = float(r.get("edd_manual") or compute_edd(adults, children))
            extras_sum = float(r.get("extra_medical") or 0) + float(r.get("extra_students") or 0) + float(r.get("extra_legal") or 0)
            secured_amt = 0.0
            for d in debts:
                if d.get("secured"):
                    secured_amt += float(d.get("collateral_value") or 0)
            features = simple_features_row(
                total_income, edd_household, extras_sum, bal, secured_amt,
                float(r.get("property_value") or 0),
                float(r.get("annual_rate_pct") or 0),
                int(r.get("age_cap") or months_cap_from_age(int(r.get("debtor_age") or 45)))
            )
            rows.append(features[0])
            y.append(float(rd.get("monthly_payment") or 0.0))
    if not rows:
        return None, None
    X = np.vstack(rows)
    y = np.array(y)
    model = get_lr_model()
    try:
        model.fit(X, y)
        # πρόχειρο MAE (στο ίδιο σύνολο – απλά ένδειξη)
        preds = model.predict(X)
        mae = float(np.mean(np.abs(preds - y)))
    except Exception:
        return None, None
    return model, mae

def predict_single_debt_monthly(model,
                                monthly_income, edd_val, extras_sum,
                                debt_balance, debt_secured_amt,
                                property_value, annual_rate_pct,
                                age_cap_months, creditor_name):
    """Επιστρέφει (pred_monthly, residual_to_settle, writeoff_amount, haircut_pct, term_cap) για ΜΙΑ οφειλή."""
    term_cap = term_cap_for_single_debt(creditor_name, age_cap_months)

    # baseline κανόνας: μέχρι 70% του διαθέσιμου (ανά υπόθεση) αλλά όχι πάνω από balance/term_cap
    avail = max(0.0, (monthly_income or 0) - (edd_val or 0) - (extras_sum or 0))
    rule_pred = min(max(0.0, 0.70 * avail), float(debt_balance or 0) / max(term_cap,1))
    pred = rule_pred

    # αν υπάρχει μοντέλο, δοκίμασε να «βελτιώσεις» την πρόβλεψη
    if model is not None:
        Xd = simple_features_row(
            monthly_income, edd_val, extras_sum,
            debt_balance, debt_secured_amt,
            property_value, annual_rate_pct, term_cap
        )
        try:
            m_pred = float(model.predict(Xd)[0])
            # ασφαλείς κόφτες
            m_pred = max(0.0, min(m_pred, float(debt_balance or 0)/max(term_cap,1)))
            # blend 50-50 με rule για σταθερότητα
            pred = 0.5*rule_pred + 0.5*m_pred
        except NotFittedError:
            pass
        except Exception:
            pass

    expected_repay = pred * term_cap
    writeoff = max(0.0, float(debt_balance or 0) - expected_repay)
    residual_to_settle = max(0.0, float(debt_balance or 0) - writeoff)  # = expected_repay
    haircut_pct = 0.0
    if float(debt_balance or 0) > 0:
        haircut_pct = 100.0 * max(0.0, 1.0 - expected_repay / float(debt_balance))

    return float(pred), float(residual_to_settle), float(writeoff), float(haircut_pct), int(term_cap)

# ────────────────────────────── PDF ──────────────────────────────
def make_pdf(case_header: dict, debts_rows: list) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 2*cm

    # Logo
    try:
        if os.path.exists(LOGO_PATH):
            img = ImageReader(LOGO_PATH)
            c.drawImage(img, width-6*cm, y-1.8*cm, 5.2*cm, 1.5*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    c.setFont(PDF_FONT, 16)
    c.drawString(2*cm, y, "Bizboost – Πρόβλεψη Ρύθμισης (Ανά Οφειλή)")
    y -= 1.0*cm

    # Header στοιχεία
    c.setFont(PDF_FONT, 10)
    header_lines = [
        ("Υπόθεση", case_header.get("case_id","")),
        ("Οφειλέτης", case_header.get("borrower","")),
        ("Ηλικία", str(case_header.get("debtor_age",""))),
        ("Ενήλικες/Ανήλικοι", f"{case_header.get('adults',0)}/{case_header.get('children',0)}"),
        ("Σύνολο ετήσιων εισοδημάτων συνοφ.", f"{case_header.get('codebtors_annual',0):,.2f} €"),
        ("Συνολικά καθαρά ΜΗΝΙΑΙΑ εισοδήματα", f"{case_header.get('monthly_income',0):,.2f} €"),
        ("ΕΔΔ νοικοκυριού", f"{case_header.get('edd_household',0):,.2f} €"),
        ("Επιπλέον δαπάνες (ιατρικά/φοιτητές/δικαστικά)", f"{case_header.get('extras_sum',0):,.2f} €"),
        ("Διαθέσιμο εισόδημα", f"{case_header.get('avail',0):,.2f} €"),
        ("Ακίνητη περιουσία (σύνολο)", f"{case_header.get('property_value',0):,.2f} €"),
        ("Επιτόκιο (ετησίως)", f"{case_header.get('annual_rate_pct',0):,.2f}%"),
        ("Ημερ/νία", case_header.get("predicted_at","")),
    ]
    for k,v in header_lines:
        c.drawString(2*cm, y, f"{k}: {v}")
        y -= 0.55*cm
        if y < 3*cm:
            c.showPage(); y = height - 2*cm; c.setFont(PDF_FONT, 10)

    # Πίνακας οφειλών
    c.setFont(PDF_FONT, 12)
    c.drawString(2*cm, y, "Αποτελέσματα ανά οφειλή")
    y -= 0.7*cm
    c.setFont(PDF_FONT, 10)

    for row in debts_rows:
        line1 = f"- {row['Πιστωτής']} | {row['Είδος']} | Υπόλοιπο: {row['Υπόλοιπο (€)']:,.2f} € | Οροφή: {row['Οροφή μηνών']} μ."
        c.drawString(2*cm, y, line1)
        y -= 0.5*cm
        line2 = f"  Πρόταση δόσης: {row['Πρόταση δόσης (€)']:,.2f} € | Υπόλοιπο προς ρύθμιση: {row['Υπόλοιπο προς ρύθμιση (€)']:,.2f} € | Διαγραφή: {row['Διαγραφή (€)']:,.2f} € ({row['Πρόταση διαγραφής (%)']:.1f}%)"
        c.drawString(2*cm, y, line2)
        y -= 0.6*cm
        if y < 3*cm:
            c.showPage(); y = height - 2*cm; c.setFont(PDF_FONT, 10)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ────────────────────────────── UI ──────────────────────────────
st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost")
page = st.sidebar.radio("Μενού", ["Νέα Πρόβλεψη", "Διαχείριση Δεδομένων", "Εκπαίδευση Μοντέλου"], index=0)

df_all = load_data()

# ────────────────────────────── ΝΕΑ ΠΡΟΒΛΕΨΗ ──────────────────────────────
if page == "Νέα Πρόβλεψη":
    st.title("🧮 Πρόβλεψη Ρύθμισης (Εξωδικαστικός) – Ανά Οφειλή")

    with st.form("case_form", clear_on_submit=False, border=True):
        colA, colB, colC, colD = st.columns(4)
        borrower   = colA.text_input("Ονοματεπώνυμο / Κωδ. Υπόθεσης", "")
        debtor_age = colB.number_input("Ηλικία οφειλέτη", 18, 99, 45)
        adults     = colC.number_input("Ενήλικες στο νοικοκυριό", 1, 6, 1)
        children   = colD.number_input("Ανήλικοι στο νοικοκυριό", 0, 6, 0)

        col1, col2, col3 = st.columns(3)
        # Ετήσια εισοδήματα συνοφειλετών θα περαστούν ανα άτομο παρακάτω (data_editor)
        property_value = col1.number_input("Σύνολο αξίας ακίνητης περιουσίας (€)", 0.0, 1e10, 0.0, step=1000.0)
        annual_rate_pct= col2.number_input("Επιτόκιο ετησίως (%)", 0.0, 30.0, 6.0, step=0.1)
        borrower_annual_income = col3.number_input("Ετήσιο εισόδημα ΟΦΕΙΛΕΤΗ (€)", 0.0, 1e10, 0.0, step=100.0)

        st.subheader("Συνοφειλέτες")
        st.caption("Συμπλήρωσε ανά συνοφειλέτη: ονοματεπώνυμο, **ετήσιο εισόδημα**, ακίνητη περιουσία, ηλικία, ενήλικες/ανήλικοι που τον βαραίνουν (για ΕΔΔ).")
        default_codes = pd.DataFrame([{
            "name":"", "annual_income":0.0, "property_value":0.0, "age":40, "adults":1, "children":0
        }])
        co_df = st.data_editor(
            default_codes,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Ονοματεπώνυμο"),
                "annual_income": st.column_config.NumberColumn("Ετήσιο εισόδημα (€)", step=100.0, format="%.2f"),
                "property_value": st.column_config.NumberColumn("Ακίνητη περιουσία (€)", step=1000.0, format="%.2f"),
                "age": st.column_config.NumberColumn("Ηλικία", step=1),
                "adults": st.column_config.NumberColumn("Ενήλικες για ΕΔΔ", step=1),
                "children": st.column_config.NumberColumn("Ανήλικοι για ΕΔΔ", step=1),
            },
            use_container_width=True
        )

        st.subheader("Επιπλέον Δαπάνες (πέραν ΕΔΔ)")
        c1,c2,c3 = st.columns(3)
        extra_medical = c1.number_input("Ιατρικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)
        extra_students= c2.number_input("Φοιτητές / Σπουδές (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)
        extra_legal   = c3.number_input("Δικαστικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)

        st.markdown("---")
        st.subheader("Οφειλές")
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

        # ΕΔΔ νοικοκυριού (αυτόματο/χειροκίνητο)
        st.subheader("Ελάχιστες Δαπάνες Διαβίωσης (ΕΔΔ) Νοικοκυριού")
        use_manual = st.checkbox("Χειροκίνητη εισαγωγή ΕΔΔ;", value=False)
        if use_manual:
            edd_val = st.number_input("ΕΔΔ νοικοκυριού (€ / μήνα)", 0.0, 20000.0, 800.0, step=10.0)
        else:
            edd_val = compute_edd(int(adults), int(children))
            st.info(f"Αυτόματος υπολογισμός ΕΔΔ: **{edd_val:,.2f} €**")

        submitted = st.form_submit_button("Υπολογισμός Πρόβλεψης & Αποθήκευση", use_container_width=True)

    if submitted:
        # Μετατροπή εισοδημάτων: ετήσια -> μηνιαία (οφειλέτης + συνοφειλέτες)
        borrower_monthly = float(borrower_annual_income or 0) / 12.0

        co_list = co_df.fillna(0).to_dict(orient="records")
        codebtors_annual = sum(float(r.get("annual_income",0) or 0) for r in co_list)
        codebtors_monthly = codebtors_annual / 12.0

        # ΕΔΔ που «βαραίνουν» κάθε συνοφειλέτη (για να αφαιρεθούν από το δικό του μηνιαίο)
        codebtors_edd_monthly = 0.0
        for r in co_list:
            cad = int(r.get("adults") or 0)
            cch = int(r.get("children") or 0)
            codebtors_edd_monthly += compute_edd(cad, cch)

        monthly_income = borrower_monthly + codebtors_monthly
        edd_household  = float(edd_val or 0)
        extras_sum = float(extra_medical or 0) + float(extra_students or 0) + float(extra_legal or 0)

        # διαθέσιμο εισόδημα: από το ΣΥΝΟΛΟ αφαιρούμε το ΕΔΔ νοικοκυριού + επιπλέον δαπάνες
        # (έχεις ήδη ενσωματώσει τις ΕΔΔ των συνοφειλετών στο δικό τους μηνιαίο; εδώ κρατάμε απλό και ασφαλές: household EDD)
        avail = available_income(monthly_income, edd_household, extras_sum)

        # Συγκεντρωτικά από οφειλές
        debts = debts_df.fillna(0).to_dict(orient="records")
        total_debt = sum([float(d["balance"] or 0) for d in debts])
        secured_amt = sum([float(d["collateral_value"] or 0) for d in debts if d.get("secured")])

        age_cap_months = months_cap_from_age(int(debtor_age))

        # ML (αν υπάρχουν labels)
        model, mae = train_if_labels(df_all)

        # Υπολογισμός ανά οφειλή
        per_debt_rows = []
        enriched_debts = []
        for d in debts:
            creditor = str(d.get("creditor","")).strip()
            balance  = float(d.get("balance",0) or 0)
            is_sec   = bool(d.get("secured"))
            coll_val = float(d.get("collateral_value",0) or 0)

            pred_m, residual_to_settle, writeoff, hair_pct, term_cap_single = predict_single_debt_monthly(
                model=model,
                monthly_income=monthly_income,
                edd_val=edd_household,
                extras_sum=extras_sum,
                debt_balance=balance,
                debt_secured_amt=(coll_val if is_sec else 0.0),
                property_value=float(property_value or 0),
                annual_rate_pct=float(annual_rate_pct or 0),
                age_cap_months=age_cap_months,
                creditor_name=creditor
            )

            row_view = {
                "Πιστωτής": creditor,
                "Είδος": d.get("loan_type",""),
                "Υπόλοιπο (€)": balance,
                "Εξασφαλισμένο": "Ναι" if is_sec else "Όχι",
                "Εξασφάλιση (€)": coll_val if is_sec else 0.0,
                "Οροφή μηνών": term_cap_single,
                "Πρόταση δόσης (€)": round(pred_m, 2),
                "Υπόλοιπο προς ρύθμιση (€)": round(residual_to_settle, 2),
                "Διαγραφή (€)": round(writeoff, 2),
                "Πρόταση διαγραφής (%)": round(hair_pct, 2),
            }
            per_debt_rows.append(row_view)

            # αποθήκευση στην οφειλή
            d_out = dict(d)
            d_out.update({
                "predicted_monthly": round(pred_m,2),
                "predicted_residual": round(residual_to_settle,2),  # υπόλοιπο προς ρύθμιση
                "predicted_writeoff": round(writeoff,2),
                "predicted_haircut_pct": round(hair_pct,2),
                "term_cap": int(term_cap_single)
            })
            enriched_debts.append(d_out)

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("❗ Οι προτάσεις δίνονται **ανά οφειλή** (δεν γίνεται άθροιση). Οι οροφές: 420 μήνες για τράπεζες/servicers, 240 μήνες για ΑΑΔΕ/ΕΦΚΑ, με κόφτη ηλικίας.")

        # Αποθήκευση υπόθεσης
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        row = {
            "case_id": case_id,
            "borrower": borrower,
            "debtor_age": int(debtor_age),
            "adults": int(adults),
            "children": int(children),
            "monthly_income": float(monthly_income),      # ήδη σε μηνιαία βάση (οφειλέτης + συνοφειλέτες/12)
            "property_value": float(property_value),
            "annual_rate_pct": float(annual_rate_pct),

            "edd_use_manual": 1 if use_manual else 0,
            "edd_manual": float(edd_household),
            "extra_medical": float(extra_medical or 0),
            "extra_students": float(extra_students or 0),
            "extra_legal": float(extra_legal or 0),

            "age_cap": int(age_cap_months),

            "debts_json": enriched_debts,      # με προβλέψεις ανά οφειλή
            "co_debtors_json": co_list,
            "real_debts_json": [],

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

        # PDF
        header_for_pdf = {
            "case_id": case_id,
            "borrower": borrower,
            "debtor_age": int(debtor_age),
            "adults": int(adults),
            "children": int(children),
            "monthly_income": float(monthly_income),
            "edd_household": float(edd_household),
            "extras_sum": float(extras_sum),
            "avail": float(available_income(monthly_income, edd_household, extras_sum)),
            "property_value": float(property_value),
            "annual_rate_pct": float(annual_rate_pct),
            "predicted_at": now_str,
            "codebtors_annual": float(codebtors_annual)
        }
        pdf_bytes = make_pdf(header_for_pdf, per_debt_rows)
        st.download_button("⬇️ Λήψη Πρόβλεψης (PDF)", data=pdf_bytes, file_name=f"{case_id}_prediction.pdf", mime="application/pdf", use_container_width=True)

        if mae is not None:
            st.caption(f"MAE (εκπαίδευση στο ιστορικό): ~{mae:,.2f} €/μήνα")

# ────────────────────────────── ΔΙΑΧΕΙΡΙΣΗ ΔΕΔΟΜΕΝΩΝ ──────────────────────────────
elif page == "Διαχείριση Δεδομένων":
    st.title("📚 Διαχείριση Δεδομένων")

    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
    else:
        # Ελαφρύ view
        view = df_all[["case_id","borrower","predicted_at","debts_json"]].copy()
        st.dataframe(view.sort_values("predicted_at", ascending=False), use_container_width=True)

        st.markdown("---")
        st.subheader("Ενημέρωση με Πραγματική Ρύθμιση (ανά οφειλή)")
        case_ids = df_all["case_id"].tolist()
        pick_id = st.selectbox("Διάλεξε Υπόθεση", case_ids)
        row = df_all[df_all["case_id"]==pick_id].iloc[0].to_dict()
        try:
            pred_debts = json.loads(row.get("debts_json") if isinstance(row.get("debts_json"), str) else json.dumps(row.get("debts_json") or []))
        except Exception:
            pred_debts = []

        st.caption("Συμπλήρωσε ανά οφειλή τα στοιχεία της ΠΡΑΓΜΑΤΙΚΗΣ ρύθμισης.")
        real_df_default = []
        for d in pred_debts:
            real_df_default.append({
                "creditor": d.get("creditor",""),
                "balance": float(d.get("balance",0) or 0),
                "writeoff_amount": 0.0,
                "residual_to_settle": float(d.get("predicted_residual",0) or 0),  # αρχική πρόταση
                "term_months": int(d.get("term_cap",0) or 0),
                "monthly_payment": float(d.get("predicted_monthly",0) or 0),
            })
        real_df = st.data_editor(
            pd.DataFrame(real_df_default), num_rows="dynamic",
            column_config={
                "creditor": st.column_config.SelectboxColumn("Πιστωτής", options=CREDITORS),
                "balance": st.column_config.NumberColumn("Ποσό δανείου (€)", step=500.0, format="%.2f"),
                "writeoff_amount": st.column_config.NumberColumn("Διαγραφή (€)", step=100.0, format="%.2f"),
                "residual_to_settle": st.column_config.NumberColumn("Υπόλοιπο προς ρύθμιση (€)", step=100.0, format="%.2f"),
                "term_months": st.column_config.NumberColumn("Μήνες δόσεων", step=1),
                "monthly_payment": st.column_config.NumberColumn("Δόση (€)", step=10.0, format="%.2f"),
            },
            use_container_width=True
        )

        if st.button("💾 Αποθήκευση Πραγματικής Ρύθμισης", type="primary"):
            real_rows = real_df.fillna(0).to_dict(orient="records")
            # υπολόγισε % κουρέματος ανά οφειλή
            for r in real_rows:
                bal = float(r.get("balance",0) or 0)
                wr  = float(r.get("writeoff_amount",0) or 0)
                r["haircut_pct"] = float(100.0 * (wr / (bal+1e-6)) if bal>0 else 0.0)
            row_update = row.copy()
            row_update["real_debts_json"] = real_rows
            save_data(pd.DataFrame([row_update]))
            st.success("✅ Αποθηκεύτηκε η πραγματική ρύθμιση για την υπόθεση.")

# ────────────────────────────── ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΟΥ ──────────────────────────────
else:
    st.title("🤖 Εκπαίδευση & Απόδοση Μοντέλου (προαιρετικό)")
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα δεδομένα.")
    else:
        model, mae = train_if_labels(df_all)
        if model is None:
            st.warning("Δεν βρέθηκαν επαρκή labels (πραγματικές δόσεις) για εκπαίδευση. Γίνεται fallback σε κανόνες.")
        else:
            st.success("Το ελαφρύ μοντέλο εκπαιδεύτηκε και χρησιμοποιείται στις νέες προβλέψεις.")
            st.metric("MAE (€/μήνα)", f"{mae:,.2f}")
