# app.py
# Bizboost - Πρόβλεψη Ρυθμίσεων Εξωδικαστικού (κανόνες, χωρίς ML)
# - Ανά οφειλή: δόση + υπόλοιπο προς ρύθμιση + διαγραφή (€/%)
# - 420 μήνες για Τράπεζες/Servicers, 240 μήνες για ΑΑΔΕ/ΕΦΚΑ, με κόφτη ηλικίας
# - Συνοφειλέτες με ετήσιο εισόδημα -> μηνιαίο και αφαίρεση ΕΔΔ
# - Supabase Postgres μέσω SQLAlchemy/psycopg v3
# - PDF με ελληνικά (αν υπάρχει assets/fonts/DejaVuSans.ttf)

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ──────────────────────── ΒΑΣΙΚΕΣ ΡΥΘΜΙΣΕΙΣ ────────────────────────
st.set_page_config(page_title="Bizboost - Πρόβλεψη Ρυθμίσεων", page_icon="💠", layout="wide")

LOGO_PATH = "logo.png"
DATA_CSV  = "cases.csv"

# Προσπάθεια για ελληνική γραμματοσειρά
FONT_PATH = "assets/fonts/DejaVuSans.ttf"
HAS_GREEK_FONT = os.path.exists(FONT_PATH)
if HAS_GREEK_FONT:
    try:
        pdfmetrics.registerFont(TTFont("GR", FONT_PATH))
        PDF_FONT = "GR"
    except Exception:
        PDF_FONT = "Helvetica"
else:
    PDF_FONT = "Helvetica"

# Πιστωτές
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

# ───────────────────── ΕΔΔ (ελάχιστες δαπάνες) ─────────────────────
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

def term_cap_for_single_debt(creditor_name: str, age_cap_months: int) -> int:
    c = (creditor_name or "").strip()
    policy_cap = 420 if c in BANK_SERVICERS else 240  # ΑΑΔΕ/ΕΦΚΑ = 240
    return max(1, min(policy_cap, age_cap_months))

def available_income(total_income:float, edd_household:float,
                     extra_medical:float, extra_students:float, extra_legal:float)->float:
    extras = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
    return max(0.0, float(total_income or 0) - float(edd_household or 0) - extras)

# ───────────────────── ΑΠΛΗ ΟΙΚΟΝΟΜΙΚΗ ΛΟΓΙΚΗ ─────────────────────
def annuity_payment(balance: float, months: int, annual_rate_pct: float) -> float:
    """Μηνιαία δόση για να αποσβεστεί πλήρως ένα υπόλοιπο (αν το αντέχει ο οφειλέτης)."""
    b = float(balance or 0.0)
    n = max(1, int(months or 1))
    r = float(annual_rate_pct or 0.0) / 12.0 / 100.0
    if b <= 0:
        return 0.0
    if r <= 0:
        return b / n
    denom = 1 - (1 + r) ** (-n)
    if abs(denom) < 1e-12:
        return b / n
    return b * r / denom

def rule_based_monthly(avail_monthly: float,
                       balance: float,
                       months_cap: int,
                       annual_rate_pct: float) -> tuple[float, str]:
    """
    Δόση = min( 70% διαθέσιμου εισοδήματος , annuity(balance, months_cap, rate) )
    Επιστρέφει (μηνιαία_δόση, σύντομη_αιτιολόγηση)
    """
    cap_ratio = 0.70
    cap_from_income = max(0.0, float(avail_monthly or 0) * cap_ratio)
    cap_from_annuity = annuity_payment(balance, months_cap, annual_rate_pct)
    monthly = min(cap_from_income, cap_from_annuity)
    monthly = round(max(0.0, monthly), 2)

    reason = (
        f"Χρησιμοποιήθηκε 70% του διαθέσιμου ({cap_from_income:,.2f} €) "
        f"και όριο αναπόσβεσης {months_cap} μηνών στο {annual_rate_pct:.2f}% "
        f"(annuity: {cap_from_annuity:,.2f} €). Επιλέχθηκε η μικρότερη δόση."
    )
    return monthly, reason

# ───────────────────────────── ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ─────────────────────────────
def get_db_engine():
    # 1) Secrets (Streamlit/Local)  2) Env var
    try:
        db_url = st.secrets["DATABASE_URL"]
    except Exception:
        db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        st.error("Δεν έχει οριστεί DATABASE_URL στα Secrets ή στα Environment variables.")
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
    engine = get_db_engine()
    init_db(engine)
    try:
        return pd.read_sql("SELECT * FROM cases", con=engine)
    except Exception as e:
        st.error(f"Σφάλμα ανάγνωσης DB: {e}")
        return pd.DataFrame()

def _nan_to_none(x):
    if x is None: return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return None
    return x

def upsert_cases_db(df: pd.DataFrame):
    if df.empty: return
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

    for c in ["debts_json","co_debtors_json"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: v if isinstance(v,str) else json.dumps(v, ensure_ascii=False))

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

# ────────────────────────────── PDF EXPORT ──────────────────────────────
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

    c.setFont(PDF_FONT if HAS_GREEK_FONT else "Helvetica-Bold", 16)
    c.drawString(2*cm, y, "Bizboost - Πρόβλεψη Ρύθμισης")
    y -= 1.0*cm

    c.setFont(PDF_FONT, 10)
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
        ("Επιτόκιο (ετ.)", f"{case_dict.get('annual_rate_pct',0):,.2f}%"),
        ("Ημ/νία", case_dict.get("predicted_at","")),
    ]:
        c.drawString(2*cm, y, f"{k}: {v}")
        y -= 0.6*cm
        if y < 3*cm:
            c.showPage(); y = height - 2*cm
            c.setFont(PDF_FONT, 10)

    debts = case_dict.get("debts", [])
    if debts:
        c.setFont(PDF_FONT, 12)
        c.drawString(2*cm, y, "Αναλυτικά Οφειλές:")
        y -= 0.7*cm
        c.setFont(PDF_FONT, 10)
        for d in debts:
            balance = float(d.get("balance",0) or 0)
            pm = float(d.get("predicted_monthly",0) or 0)
            term_cap = int(d.get("term_cap",0) or 0)
            expected_repay = pm * max(1,term_cap)
            writeoff_amount = max(0.0, balance - expected_repay)
            remaining_after_writeoff = max(0.0, balance - writeoff_amount)
            writeoff_pct = (writeoff_amount / (balance+1e-6)) * 100.0 if balance>0 else 0.0

            line1 = f"- {d.get('creditor')} | {d.get('loan_type')} | Υπόλοιπο: {balance:,.2f} €"
            line2 = f"  → Δόση: {pm:,.2f} € • Μήνες: {term_cap} • Υπόλοιπο προς ρύθμιση: {remaining_after_writeoff:,.2f} € • Διαγραφή: {writeoff_amount:,.2f} € ({writeoff_pct:.1f}%)"
            c.drawString(2*cm, y, line1); y -= 0.55*cm
            c.drawString(2*cm, y, line2); y -= 0.55*cm

            # Σκεπτικό
            reason = d.get("rationale","")
            if reason:
                c.drawString(2*cm, y, f"  Σκεπτικό: {reason}")
                y -= 0.55*cm

            if y < 3*cm:
                c.showPage(); y = height - 2*cm
                c.setFont(PDF_FONT, 10)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ────────────────────────────── UI ──────────────────────────────
st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost")
page = st.sidebar.radio("Μενού", ["Νέα Πρόβλεψη", "Αποθηκευμένες Προβλέψεις"], index=0)

df_all = load_data()

# ───────────────────────── ΝΕΑ ΠΡΟΒΛΕΨΗ ─────────────────────────
if page == "Νέα Πρόβλεψη":
    st.title("🧮 Πρόβλεψη Ρύθμισης (ανά οφειλή)")

    with st.form("case_form", clear_on_submit=False):
        colA, colB, colC, colD = st.columns(4)
        borrower   = colA.text_input("Ονοματεπώνυμο / Κωδ. Υπόθεσης", "")
        debtor_age = colB.number_input("Ηλικία οφειλέτη", 18, 99, 45)
        adults     = colC.number_input("Ενήλικες στο νοικοκυριό", 1, 6, 1)
        children   = colD.number_input("Ανήλικοι στο νοικοκυριό", 0, 6, 0)

        st.markdown("### Συνολικό μηνιαίο εισόδημα (οφειλέτη + συνοφειλετών)")
        calc_from_codes = st.checkbox("Υπολογισμός από πίνακα συνοφειλετών (ετήσιο→μηνιαίο & αφαίρεση ΕΔΔ)", value=True)
        monthly_income_input = st.number_input("Μηνιαίο εισόδημα (€) [αν δεν χρησιμοποιήσεις τον πίνακα]", 0.0, 1e9, 0.0, step=50.0)

        col1, col2 = st.columns(2)
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
        # Διαθέσιμο εισόδημα από συνοφειλέτες
        codebtors = codef_df.fillna(0).to_dict(orient="records") if codef_df is not None else []
        def edd_single_adult(): return compute_edd(1, 0)
        monthly_income_codes = 0.0
        for co in codebtors:
            ann = float(co.get("annual_income",0) or 0.0)
            mon_gross = ann/12.0
            mon_net = max(0.0, mon_gross - edd_single_adult())
            monthly_income_codes += mon_net

        monthly_income = monthly_income_codes if calc_from_codes else monthly_income_input

        debts = debts_df.fillna(0).to_dict(orient="records")
        total_debt = sum([float(d["balance"] or 0) for d in debts])
        secured_amt = sum([float(d["collateral_value"] or 0) for d in debts if d.get("secured")])

        extras_sum = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
        avail = available_income(monthly_income, edd_val, extra_medical, extra_students, extra_legal)
        age_cap_months = months_cap_from_age(int(debtor_age))

        # Πρόβλεψη ανά οφειλή (κανόνες)
        per_debt_rows = []
        for d in debts:
            creditor = str(d.get("creditor", "")).strip()
            balance  = float(d.get("balance", 0) or 0)
            is_sec   = bool(d.get("secured"))
            coll_val = float(d.get("collateral_value", 0) or 0)

            term_cap_single = term_cap_for_single_debt(creditor, age_cap_months)
            pred_m, rationale = rule_based_monthly(
                avail_monthly=avail,
                balance=balance,
                months_cap=term_cap_single,
                annual_rate_pct=annual_rate_pct
            )
            expected_repay = pred_m * term_cap_single
            writeoff_amount = max(0.0, balance - expected_repay)
            remaining_after_writeoff = max(0.0, balance - writeoff_amount)
            writeoff_pct = (writeoff_amount / (balance+1e-6)) * 100.0 if balance>0 else 0.0

            d["predicted_monthly"] = round(pred_m, 2)
            d["predicted_haircut_pct"] = round(writeoff_pct, 2)   # μόνο για πληροφορία
            d["term_cap"] = int(term_cap_single)
            d["rationale"] = (
                f"Πιστωτής: {creditor} • Πολιτική μηνών: {420 if creditor in BANK_SERVICERS else 240} • "
                f"Κόφτης ηλικίας: {age_cap_months} • Επιλέχθηκε {term_cap_single} μήνες. "
                + rationale
            )

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
                "Σκεπτικό": d["rationale"]
            })

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("Οι προτάσεις είναι **ανά οφειλή** (δεν γίνεται συνολική άθροιση).")

        # Αποθήκευση υπόθεσης
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
        st.success(f"✅ Αποθήκευση ολοκληρώθηκε. Κωδικός: {case_id}")

        # PDF
        case_for_pdf = dict(row)
        case_for_pdf["edd_household"] = float(edd_val)
        case_for_pdf["extras_sum"] = float(extras_sum)
        case_for_pdf["avail"] = float(avail)
        case_for_pdf["debts"] = debts
        pdf_bytes = make_pdf(case_for_pdf)
        st.download_button("⬇️ Λήψη Πρόβλεψης (PDF)", data=pdf_bytes,
                           file_name=f"{case_id}_prediction.pdf", mime="application/pdf",
                           use_container_width=True)

# ───────────────────── ΑΠΟΘΗΚΕΥΜΕΝΕΣ ΠΡΟΒΛΕΨΕΙΣ ─────────────────────
else:
    st.title("📂 Αποθηκευμένες Προβλέψεις & Καταχώριση Πραγματικών Ρυθμίσεων")
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
    else:
        cases = df_all.sort_values("predicted_at", ascending=False)
        pick = st.selectbox(
            "Διάλεξε Υπόθεση",
            cases["case_id"].tolist(),
            format_func=lambda cid: f"{cid} — {cases[cases.case_id==cid].iloc[0].get('borrower','')}"
        )
        row = cases[cases["case_id"]==pick].iloc[0].to_dict()

        st.markdown(f"**Οφειλέτης:** {row.get('borrower','')}  |  **Ημ/νία:** {row.get('predicted_at','')}")
        try:
            debts = json.loads(row.get("debts_json") or "[]")
        except Exception:
            debts = []

        st.subheader("Οφειλές & Πρόβλεψη (όπως αποθηκεύτηκαν)")
        df_pred = []
        for d in debts:
            balance = float(d.get("balance",0) or 0)
            pm = float(d.get("predicted_monthly",0) or 0)
            term_cap = int(d.get("term_cap",0) or 0)
            expected_repay = pm * max(1,term_cap)
            writeoff_amount = max(0.0, balance - expected_repay)
            remaining_after_writeoff = max(0.0, balance - writeoff_amount)
            writeoff_pct = (writeoff_amount / (balance+1e-6)) * 100.0 if balance>0 else 0.0

            df_pred.append({
                "Πιστωτής": d.get("creditor",""),
                "Είδος": d.get("loan_type",""),
                "Υπόλοιπο (€)": balance,
                "Οροφή μηνών": term_cap,
                "Δόση πρότασης (€)": round(pm,2),
                "Υπόλοιπο προς ρύθμιση (€)": round(remaining_after_writeoff,2),
                "Διαγραφή (€)": round(writeoff_amount,2),
                "Διαγραφή (%)": round(writeoff_pct,2),
            })
        st.dataframe(pd.DataFrame(df_pred), use_container_width=True)

        st.markdown("---")
        st.subheader("Καταχώριση Πραγματικής Ρύθμισης ανά οφειλή")
        editable = []
        for idx, d in enumerate(debts):
            st.markdown(f"**{idx+1}. {d.get('creditor','')} — {d.get('loan_type','')}**")
            cols = st.columns(6)
            real_amount   = cols[0].number_input("Ποσό δανείου (€)", 0.0, 1e12, float(d.get("balance",0) or 0.0), key=f"ra_{idx}")
            real_writeoff = cols[1].number_input("Διαγραφή (€)", 0.0, 1e12, float(d.get("real_writeoff_amount", d.get("writeoff_amount", 0.0)) or 0.0), key=f"rw_{idx}")
            real_term     = cols[2].number_input("Μήνες δόσεων", 0, 1200, int(d.get("real_term_months", d.get("term_cap",0)) or 0), key=f"rt_{idx}")
            real_monthly  = cols[3].number_input("Δόση (€)", 0.0, 1e9, float(d.get("real_monthly", d.get("predicted_monthly",0.0)) or 0.0), key=f"rm_{idx}")
            real_residual = cols[4].number_input("Υπόλοιπο προς ρύθμιση (€)", 0.0, 1e12,
                                                 float(d.get("real_residual_balance", max(0.0, real_amount - real_writeoff)) or 0.0),
                                                 key=f"rr_{idx}")
            real_haircut  = (float(real_writeoff or 0) / (float(real_amount or 1e-6))) * 100.0 if real_amount>0 else 0.0
            cols[5].metric("Κούρεμα (%)", f"{real_haircut:.1f}")

            editable.append({
                "idx": idx,
                "real_amount": real_amount,
                "real_writeoff": real_writeoff,
                "real_term": real_term,
                "real_monthly": real_monthly,
                "real_residual": real_residual,
                "real_haircut_pct": real_haircut
            })

        if st.button("💾 Αποθήκευση Πραγματικών Ρυθμίσεων", type="primary"):
            # γράφουμε τα real_* πίσω στο debts_json
            for e in editable:
                i = e["idx"]
                if i < len(debts):
                    debts[i]["real_amount"] = float(e["real_amount"])
                    debts[i]["real_writeoff_amount"] = float(e["real_writeoff"])
                    debts[i]["real_term_months"] = int(e["real_term"])
                    debts[i]["real_monthly"] = float(e["real_monthly"])
                    debts[i]["real_residual_balance"] = float(e["real_residual"])
                    debts[i]["real_haircut_pct"] = float(e["real_haircut_pct"])

            row_update = row.copy()
            row_update["debts_json"] = json.dumps(debts, ensure_ascii=False)

            # Παράγουμε και συνοπτικά πεδία (προαιρετικά)
            try:
                # μέση πραγματική δόση (αν υπάρχουν)
                reals = [float(d.get("real_monthly",0) or 0) for d in debts if d.get("real_monthly") is not None]
                row_update["real_monthly"] = float(np.mean(reals)) if reals else None
            except Exception:
                pass

            save_data(pd.DataFrame([row_update]))
            st.success("✅ Αποθήκευση πραγματικών ρυθμίσεων ολοκληρώθηκε.")
