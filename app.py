# app.py
# Bizboost - Πρόβλεψη Ρυθμίσεων Εξωδικαστικού (Streamlit + Postgres + PDF)
# - Ελληνικό UI
# - Postgres (Supabase) μέσω SQLAlchemy + psycopg v3
# - PDF export με DejaVuSans (σωστά Ελληνικά), logo και πίνακες
# - Rule-based υπολογισμός ανά οφειλή (δόση, διαγραφή, υπόλοιπο προς ρύθμιση, %)
# - Αποθήκευση προβλέψεων και "πραγματικών ρυθμίσεων" ανά οφειλή (real_debts_json)

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ────────────────────────────── UI / PATHS ──────────────────────────────
st.set_page_config(page_title="Bizboost - Πρόβλεψη Ρυθμίσεων", page_icon="💠", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
FONT_PATH = os.path.join(BASE_DIR, "assets", "fonts", "DejaVuSans.ttf")
DATA_CSV  = os.path.join(BASE_DIR, "cases.csv")  # προαιρετικό αρχικό import

# Γραμματοσειρά για Ελληνικά στο PDF
try:
    if os.path.exists(FONT_PATH):
        pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_PATH))
        PDF_FONT = "DejaVuSans"
    else:
        st.warning("Λείπει η γραμματοσειρά PDF: assets/fonts/DejaVuSans.ttf (για σωστά Ελληνικά).")
        PDF_FONT = "Helvetica"
except Exception as e:
    st.warning(f"Αποτυχία φόρτωσης γραμματοσειράς PDF: {e}")
    PDF_FONT = "Helvetica"

# ────────────────────────────── ΠΑΡΑΜΕΤΡΟΙ ΠΟΛΙΤΙΚΗΣ ──────────────────────────────
PUBLIC_CREDITORS = {"ΑΑΔΕ", "ΕΦΚΑ"}
BANK_SERVICERS   = {
    "DoValue","Intrum","Cepal","Qquant","APS","EOS","Veralitis",
    "Πειραιώς","Εθνική","Eurobank","Alpha"
}
CREDITORS = list(PUBLIC_CREDITORS) + list(BANK_SERVICERS)
LOAN_TYPES = ["Στεγαστικό","Καταναλωτικό","Επαγγελματικό"]

POLICY = {
    # Σειρά εξυπηρέτησης διαθέσιμου εισοδήματος:
    # PUBLIC -> SECURED -> UNSECURED
    "priority": ["PUBLIC", "SECURED", "UNSECURED"],
    # Οροφές μηνών ανά κατηγορία
    "term_caps": {"PUBLIC": 240, "BANK": 420, "DEFAULT": 240},
    # Τρόπος κατανομής διαθέσιμου:
    #  - "priority_first": πρώτα PUBLIC (αναλογικά μεταξύ τους), μετά SECURED, μετά UNSECURED
    #  - "proportional": αναλογικά σε ΟΛΕΣ τις οφειλές βάσει υπολοίπου
    "allocate": "priority_first",
    # Προαιρετικό max haircut ανά κατηγορία (None = χωρίς όριο). Παράδειγμα: 0.4 για 40%
    "max_haircut": {"PUBLIC": None, "BANK": None, "DEFAULT": None},
}

# ─────────────────────── ΕΔΔ & ΔΙΑΘΕΣΙΜΑ ───────────────────────
def compute_edd(adults:int, children:int)->float:
    """Απλή κλίμακα ΕΔΔ: 1 ενήλικας 537€, κάθε επιπλέον ενήλικας +269€, κάθε ανήλικος +211€."""
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
    """Κόφτης διάρκειας με βάση ηλικία οφειλέτη (προσαρμόσιμος)."""
    try:
        a = int(age)
    except:
        return 120
    if a <= 35:  return 240
    if a <= 50:  return 180
    if a <= 65:  return 120
    return 60

def available_income(total_income:float, edd_household:float, extra_medical:float, extra_students:float, extra_legal:float)->float:
    """Μηνιαίο διαθέσιμο = εισόδημα - ΕΔΔ - επιπλέον δαπάνες."""
    extras = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
    return max(0.0, float(total_income or 0) - float(edd_household or 0) - extras)

# ─────────────────────────────── RULE ENGINE ───────────────────────────────
def classify_debt(creditor:str, secured:bool)->str:
    if creditor in PUBLIC_CREDITORS: return "PUBLIC"
    if creditor in BANK_SERVICERS:   return "BANK" if secured else "UNSECURED"
    return "DEFAULT"

def term_cap_for(creditor:str, age_cap:int, secured:bool)->int:
    cat = classify_debt(creditor, secured)
    base = POLICY["term_caps"].get(cat, POLICY["term_caps"]["DEFAULT"])
    return max(1, min(base, age_cap))

def security_floor(balance:float, secured:bool, collateral_value:float)->float:
    """Κατώφλι λόγω εξασφάλισης: δεν διαγράφεις κάτω από το (balance - collateral)."""
    if not secured:
        return 0.0
    return max(0.0, float(balance or 0.0) - float(collateral_value or 0.0))

def split_available_proportional(avail:float, debts:list)->dict:
    total = sum(d["balance"] for d in debts if d["balance"]>0)
    if total <= 0: return {i:0.0 for i in range(len(debts))}
    return {i: avail * (d["balance"]/total) for i,d in enumerate(debts)}

def split_available_priority(avail:float, debts:list)->dict:
    """Πρώτα PUBLIC (αναλογικά μεταξύ τους), μετά SECURED, μετά UNSECURED."""
    out = {i:0.0 for i in range(len(debts))}
    groups = {"PUBLIC":[], "SECURED":[], "UNSECURED":[]}
    for i,d in enumerate(debts):
        if d["cat"] == "PUBLIC":
            groups["PUBLIC"].append(i)
        elif d["secured"]:
            groups["SECURED"].append(i)
        else:
            groups["UNSECURED"].append(i)

    remaining = avail
    for key in POLICY["priority"]:
        idxs = groups.get(key, [])
        if not idxs: 
            continue
        subtotal = sum(debts[i]["balance"] for i in idxs if debts[i]["balance"]>0)
        if subtotal <= 0:
            continue
        for i in idxs:
            share = remaining * (debts[i]["balance"]/subtotal)
            out[i] += share
        # εξαντλούμε σε αυτό το “κύμα”
        remaining = 0.0
        if remaining <= 0:
            break
    return out

def compute_offer_per_debt(d, monthly_share, age_cap):
    """Υπολογισμός πρότασης ανά οφειλή με όρους εξωδικαστικού."""
    term = term_cap_for(d["creditor"], age_cap, d["secured"])
    inst = max(0.0, float(monthly_share))
    gross_residual = max(0.0, d["balance"] - inst*term)
    floor = security_floor(d["balance"], d["secured"], d.get("collateral_value",0.0))
    residual = max(gross_residual, floor)
    writeoff = max(0.0, d["balance"] - residual)
    haircut = 0.0 if d["balance"]<=0 else 100.0*writeoff/(d["balance"]+1e-6)

    # προαιρετικό όριο κουρέματος
    cat = classify_debt(d["creditor"], d["secured"])
    max_hc = POLICY["max_haircut"].get(cat)
    if isinstance(max_hc, (int,float)) and max_hc is not None:
        max_write = d["balance"]*max_hc
        if writeoff > max_write:
            writeoff = max_write
            residual = d["balance"] - writeoff
            haircut = 100.0*max_hc

    return {
        "term_cap": int(term),
        "predicted_monthly": round(inst,2),
        "predicted_writeoff": round(writeoff,2),
        "predicted_residual": round(residual,2),
        "predicted_haircut_pct": round(haircut,1),
    }

# ─────────────────────────────── ΒΑΣΗ ΔΕΔΟΜΕΝΩΝ ───────────────────────────────
def get_db_engine():
    db_url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL",""))
    if not db_url:
        st.error("Δεν έχει οριστεί DATABASE_URL στα Secrets.")
        st.stop()
    # psycopg v3 driver string
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def init_db(engine):
    """Δημιουργία/αναβάθμιση σχήματος cases, συμπεριλαμβανομένου του real_debts_json."""
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
      real_debts_json JSONB,
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
        try:
            conn.execute(text("ALTER TABLE cases ADD COLUMN IF NOT EXISTS real_debts_json JSONB DEFAULT '[]';"))
        except Exception:
            pass

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
        "case_id","borrower","debtor_age","adults","children","monthly_income","property_value",
        "annual_rate_pct","edd_use_manual","edd_manual","extra_medical","extra_students","extra_legal",
        "age_cap","debts_json","co_debtors_json","real_debts_json","term_months","predicted_at",
        "predicted_monthly","predicted_haircut_pct","prob_accept","real_monthly","real_haircut_pct",
        "accepted","real_term_months","real_writeoff_amount","real_residual_balance"
    ]
    df2 = df.copy()

    for c in ["debts_json","co_debtors_json","real_debts_json"]:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: x if isinstance(x,str) else json.dumps(x if x is not None else [], ensure_ascii=False))

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
            for c in ["debts_json","co_debtors_json","real_debts_json"]:
                if c in dfcsv.columns:
                    dfcsv[c] = dfcsv[c].fillna("[]")
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
    """Επαγγελματικό PDF με Ελληνικά, logo, πίνακες, αιτιολόγηση."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm)

    styles = getSampleStyleSheet()
    base_font = PDF_FONT
    styles.add(ParagraphStyle(name="H1", fontName=base_font, fontSize=16, leading=20, spaceAfter=10, textColor=colors.HexColor("#0F4C81")))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, leading=16, spaceAfter=6, textColor=colors.HexColor("#333333")))
    styles.add(ParagraphStyle(name="P",  fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Small", fontName=base_font, fontSize=9, leading=12, textColor=colors.HexColor("#555")))

    story = []

    # Header με logo
    if os.path.exists(LOGO_PATH):
        try:
            story.append(Image(LOGO_PATH, width=140, height=40))
        except Exception:
            pass

    story.append(Spacer(1, 8))
    story.append(Paragraph("Bizboost – Πρόβλεψη Ρύθμισης", styles["H1"]))

    # Στοιχεία περίληψης
    meta = [
        ["Υπόθεση", case_dict.get("case_id","")],
        ["Οφειλέτης", case_dict.get("borrower","")],
        ["Ηλικία", str(case_dict.get("debtor_age",""))],
        ["Μέλη νοικοκυριού (ενήλ./ανήλ.)", f"{case_dict.get('adults',0)}/{case_dict.get('children',0)}"],
        ["Συνολικό μηνιαίο εισόδημα", f"{case_dict.get('monthly_income',0):,.2f} €"],
        ["ΕΔΔ νοικοκυριού", f"{case_dict.get('edd_household',0):,.2f} €"],
        ["Επιπλέον δαπάνες", f"{case_dict.get('extras_sum',0):,.2f} €"],
        ["Καθαρό διαθέσιμο", f"{case_dict.get('avail',0):,.2f} €"],
        ["Ακίνητη περιουσία", f"{case_dict.get('property_value',0):,.2f} €"],
        ["Ημερομηνία", case_dict.get("predicted_at","")],
    ]
    t = Table(meta, colWidths=[6*cm, 8*cm])
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), base_font, 10),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#DDD")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#AAA")),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F5F7FA")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
        ("TOPPADDING",(0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Αναλυτικά οφειλές (πρόβλεψη)
    debts = case_dict.get("debts", [])
    if debts:
        story.append(Paragraph("Αναλυτικά ανά οφειλή (πρόβλεψη):", styles["H2"]))
        rows = [["Πιστωτής","Είδος","Υπόλοιπο (€)","Εξασφαλ.","Εξασφάλιση (€)","Οροφή μηνών","Πρόταση δόσης (€)","Διαγραφή (€)","Υπόλοιπο Ρύθμισης (€)","Κούρεμα (%)"]]
        for d in debts:
            rows.append([
                d.get("creditor",""),
                d.get("loan_type",""),
                f"{float(d.get('balance',0)):,.2f}",
                "Ναι" if d.get("secured") else "Όχι",
                f"{float(d.get('collateral_value',0)):,.2f}" if d.get("secured") else "0.00",
                str(d.get("term_cap","")),
                f"{float(d.get('predicted_monthly',0)):,.2f}",
                f"{float(d.get('predicted_writeoff',0)):,.2f}",
                f"{float(d.get('predicted_residual',0)):,.2f}",
                f"{float(d.get('predicted_haircut_pct',0)):.1f}%",
            ])
        tt = Table(rows, colWidths=[3*cm,2.5*cm,2.7*cm,1.6*cm,2.9*cm,2.3*cm,3.0*cm,3.0*cm,3.3*cm,2.2*cm])
        tt.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), base_font, 9),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F4C81")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (2,1), (-1,-1), "RIGHT"),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#DDD")),
            ("BOX", (0,0), (-1,-1), 0.6, colors.HexColor("#0F4C81")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
        ]))
        story.append(tt)
        story.append(Spacer(1, 8))

    # Σκεπτικό
    reasoning = (
        "Η πρόταση ανά οφειλή προκύπτει βάσει κανόνων εξωδικαστικού: "
        "για απαιτήσεις Δημοσίου (ΑΑΔΕ/ΕΦΚΑ) ο μέγιστος αριθμός δόσεων λαμβάνεται "
        "έως 240 μήνες, ενώ για τραπεζικά/servicers έως 420 μήνες. "
        "Εφαρμόζεται επίσης κόφτης βάσει ηλικίας του οφειλέτη. "
        "Ως διαθέσιμο εισόδημα λαμβάνεται το συνολικό μηνιαίο εισόδημα μειωμένο "
        "κατά τις Ελάχιστες Δαπάνες Διαβίωσης και τις επιπλέον δηλωθείσες δαπάνες. "
        "Το υπόλοιπο προς ρύθμιση ανά οφειλή ισούται με: Υπόλοιπο οφειλής − Διαγραφή. "
        "Το ποσοστό κουρέματος είναι Διαγραφή / Υπόλοιπο οφειλής."
    )
    story.append(Paragraph("Σκεπτικό πρότασης", styles["H2"]))
    story.append(Paragraph(reasoning, styles["P"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ────────────────────────────── UI ──────────────────────────────
st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost")
page = st.sidebar.radio("Μενού", ["Νέα Πρόβλεψη", "Προβλέψεις & Πραγματικές Ρυθμίσεις"], index=0)

df_all = load_data()

# ────────────────────────────── ΝΕΑ ΠΡΟΒΛΕΨΗ ──────────────────────────────
if page == "Νέα Πρόβλεψη":
    st.title("🧮 Πρόβλεψη Ρύθμισης (Εξωδικαστικός)")

    with st.form("case_form", clear_on_submit=False, border=True):
        colA, colB, colC, colD = st.columns(4)
        borrower   = colA.text_input("Ονοματεπώνυμο / Κωδ. Υπόθεσης", "")
        debtor_age = colB.number_input("Ηλικία οφειλέτη", 18, 99, 45)
        adults     = colC.number_input("Ενήλικες στο νοικοκυριό (οφειλέτη)", 1, 6, 1)
        children   = colD.number_input("Ανήλικοι στο νοικοκυριό (οφειλέτη)", 0, 6, 0)

        col1, col2, col3 = st.columns(3)
        annual_income_main = col1.number_input("Ετήσιο καθαρό εισόδημα (οφειλέτη) €", 0.0, 1e9, 24000.0, step=500.0)
        monthly_income_main = annual_income_main / 12.0
        property_value = col2.number_input("Σύνολο αξίας ακίνητης περιουσίας (€)", 0.0, 1e9, 0.0, step=1000.0)
        annual_rate_pct= col3.number_input("Επιτόκιο ετησίως (%) (ενδεικτικό)", 0.0, 30.0, 6.0, step=0.1)

        # Συνοφειλέτες (μορφή JSON λίστας)
        st.subheader("Συνοφειλέτες (προαιρετικά)")
        st.caption("Λίστα αντικειμένων με: name, annual_income, property_value, age, adults, children.\n"
                   "Παράδειγμα: "
                   "[{'name':'Μαρία','annual_income':12000,'property_value':0,'age':40,'adults':1,'children':1}]")
        co_raw = st.text_area("Δώσε JSON (ή άφησε κενό)", "")

        # Επιπλέον Δαπάνες
        st.subheader("Επιπλέον Δαπάνες (πέραν ΕΔΔ)")
        c1,c2,c3 = st.columns(3)
        extra_medical = c1.number_input("Ιατρικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)
        extra_students= c2.number_input("Φοιτητές / Σπουδές (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)
        extra_legal   = c3.number_input("Δικαστικά (€ / μήνα)", 0.0, 100000.0, 0.0, step=10.0)

        st.markdown("---")

        # Οφειλές
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

        # ΕΔΔ (οφειλέτη μόνο)
        st.subheader("Ελάχιστες Δαπάνες Διαβίωσης (οφειλέτη)")
        use_manual = st.checkbox("Χειροκίνητη εισαγωγή ΕΔΔ;", value=False)
        if use_manual:
            edd_val = st.number_input("ΕΔΔ νοικοκυριού (οφειλέτη) € / μήνα", 0.0, 10000.0, 800.0, step=10.0)
        else:
            edd_val = compute_edd(int(adults), int(children))
            st.info(f"Αυτόματος υπολογισμός ΕΔΔ οφειλέτη: **{edd_val:,.2f} €**")

        submitted = st.form_submit_button("Υπολογισμός Πρόβλεψης & Αποθήκευση", use_container_width=True)

    if submitted:
        # Parse συνοφειλέτες
        try:
            co_list = json.loads(co_raw) if co_raw.strip() else []
            if not isinstance(co_list, list): co_list = []
        except Exception:
            co_list = []
            st.warning("Μη έγκυρο JSON στους συνοφειλέτες. Αγνοήθηκε.")

        # Υπολογισμός μηνιαίου εισοδήματος & ΕΔΔ με συνεκτίμηση συνοφειλετών
        monthly_income_codes = 0.0
        edd_codes = 0.0
        for c in co_list:
            aincome = float(c.get("annual_income") or 0.0)
            monthly_income_codes += aincome/12.0
            cadults = int(c.get("adults") or 1)
            cchildren = int(c.get("children") or 0)
            edd_codes += compute_edd(cadults, cchildren)

        monthly_income = monthly_income_main + monthly_income_codes
        edd_total_house = edd_val + edd_codes

        # Συγκεντρωτικά για header/PDF
        debts = debts_df.fillna(0).to_dict(orient="records")
        total_debt = sum([float(d["balance"] or 0) for d in debts])
        secured_amt = sum([float(d["collateral_value"] or 0) for d in debts if d.get("secured")])

        extras_sum = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
        avail = available_income(monthly_income, edd_total_house, extra_medical, extra_students, extra_legal)

        age_cap_months = months_cap_from_age(int(debtor_age))

        # ── Rule-based υπολογισμός ΑΝΑ ΟΦΕΙΛΗ ──
        enriched = []
        for d in debts:
            enriched.append({
                "creditor": str(d.get("creditor","")).strip(),
                "loan_type": d.get("loan_type",""),
                "balance": float(d.get("balance",0) or 0.0),
                "secured": bool(d.get("secured")),
                "collateral_value": float(d.get("collateral_value",0) or 0.0),
                "cat": classify_debt(str(d.get("creditor","")).strip(), bool(d.get("secured")))
            })

        if POLICY["allocate"] == "proportional":
            shares = split_available_proportional(avail, enriched)
        else:
            shares = split_available_priority(avail, enriched)

        per_debt_rows = []
        for i, d in enumerate(enriched):
            r = compute_offer_per_debt(d, monthly_share=shares.get(i,0.0), age_cap=age_cap_months)
            d.update(r)
            per_debt_rows.append({
                "Πιστωτής": d["creditor"],
                "Είδος": d["loan_type"],
                "Υπόλοιπο (€)": d["balance"],
                "Εξασφαλισμένο": "Ναι" if d["secured"] else "Όχι",
                "Εξασφάλιση (€)": d["collateral_value"] if d["secured"] else 0.0,
                "Οροφή μηνών": d["term_cap"],
                "Πρόταση δόσης (€)": d["predicted_monthly"],
                "Διαγραφή (€)": d["predicted_writeoff"],
                "Υπόλοιπο ρύθμισης (€)": d["predicted_residual"],
                "Κούρεμα (%)": d["predicted_haircut_pct"],
            })

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("Οι προτάσεις δίνονται **ανά οφειλή**. Κατανομή διαθέσιμου: "
                + ("προτεραιοτήτων (Δημόσιο→Εξασφαλισμένα→Λοιπά)." if POLICY["allocate"]=="priority_first" else "αναλογικά βάσει υπολοίπου."))

        # Αποθήκευση υπόθεσης
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        # γράφουμε πίσω τα enrich πεδία στον πίνακα debts
        debts_to_store = []
        for d in enriched:
            debts_to_store.append({
                "creditor": d["creditor"],
                "loan_type": d["loan_type"],
                "balance": d["balance"],
                "secured": d["secured"],
                "collateral_value": d["collateral_value"],
                "term_cap": d["term_cap"],
                "predicted_monthly": d["predicted_monthly"],
                "predicted_writeoff": d["predicted_writeoff"],
                "predicted_residual": d["predicted_residual"],
                "predicted_haircut_pct": d["predicted_haircut_pct"],
            })

        row = {
            "case_id": case_id,
            "borrower": borrower,
            "debtor_age": int(debtor_age),
            "adults": int(adults),
            "children": int(children),

            # συνολικό ΜΗΝΙΑΙΟ εισόδημα (οφειλέτη + συνοφειλέτες)
            "monthly_income": float(monthly_income),

            "property_value": float(property_value),
            "annual_rate_pct": float(annual_rate_pct),

            # ΕΔΔ & extra δαπάνες
            "edd_use_manual": 1 if use_manual else 0,
            "edd_manual": float(edd_val),
            "extra_medical": float(extra_medical or 0),
            "extra_students": float(extra_students or 0),
            "extra_legal": float(extra_legal or 0),

            # Κόφτης ηλικίας
            "age_cap": int(age_cap_months),

            # Οφειλές με πεδία πρόβλεψης
            "debts_json": json.dumps(debts_to_store, ensure_ascii=False),

            # Συνοφειλέτες (όπως δόθηκαν)
            "co_debtors_json": json.dumps(co_list, ensure_ascii=False),

            # Πραγματικές ρυθμίσεις ανα οφειλή (άδειο αρχικά)
            "real_debts_json": json.dumps([], ensure_ascii=False),

            # Συνολικά πεδία (μη χρησιμοποιούνται εδώ)
            "term_months": None,
            "predicted_at": now_str,
            "predicted_monthly": None,
            "predicted_haircut_pct": None,
            "prob_accept": None,

            # Roll-up πραγματικής ρύθμισης (προαιρετικά/μελλοντικά)
            "real_monthly": None,
            "real_haircut_pct": None,
            "accepted": None,
            "real_term_months": None,
            "real_writeoff_amount": None,
            "real_residual_balance": None
        }
        save_data(pd.DataFrame([row]))
        st.success(f"✅ Αποθηκεύτηκε η πρόβλεψη: {case_id}")

        # PDF
        case_for_pdf = {
            "case_id": case_id,
            "borrower": borrower,
            "debtor_age": int(debtor_age),
            "adults": int(adults),
            "children": int(children),
            "monthly_income": float(monthly_income),
            "edd_household": float(edd_total_house),
            "extras_sum": float(extras_sum),
            "avail": float(avail),
            "property_value": float(property_value),
            "debts": debts_to_store,
            "predicted_at": now_str
        }
        pdf_bytes = make_pdf(case_for_pdf)
        st.download_button(
            "⬇️ Λήψη Πρόβλεψης (PDF)",
            data=pdf_bytes,
            file_name=f"{case_id}_prediction.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ─────────────────────── ΠΡΟΒΛΕΨΕΙΣ & ΠΡΑΓΜΑΤΙΚΕΣ ΡΥΘΜΙΣΕΙΣ ───────────────────────
elif page == "Προβλέψεις & Πραγματικές Ρυθμίσεις":
    st.title("📁 Προβλέψεις & Πραγματικές Ρυθμίσεις")
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
    else:
        dfv = df_all.copy()
        dfv = dfv[["case_id","borrower","predicted_at"]].sort_values("predicted_at", ascending=False)
        st.dataframe(dfv, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Άνοιγμα υπόθεσης")

        case_ids = df_all["case_id"].tolist()
        pick = st.selectbox("Διάλεξε Υπόθεση", case_ids)
        if pick:
            row = df_all[df_all["case_id"]==pick].iloc[0].to_dict()
            debts = []
            try:
                debts = json.loads(row.get("debts_json") or "[]")
            except Exception:
                debts = []

            st.write(f"**Οφειλέτης:** {row.get('borrower','')}  |  **Ημερομηνία πρόβλεψης:** {row.get('predicted_at','')}")

            st.markdown("#### Πραγματική ρύθμιση ανά οφειλή")
            real_list = []
            for i, d in enumerate(debts):
                with st.expander(f"Οφειλή #{i+1} – {d.get('creditor','')} / {d.get('loan_type','')} / Υπόλοιπο: {float(d.get('balance',0)):,.2f} €"):
                    col1,col2,col3,col4 = st.columns(4)
                    real_term    = col1.number_input("Πραγμ. μήνες", 0, 1200, 0, key=f"rt_{i}")
                    real_monthly = col2.number_input("Πραγμ. δόση (€)", 0.0, 1e9, 0.0, step=10.0, key=f"rm_{i}")
                    real_write   = col3.number_input("Διαγραφή (€)", 0.0, float(d.get("balance",0) or 0.0), 0.0, step=100.0, key=f"rw_{i}")
                    real_resid   = max(0.0, float(d.get("balance",0) or 0.0) - float(real_write or 0.0))
                    col4.metric("Υπόλοιπο ρύθμισης (€)", f"{real_resid:,.2f}")

                    haircut_pct = 0.0 if (float(d.get("balance",0) or 0.0) <= 0) else 100.0 * (float(real_write or 0.0) / float(d.get("balance") or 1.0))
                    st.caption(f"Ποσοστό κουρέματος: **{haircut_pct:.1f}%**")

                    real_list.append({
                        "creditor": d.get("creditor",""),
                        "loan_type": d.get("loan_type",""),
                        "balance": float(d.get("balance",0) or 0.0),
                        "real_term_months": int(real_term) if real_term else None,
                        "real_monthly": float(real_monthly) if real_monthly else None,
                        "real_writeoff": float(real_write) if real_write else None,
                        "real_residual": float(real_resid),
                        "real_haircut_pct": float(haircut_pct)
                    })

            if st.button("💾 Αποθήκευση πραγματικής ρύθμισης", type="primary"):
                row_update = row.copy()
                row_update["real_debts_json"] = json.dumps(real_list, ensure_ascii=False)

                # Προαιρετικό roll-up
                try:
                    monthly_vals = [x.get("real_monthly") for x in real_list if x.get("real_monthly") is not None]
                    row_update["real_monthly"] = float(np.mean(monthly_vals)) if monthly_vals else None

                    total_bal = sum([x.get("balance",0.0) for x in real_list])
                    total_write = sum([x.get("real_writeoff",0.0) or 0.0 for x in real_list])
                    row_update["real_haircut_pct"] = (100.0*total_write/total_bal) if total_bal>0 else None
                except Exception:
                    pass

                save_data(pd.DataFrame([row_update]))
                st.success("✅ Αποθηκεύτηκε η πραγματική ρύθμιση για την υπόθεση.")
