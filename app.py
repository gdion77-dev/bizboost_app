# app.py
# Bizboost - Εξωδικαστικός: Πρόβλεψη & Καταγραφή Ρυθμίσεων (Streamlit + Postgres + PDF)

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# PDF / ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ────────────────────────────── UI / PATHS ──────────────────────────────
st.set_page_config(page_title="Bizboost - Εξωδικαστικός", page_icon="💠", layout="wide")

# session defaults
if "open_case_id" not in st.session_state:
    st.session_state.open_case_id = None

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
DATA_CSV  = os.path.join(BASE_DIR, "cases.csv")

# ───────── Γραμματοσειρές PDF (NotoSans → NotoSerif → fallback) ─────────
FONT_DIR = os.path.join(BASE_DIR, "assets", "fonts")
FONT_CANDIDATES = [
    ("NotoSans",  os.path.join(FONT_DIR, "NotoSans-Regular.ttf")),
    ("NotoSerif", os.path.join(FONT_DIR, "NotoSerif-Regular.ttf")),
]
PDF_FONT = "Helvetica"
try:
    chosen = None
    for name, path in FONT_CANDIDATES:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 100_000:
                pdfmetrics.registerFont(TTFont(name, path))
                chosen = name
                break
        except Exception:
            continue
    if chosen:
        PDF_FONT = chosen
    else:
        st.warning("Δεν βρέθηκε έγκυρο TTF (NotoSans/NotoSerif). Θα χρησιμοποιηθεί Helvetica.")
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
    "priority": ["PUBLIC", "SECURED", "UNSECURED"],
    "term_caps": {"PUBLIC": 240, "BANK": 420, "DEFAULT": 240},
    "allocate": "priority_first",
    "max_haircut": {"PUBLIC": None, "BANK": None, "DEFAULT": None},
}

# ─────────────────────── ΕΔΔ & ΔΙΑΘΕΣΙΜΑ ───────────────────────
def compute_edd(adults:int, children:int)->float:
    if adults <= 0 and children <= 0:
        return 0.0
    base_adult = 537; add_adult = 269; per_child = 211
    if adults <= 0: adults = 1
    return float(base_adult + max(adults-1,0)*add_adult + children*per_child)

def months_cap_from_age(age:int)->int:
    try: a = int(age)
    except: return 120
    if a <= 35:  return 240
    if a <= 50:  return 180
    if a <= 65:  return 120
    return 60

def available_income(total_income:float, edd_household:float, e_med:float, e_stud:float, e_leg:float)->float:
    extras = (e_med or 0) + (e_stud or 0) + (e_leg or 0)
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
    if not secured: return 0.0
    return max(0.0, float(balance or 0.0) - float(collateral_value or 0.0))

def split_available_proportional(avail:float, debts:list)->dict:
    total = sum(d["balance"] for d in debts if d["balance"]>0)
    if total <= 0: return {i:0.0 for i in range(len(debts))}
    return {i: avail * (d["balance"]/total) for i,d in enumerate(debts)}

def split_available_priority(avail:float, debts:list)->dict:
    out = {i:0.0 for i in range(len(debts))}
    groups = {"PUBLIC":[], "SECURED":[], "UNSECURED":[]}
    for i,d in enumerate(debts):
        if d["cat"] == "PUBLIC": groups["PUBLIC"].append(i)
        elif d["secured"]:       groups["SECURED"].append(i)
        else:                    groups["UNSECURED"].append(i)
    remaining = avail
    for key in POLICY["priority"]:
        idxs = groups.get(key, [])
        if not idxs: continue
        subtotal = sum(debts[i]["balance"] for i in idxs if debts[i]["balance"]>0)
        if subtotal <= 0: continue
        for i in idxs:
            out[i] += remaining * (debts[i]["balance"]/subtotal)
        remaining = 0.0
        if remaining <= 0: break
    return out

def compute_offer_per_debt(d, monthly_share, age_cap):
    term = term_cap_for(d["creditor"], age_cap, d["secured"])
    inst = max(0.0, float(monthly_share))
    gross_residual = max(0.0, d["balance"] - inst*term)
    floor = security_floor(d["balance"], d["secured"], d.get("collateral_value",0.0))
    residual = max(gross_residual, floor)
    writeoff = max(0.0, d["balance"] - residual)
    haircut = 0.0 if d["balance"]<=0 else 100.0*writeoff/(d["balance"]+1e-6)

    cat = classify_debt(d["creditor"], d["secured"])
    max_hc = POLICY["max_haircut"].get(cat)
    if isinstance(max_hc, (int,float)) and max_hc is not None:
        max_write = d["balance"]*max_hc
        if writeoff > max_write:
            writeoff = max_write
            residual = d["balance"] - writeoff
            haircut  = 100.0*max_hc

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
        st.error("Δεν έχει οριστεί DATABASE_URL στα Secrets."); st.stop()
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
    engine = get_db_engine(); init_db(engine)
    try:
        return pd.read_sql("SELECT * FROM cases", con=engine)
    except Exception as e:
        st.error(f"Σφάλμα ανάγνωσης DB: {e}")
        return pd.DataFrame()

def upsert_cases_db(df: pd.DataFrame):
    if df.empty: return
    engine = get_db_engine(); init_db(engine)
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

def delete_case_db(case_id: str):
    engine = get_db_engine(); init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM cases WHERE case_id = :cid"), {"cid": case_id})

def csv_to_db_once_if_empty():
    engine = get_db_engine(); init_db(engine)
    with engine.begin() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM cases")).scalar()
    if cnt == 0 and os.path.exists(DATA_CSV):
        try:
            dfcsv = pd.read_csv(DATA_CSV)
            for c in ["debts_json","co_debtors_json","real_debts_json"]:
                if c in dfcsv.columns: dfcsv[c] = dfcsv[c].fillna("[]")
            upsert_cases_db(dfcsv)
            st.success("Έγινε αρχικό import από cases.csv")
        except Exception as e:
            st.warning(f"Αποτυχία import από cases.csv: {e}")

def load_data():
    csv_to_db_once_if_empty()
    return load_data_db()

def save_data(df: pd.DataFrame):
    upsert_cases_db(df)

# ───────── Safe helpers ─────────
def parse_json_field(val):
    """Δέχεται JSONB (list/dict) ή string JSON και επιστρέφει πάντα list/dict ή []."""
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            return json.loads(s)
        except Exception:
            return []
    return []

def _to_float(x, default=0.0):
    try:
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def eur_num(x) -> str: return f"{_to_float(x):,.2f}"
def eur(x) -> str:     return f"{eur_num(x)} €"
def pct1(x) -> str:    return f"{_to_float(x):.1f}%"
def s(x) -> str:       return "" if x is None else str(x)

# ────────────────────────────── PDF EXPORT (ReportLab) ──────────────────────────────
def make_pdf(case_dict:dict)->bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2.2*cm)
    styles = getSampleStyleSheet()
    base_font = PDF_FONT
    styles.add(ParagraphStyle(name="H1", fontName=base_font, fontSize=16, leading=20, spaceAfter=10,
                              textColor=colors.HexColor("#0F4C81"), alignment=1))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, leading=16, spaceAfter=6,
                              textColor=colors.HexColor("#333333")))
    styles.add(ParagraphStyle(name="P", fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="SmallCenter", fontName=base_font, fontSize=8, leading=11,
                              alignment=1, textColor=colors.HexColor("#666")))

    def _available_width(doc_): return doc_.pagesize[0] - doc_.leftMargin - doc_.rightMargin
    def _cm_list_to_points(widths_cm, doc_):
        pts = [w*cm for w in widths_cm]; total = sum(pts); avail = _available_width(doc_)
        if total > avail and total > 0: pts = [p*(avail/total) for p in pts]
        return pts

    def _personalized_reasoning(cd):
        mi   = _to_float(cd.get("monthly_income"))
        edd  = _to_float(cd.get("edd_household"))
        extra= _to_float(cd.get("extras_sum"))
        avail= _to_float(cd.get("avail"))
        debts= cd.get("debts",[]) or []
        public_cnt  = sum(1 for d in debts if str(d.get("creditor","")) in PUBLIC_CREDITORS)
        secured_cnt = sum(1 for d in debts if bool(d.get("secured")))
        other_cnt   = max(0, len(debts) - public_cnt - secured_cnt)
        public_terms= sorted({int(d.get("term_cap",0) or 0) for d in debts if str(d.get("creditor","")) in PUBLIC_CREDITORS and d.get("term_cap")})
        bank_terms  = sorted({int(d.get("term_cap",0) or 0) for d in debts if str(d.get("creditor","")) in BANK_SERVICERS and d.get("term_cap")})
        line1 = f"Η πρόταση βασίζεται σε καθαρό διαθέσιμο **{eur(avail)}** (εισόδημα **{eur(mi)}** − ΕΔΔ **{eur(edd)}** − πρόσθετες δαπάνες **{eur(extra)}**)."
        parts = []
        if public_cnt:
            cap_info = f"με όριο **{(max(public_terms) if public_terms else 240)} μήνες**" if public_terms else "έως **240 μήνες**"
            parts.append(f"Δημόσιο (ΑΑΔΕ/ΕΦΚΑ, {public_cnt} οφ.): {cap_info}.")
        if secured_cnt:
            parts.append("Εξασφαλισμένες οφειλές: εφαρμόστηκε κατώφλι εξασφάλισης.")
        if other_cnt:
            cap_bank = f"{max(bank_terms)} μήνες" if bank_terms else "έως **420 μήνες**"
            parts.append(f"Λοιπές τραπεζικές/servicers ({other_cnt} οφ.): μέγιστη διάρκεια {cap_bank}.")
        dist = "Κατανομή διαθέσιμου: **Δημόσιο → Εξασφαλισμένα → Λοιπά**."
        end  = "Υπόλοιπο ρύθμισης = Υπόλοιπο − Διαγραφή, Κούρεμα = Διαγραφή / Υπόλοιπο."
        return " ".join([line1, *parts, dist, end])

    story = []

    # Λογότυπο/label
    if os.path.exists(LOGO_PATH):
        try:
            img = Image(LOGO_PATH, width=150); img.hAlign = 'CENTER'
            story.append(img); story.append(Spacer(1, 6))
        except Exception:
            pass
    else:
        story.append(Paragraph("The Bizboost by G. Dionysiou", styles["H1"]))

    story.append(Paragraph("Πρόβλεψη Ρύθμισης (Εξωδικαστικός)", styles["H1"]))

    # Summary table
    meta = [
        ["Υπόθεση",         s(case_dict.get("case_id",""))],
        ["Οφειλέτης",       s(case_dict.get("borrower",""))],
        ["Ηλικία",          s(case_dict.get("debtor_age",""))],
        ["Μέλη νοικοκυριού (ενήλ./ανήλ.)", f"{_to_float(case_dict.get('adults',0)):.0f}/{_to_float(case_dict.get('children',0)):.0f}"],
        ["Συνολικό μηνιαίο εισόδημα", eur(case_dict.get("monthly_income"))],
        ["ΕΔΔ νοικοκυριού",             eur(case_dict.get("edd_household"))],
        ["Επιπλέον δαπάνες",            eur(case_dict.get("extras_sum"))],
        ["Καθαρό διαθέσιμο",            eur(case_dict.get("avail"))],
        ["Ακίνητη περιουσία",           eur(case_dict.get("property_value"))],
        ["Ημερομηνία",       s(case_dict.get("predicted_at",""))],
    ]
    t = Table(meta, colWidths=_cm_list_to_points([6.0, 9.5], doc))
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
    story.append(t); story.append(Spacer(1, 10))

    # Debts table
    debts = case_dict.get("debts", []) or []
    if debts:
        story.append(Paragraph("Αναλυτικά ανά οφειλή (πρόβλεψη):", styles["H2"]))
        rows = [["Πιστωτής","Είδος","Υπόλοιπο (€)","Εξασφαλ.","Εξασφάλιση (€)","Οροφή μηνών","Πρόταση δόσης (€)","Διαγραφή (€)","Υπόλοιπο Ρύθμισης (€)","Κούρεμα (%)"]]
        for d in debts:
            rows.append([
                s(d.get("creditor","")),
                s(d.get("loan_type","")),
                eur_num(d.get("balance")),
                "Ναι" if bool(d.get("secured")) else "Όχι",
                eur_num(d.get("collateral_value")) if bool(d.get("secured")) else "0.00",
                f"{_to_float(d.get('term_cap')):.0f}",
                eur_num(d.get("predicted_monthly")),
                eur_num(d.get("predicted_writeoff")),
                eur_num(d.get("predicted_residual")),
                pct1(d.get("predicted_haircut_pct")),
            ])
        widths_cm = [2.6, 2.1, 2.2, 1.0, 1.8, 1.6, 2.1, 2.1, 2.2, 1.1]
        tt = Table(rows, colWidths=_cm_list_to_points(widths_cm, doc), repeatRows=1)
        tt.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), base_font, 9),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F4C81")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (2,1), (-1,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#DDD")),
            ("BOX", (0,0), (-1,-1), 0.6, colors.HexColor("#0F4C81")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(tt); story.append(Spacer(1, 10))

    # Reasoning
    story.append(Paragraph("Σκεπτικό πρότασης", styles["H2"]))
    story.append(Paragraph(_personalized_reasoning(case_dict), styles["P"]))
    story.append(Spacer(1, 12))

    # Footer
    CONTACT_NAME   = "Γεώργιος Φ. Διονυσίου Οικονομολόγος BA, MSc"
    CONTACT_PHONE  = "+30 2273081618"
    CONTACT_EMAIL  = "info@bizboost.gr"
    CONTACT_SITE   = "www.bizboost.gr"
    CONTACT_ADDRESS= "Αγίου Νικολάου 1, Σάμος 83100"

    from reportlab.platypus import Flowable
    class HR(Flowable):
        def __init__(self, width=1, color=colors.HexColor("#DDD")):
            super().__init__(); self.width = width; self.color = color; self.height = 6
        def draw(self):
            c = self.canv; w = c._pagesize[0] - doc.leftMargin - doc.rightMargin
            x0 = doc.leftMargin; y  = 2
            c.setStrokeColor(self.color); c.setLineWidth(self.width); c.line(x0, y, x0 + w, y)

    story.append(HR()); story.append(Spacer(1, 4))
    story.append(Paragraph(f"{CONTACT_NAME} • Τ: {CONTACT_PHONE} • E: {CONTACT_EMAIL} • {CONTACT_SITE}", styles["SmallCenter"]))
    story.append(Paragraph(f"{CONTACT_ADDRESS}", styles["SmallCenter"]))

    doc.build(story); buf.seek(0); return buf.read()

# ────────────────────────────── CASE DETAIL PAGE ──────────────────────────────
def show_case_detail(df_all: pd.DataFrame, case_id: str):
    rowdf = df_all[df_all["case_id"]==case_id]
    if rowdf.empty:
        st.error("Η υπόθεση δεν βρέθηκε.")
        return
    row = rowdf.iloc[0].to_dict()

    debts = parse_json_field(row.get("debts_json"))
    real_debts = parse_json_field(row.get("real_debts_json"))

    st.title(f"📄 Υπόθεση {case_id}")
    st.caption(f"Οφειλέτης: **{row.get('borrower','')}** — Πρόβλεψη: {row.get('predicted_at','')}")

    # Back button
    if st.button("⬅︎ Πίσω στις υποθέσεις", key=f"back_{case_id}"):
        st.session_state.open_case_id = None
        st.rerun()

    # Πρόβλεψη (ανά οφειλή)
    if debts:
        st.subheader("Πρόβλεψη ανά οφειλή")
        df_pred = pd.DataFrame(debts)
        st.dataframe(df_pred, use_container_width=True, hide_index=True)
    else:
        st.info("Δεν υπάρχουν οφειλές καταχωρημένες στην πρόβλεψη.")

    # Φόρμα Πραγματικής Ρύθμισης
    st.subheader("Καταχώριση Πραγματικής Ρύθμισης")
    real_list = []
    for i, d in enumerate(debts):
        with st.expander(f"Οφειλή #{i+1} – {d.get('creditor','')} / {d.get('loan_type','')} / Υπόλοιπο: {eur_num(d.get('balance'))} €", expanded=False):
            # Προσυμπλήρωση από αποθηκευμένα real_debts αν υπάρχουν
            pre = None
            for r in (real_debts or []):
                if r.get("creditor")==d.get("creditor") and r.get("loan_type")==d.get("loan_type") and _to_float(r.get("balance"))==_to_float(d.get("balance")):
                    pre = r; break
            col1,col2,col3,col4 = st.columns(4)
            real_term    = col1.number_input("Πραγμ. μήνες", 0, 1200, int(pre.get("real_term_months") or 0) if pre else 0, key=f"rt_{case_id}_{i}")
            real_monthly = col2.number_input("Πραγμ. δόση (€)", 0.0, 1e9, float(pre.get("real_monthly") or 0.0) if pre else 0.0, step=10.0, key=f"rm_{case_id}_{i}")
            real_write   = col3.number_input("Διαγραφή (€)", 0.0, _to_float(d.get("balance")), float(pre.get("real_writeoff") or 0.0) if pre else 0.0, step=100.0, key=f"rw_{case_id}_{i}")
            real_resid   = max(0.0, _to_float(d.get("balance")) - _to_float(real_write))
            col4.metric("Υπόλοιπο ρύθμισης (€)", f"{eur_num(real_resid)}")
            haircut_pct = 0.0 if _to_float(d.get("balance"))<=0 else 100.0*(_to_float(real_write)/max(1.0,_to_float(d.get("balance"))))
            st.caption(f"Ποσοστό κουρέματος: **{haircut_pct:.1f}%**")
            real_list.append({
                "creditor": d.get("creditor",""),
                "loan_type": d.get("loan_type",""),
                "balance": _to_float(d.get("balance")),
                "real_term_months": int(real_term) if real_term else None,
                "real_monthly": _to_float(real_monthly) if real_monthly else None,
                "real_writeoff": _to_float(real_write) if real_write else None,
                "real_residual": _to_float(real_resid),
                "real_haircut_pct": float(haircut_pct)
            })

    # Αποθήκευση πραγματικής ρύθμισης
    if debts and st.button("💾 Αποθήκευση πραγματικής ρύθμισης", type="primary", key=f"save_real_{case_id}"):
        row_update = row.copy()
        row_update["real_debts_json"] = json.dumps(real_list, ensure_ascii=False)
        try:
            monthly_vals = [x.get("real_monthly") for x in real_list if x.get("real_monthly") is not None]
            row_update["real_monthly"] = float(np.mean(monthly_vals)) if monthly_vals else None
            total_bal = sum([x.get("balance",0.0) for x in real_list])
            total_write = sum([x.get("real_writeoff",0.0) or 0.0 for x in real_list])
            row_update["real_haircut_pct"] = (100.0*total_write/total_bal) if total_bal>0 else None
        except Exception:
            pass
        save_data(pd.DataFrame([row_update]))
        st.success("✅ Αποθηκεύτηκε η πραγματική ρύθμιση.")
        st.rerun()

    # Σύγκριση (αν υπάρχουν αποθηκευμένα)
    real_debts_saved = parse_json_field(row.get("real_debts_json"))
    if real_debts_saved:
        st.subheader("Σύγκριση Πρόβλεψης vs Πραγματικής Ρύθμισης")
        def key_of(x):
            return f"{x.get('creditor','')}|{x.get('loan_type','')}|{round(float(x.get('balance',0) or 0.0),2)}"
        real_map = {key_of(r): r for r in (real_debts_saved or [])}
        rows_cmp = []
        for d in debts:
            k = key_of(d); r = real_map.get(k)
            rows_cmp.append({
                "Πιστωτής": d.get("creditor",""),
                "Είδος": d.get("loan_type",""),
                "Υπόλοιπο (€)": eur_num(d.get("balance",0)),
                "Πρόβλ. Δόση (€)": eur_num(d.get("predicted_monthly",0)),
                "Πρόβλ. Διαγραφή (€)": eur_num(d.get("predicted_writeoff",0)),
                "Πρόβλ. Υπόλ. Ρύθμισης (€)": eur_num(d.get("predicted_residual",0)),
                "Πραγμ. Μήνες": int(r.get("real_term_months")) if (r and r.get("real_term_months") is not None) else "—",
                "Πραγμ. Δόση (€)": eur_num(r.get("real_monthly")) if (r and r.get("real_monthly") is not None) else "—",
                "Πραγμ. Διαγραφή (€)": eur_num(r.get("real_writeoff")) if (r and r.get("real_writeoff") is not None) else "—",
                "Πραγμ. Υπόλ. (€)": eur_num(r.get("real_residual")) if (r and r.get("real_residual") is not None) else "—",
                "Διαφορά Δόσης (€)": eur_num((r.get("real_monthly") - d.get("predicted_monthly")) if r and r.get("real_monthly") is not None else 0.0),
                "Διαφορά Διαγραφής (€)": eur_num((r.get("real_writeoff") - d.get("predicted_writeoff")) if r and r.get("real_writeoff") is not None else 0.0),
                "Διαφορά Υπολοίπου (€)": eur_num((r.get("real_residual") - d.get("predicted_residual")) if r and r.get("real_residual") is not None else 0.0),
            })
        st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True)

# ────────────────────────────── UI ──────────────────────────────
# Sidebar brand
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost")
else:
    st.sidebar.markdown("### **Bizboost**")

page = st.sidebar.radio("Μενού", ["Νέα Πρόβλεψη", "Προβλέψεις & Πραγματικές Ρυθμίσεις"], index=0)
df_all = load_data()

# Direct open via URL (?case_id=...)
try:
    q = st.query_params
except Exception:
    q = st.experimental_get_query_params()
cid = None
if isinstance(q, dict):
    if "case_id" in q:
        cid = q["case_id"]
        if isinstance(cid, list): cid = cid[0]
if cid:
    show_case_detail(df_all, cid); st.stop()

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
        annual_rate_pct= col3.number_input("Επιτόκιο ετησίως (%) (πληροφ.)", 0.0, 30.0, 6.0, step=0.1)

        st.markdown("### Συνοφειλέτες (προαιρετικά)")
        codebtors_df_default = pd.DataFrame([{
            "name": "", "annual_income": 0.0, "property_value": 0.0, "age": 40, "adults": 1, "children": 0
        }])
        codebtors_df = st.data_editor(
            codebtors_df_default, num_rows="dynamic", use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Ονοματεπώνυμο"),
                "annual_income": st.column_config.NumberColumn("Ετήσιο εισόδημα (€)", step=500.0, format="%.2f"),
                "property_value": st.column_config.NumberColumn("Ακίνητη περιουσία (€)", step=1000.0, format="%.2f"),
                "age": st.column_config.NumberColumn("Ηλικία", min_value=18, max_value=99, step=1),
                "adults": st.column_config.NumberColumn("Ενήλικες", min_value=1, max_value=6, step=1),
                "children": st.column_config.NumberColumn("Ανήλικοι", min_value=0, max_value=6, step=1),
            }
        )

        st.markdown("### Επιπλέον Δαπάνες (πέραν ΕΔΔ)")
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
            default_debts, num_rows="dynamic", use_container_width=True,
            column_config={
                "creditor": st.column_config.SelectboxColumn("Πιστωτής", options=CREDITORS),
                "loan_type": st.column_config.SelectboxColumn("Είδος δανείου", options=LOAN_TYPES),
                "balance": st.column_config.NumberColumn("Υπόλοιπο (€)", step=500.0, format="%.2f"),
                "secured": st.column_config.CheckboxColumn("Εξασφαλισμένο"),
                "collateral_value": st.column_config.NumberColumn("Ποσό εξασφάλισης (€)", step=500.0, format="%.2f"),
            }
        )

        st.subheader("Ελάχιστες Δαπάνες Διαβίωσης (οφειλέτη)")
        use_manual = st.checkbox("Χειροκίνητη εισαγωγή ΕΔΔ;", value=False)
        if use_manual:
            edd_val = st.number_input("ΕΔΔ νοικοκυριού (οφειλέτη) € / μήνα", 0.0, 10000.0, 800.0, step=10.0)
        else:
            edd_val = compute_edd(int(adults), int(children))
            st.info(f"Αυτόματος υπολογισμός ΕΔΔ οφειλέτη: **{edd_val:,.2f} €**")

        submitted = st.form_submit_button("Υπολογισμός Πρόβλεψης & Αποθήκευση", use_container_width=True)

    if submitted:
        # Συνοφειλέτες
        codebtors = codebtors_df.fillna(0).to_dict(orient="records")
        monthly_income_codes = 0.0; edd_codes = 0.0
        for c in codebtors:
            monthly_income_codes += float(c.get("annual_income") or 0.0)/12.0
            edd_codes += compute_edd(int(c.get("adults") or 1), int(c.get("children") or 0))

        monthly_income = float(annual_income_main/12.0 + monthly_income_codes)
        edd_total_house = float(edd_val + edd_codes)

        debts = debts_df.fillna(0).to_dict(orient="records")
        extras_sum = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
        avail = available_income(monthly_income, edd_total_house, extra_medical, extra_students, extra_legal)
        age_cap_months = months_cap_from_age(int(debtor_age))

        enriched = []
        for d in debts:
            enriched.append({
                "creditor": str(d.get("creditor","")).strip(),
                "loan_type": d.get("loan_type",""),
                "balance": float(d.get("balance",0) or 0.0),
                "secured": bool(d.get("secured")),
                "collateral_value": float(d.get("collateral_value",0) or 0.0),
                "cat": classify_debt(str(d.get("creditor","")).strip(), bool(d.get("secured"))),
            })

        shares = split_available_priority(avail, enriched) if POLICY["allocate"]=="priority_first" else split_available_proportional(avail, enriched)

        per_debt_rows = []; debts_to_store = []
        for i, d in enumerate(enriched):
            r = compute_offer_per_debt(d, monthly_share=shares.get(i,0.0), age_cap=age_cap_months)
            d.update(r)
            per_debt_rows.append({
                "Πιστωτής": d["creditor"], "Είδος": d["loan_type"], "Υπόλοιπο (€)": d["balance"],
                "Εξασφαλισμένο": "Ναι" if d["secured"] else "Όχι",
                "Εξασφάλιση (€)": d["collateral_value"] if d["secured"] else 0.0,
                "Οροφή μηνών": d["term_cap"], "Πρόταση δόσης (€)": d["predicted_monthly"],
                "Διαγραφή (€)": d["predicted_writeoff"], "Υπόλοιπο ρύθμισης (€)": d["predicted_residual"],
                "Κούρεμα (%)": d["predicted_haircut_pct"],
            })
            debts_to_store.append({
                "creditor": d["creditor"], "loan_type": d["loan_type"], "balance": d["balance"],
                "secured": d["secured"], "collateral_value": d["collateral_value"],
                "term_cap": d["term_cap"], "predicted_monthly": d["predicted_monthly"],
                "predicted_writeoff": d["predicted_writeoff"], "predicted_residual": d["predicted_residual"],
                "predicted_haircut_pct": d["predicted_haircut_pct"],
            })

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("Κατανομή διαθέσιμου: Δημόσιο → Εξασφαλισμένα → Λοιπά (προτεραιότητα).")

        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        row = {
            "case_id": case_id, "borrower": borrower, "debtor_age": int(debtor_age),
            "adults": int(adults), "children": int(children),
            "monthly_income": float(monthly_income), "property_value": float(property_value),
            "annual_rate_pct": float(annual_rate_pct),
            "edd_use_manual": 1 if use_manual else 0, "edd_manual": float(edd_val),
            "extra_medical": float(extra_medical or 0), "extra_students": float(extra_students or 0),
            "extra_legal": float(extra_legal or 0), "age_cap": int(age_cap_months),
            "debts_json": json.dumps(debts_to_store, ensure_ascii=False),
            "co_debtors_json": json.dumps(codebtors, ensure_ascii=False),
            "real_debts_json": json.dumps([], ensure_ascii=False),
            "term_months": None, "predicted_at": now_str,
            "predicted_monthly": None, "predicted_haircut_pct": None, "prob_accept": None,
            "real_monthly": None, "real_haircut_pct": None, "accepted": None,
            "real_term_months": None, "real_writeoff_amount": None, "real_residual_balance": None
        }
        save_data(pd.DataFrame([row]))
        st.success(f"✅ Αποθηκεύτηκε η πρόβλεψη: {case_id}")

        # PDF Export
        case_for_pdf = {
            "case_id": case_id, "borrower": borrower, "debtor_age": int(debtor_age),
            "adults": int(adults), "children": int(children),
            "monthly_income": float(monthly_income), "edd_household": float(edd_total_house),
            "extras_sum": float(extras_sum), "avail": float(avail),
            "property_value": float(property_value), "debts": debts_to_store, "predicted_at": now_str
        }
        pdf_bytes = make_pdf(case_for_pdf)
        st.download_button("⬇️ Λήψη Πρόβλεψης (PDF)", data=pdf_bytes,
                           file_name=f"{case_id}_prediction.pdf", mime="application/pdf",
                           use_container_width=True)

# ─────────────────────── ΠΡΟΒΛΕΨΕΙΣ & ΠΡΑΓΜΑΤΙΚΕΣ ΡΥΘΜΙΣΕΙΣ ───────────────────────
else:
    st.title("📁 Προβλέψεις & Πραγματικές Ρυθμίσεις")

    # query params άνοιγμα νέου tab
    try:
        qp = st.query_params; qp_view = qp.get("view", None); qp_case = qp.get("case_id", None)
    except Exception:
        qp = st.experimental_get_query_params()
        qp_view = (qp.get("view", [None]) or [None])[0]
        qp_case = (qp.get("case_id", [None]) or [None])[0]
    if qp_view == "case" and qp_case:
        st.session_state.open_case_id = qp_case

    df_all = load_data()
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις."); st.stop()

    # Κάρτες υποθέσεων
    dfv = df_all[["case_id","borrower","predicted_at"]].fillna("").sort_values("predicted_at", ascending=False)
    st.markdown("#### Αποθηκευμένες υποθέσεις")
    cols_per_row = 3
    rows = [dfv.iloc[i:i+cols_per_row] for i in range(0, len(dfv), cols_per_row)]
    for chunk in rows:
        cc = st.columns(len(chunk))
        for idx, (_, rowc) in enumerate(chunk.iterrows()):
            with cc[idx]:
                cid = rowc["case_id"]
                st.markdown(f"**{rowc['borrower'] or '—'}**")
                st.caption(f"Υπόθεση: `{cid}`  \nΗμερ.: {rowc['predicted_at'] or '—'}")
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("📂 Άνοιγμα", key=f"open_{cid}", use_container_width=True):
                        st.session_state.open_case_id = cid
                        st.rerun()
                with c2:
                    st.markdown(f"[↗︎ Νέο παράθυρο](?view=case&case_id={cid})")
                if st.button("🗑️ Διαγραφή", key=f"del_{cid}", use_container_width=True):
                    delete_case_db(cid)
                    if st.session_state.open_case_id == cid:
                        st.session_state.open_case_id = None
                    st.success(f"Διαγράφηκε η υπόθεση {cid}")
                    st.rerun()

    st.markdown("---")

    open_id = st.session_state.open_case_id
    if not open_id:
        st.info("Πάτησε **Άνοιγμα** σε κάποια υπόθεση για να καταχωρήσεις την πραγματική ρύθμιση και να δεις τη σύγκριση.")
        st.stop()

    row = df_all[df_all["case_id"] == open_id]
    if row.empty:
        st.warning("Η υπόθεση δεν βρέθηκε (ίσως διαγράφηκε)."); st.stop()
    row = row.iloc[0].to_dict()

    debts = parse_json_field(row.get("debts_json"))

    st.subheader(f"Υπόθεση: {open_id}")
    st.write(f"**Οφειλέτης:** {row.get('borrower','—')}  \n**Ημερ. πρόβλεψης:** {row.get('predicted_at','—')}")

    if not debts:
        st.info("Δεν υπάρχουν οφειλές καταχωρημένες στην πρόβλεψη. Δεν είναι δυνατή η εισαγωγή πραγματικής ρύθμισης χωρίς οφειλές.")
        st.stop()

    # Φόρμα πραγματικής ρύθμισης (όπως στο detail)
    st.markdown("#### Πραγματική ρύθμιση ανά οφειλή")
    real_list = []
    for i, d in enumerate(debts):
        with st.expander(f"#{i+1} – {d.get('creditor','')} / {d.get('loan_type','')} / Υπόλοιπο: {eur_num(d.get('balance'))} €", expanded=True):
            col1,col2,col3,col4 = st.columns(4)
            real_term    = col1.number_input("Πραγμ. μήνες", 0, 1200, 0, key=f"rt_list_{open_id}_{i}")
            real_monthly = col2.number_input("Πραγμ. δόση (€)", 0.0, 1e9, 0.0, step=10.0, key=f"rm_list_{open_id}_{i}")
            real_write   = col3.number_input("Διαγραφή (€)", 0.0, _to_float(d.get("balance")), 0.0, step=100.0, key=f"rw_list_{open_id}_{i}")
            real_resid   = max(0.0, _to_float(d.get("balance")) - _to_float(real_write))
            col4.metric("Υπόλοιπο ρύθμισης (€)", f"{eur_num(real_resid)}")
            haircut_pct = 0.0 if _to_float(d.get("balance"))<=0 else 100.0*(_to_float(real_write)/max(1.0,_to_float(d.get("balance"))))
            st.caption(f"Ποσοστό κουρέματος: **{haircut_pct:.1f}%**")
            real_list.append({
                "creditor": d.get("creditor",""),
                "loan_type": d.get("loan_type",""),
                "balance": _to_float(d.get("balance")),
                "real_term_months": int(real_term) if real_term else None,
                "real_monthly": _to_float(real_monthly) if real_monthly else None,
                "real_writeoff": _to_float(real_write) if real_write else None,
                "real_residual": _to_float(real_resid),
                "real_haircut_pct": float(haircut_pct)
            })

    if st.button("💾 Αποθήκευση πραγματικής ρύθμισης", type="primary", key=f"save_real_list_{open_id}"):
        row_update = row.copy()
        row_update["real_debts_json"] = json.dumps(real_list, ensure_ascii=False)
        try:
            monthly_vals = [x.get("real_monthly") for x in real_list if x.get("real_monthly") is not None]
            row_update["real_monthly"] = float(np.mean(monthly_vals)) if monthly_vals else None
            total_bal = sum([x.get("balance",0.0) for x in real_list])
            total_write = sum([x.get("real_writeoff",0.0) or 0.0 for x in real_list])
            row_update["real_haircut_pct"] = (100.0*total_write/total_bal) if total_bal>0 else None
        except Exception:
            pass
        save_data(pd.DataFrame([row_update]))
        st.success("✅ Αποθηκεύτηκε η πραγματική ρύθμιση.")
        st.rerun()

    # Εμφάνιση σύγκρισης αν υπάρχουν αποθηκευμένα real
    real_debts_saved = parse_json_field(row.get("real_debts_json"))
    if real_debts_saved:
        st.markdown("#### Σύγκριση Πρόβλεψης vs Πραγματικής Ρύθμισης")
        def key_of(x): return f"{x.get('creditor','')}|{x.get('loan_type','')}|{round(float(x.get('balance',0) or 0.0),2)}"
        real_map = {key_of(r): r for r in (real_debts_saved or [])}
        rows_cmp = []
        for d in debts:
            k = key_of(d); r = real_map.get(k)
            rows_cmp.append({
                "Πιστωτής": d.get("creditor",""),
                "Είδος": d.get("loan_type",""),
                "Υπόλοιπο (€)": eur_num(d.get("balance",0)),
                "Πρόβλ. Δόση (€)": eur_num(d.get("predicted_monthly",0)),
                "Πρόβλ. Διαγραφή (€)": eur_num(d.get("predicted_writeoff",0)),
                "Πρόβλ. Υπόλ. Ρύθμισης (€)": eur_num(d.get("predicted_residual",0)),
                "Πραγμ. Μήνες": int(r.get("real_term_months")) if (r and r.get("real_term_months") is not None) else "—",
                "Πραγμ. Δόση (€)": eur_num(r.get("real_monthly")) if (r and r.get("real_monthly") is not None) else "—",
                "Πραγμ. Διαγραφή (€)": eur_num(r.get("real_writeoff")) if (r and r.get("real_writeoff") is not None) else "—",
                "Πραγμ. Υπόλ. (€)": eur_num(r.get("real_residual")) if (r and r.get("real_residual") is not None) else "—",
                "Διαφορά Δόσης (€)": eur_num((r.get("real_monthly") - d.get("predicted_monthly")) if r and r.get("real_monthly") is not None else 0.0),
                "Διαφορά Διαγραφής (€)": eur_num((r.get("real_writeoff") - d.get("predicted_writeoff")) if r and r.get("real_writeoff") is not None else 0.0),
                "Διαφορά Υπολοίπου (€)": eur_num((r.get("real_residual") - d.get("predicted_residual")) if r and r.get("real_residual") is not None else 0.0),
            })
        st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True)
