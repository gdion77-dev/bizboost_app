# app.py
# Bizboost - Εξωδικαστικός: Πρόβλεψη & Καταγραφή Ρυθμίσεων (Streamlit + Postgres + PDF)
# - Ελληνικό UI
# - Supabase Postgres μέσω SQLAlchemy + psycopg v3
# - PDF (ReportLab): σωστά Ελληνικά, κεντραρισμένο “λογότυπο/label”, πίνακες που δεν βγαίνουν εκτός
# - Viewer mode: Κάθε υπόθεση ανοίγει σε νέο παράθυρο (query params) με πρόβλεψη + πραγματική + σύγκριση
# - Χωρίς ML (ακόμα), θα το βάλουμε αφού κλειδώσουμε αυτό

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ────────────────────────────── UI / PATHS ──────────────────────────────
st.set_page_config(page_title="Bizboost - Εξωδικαστικός", page_icon="💠", layout="wide")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")  # αν δεν παίζει καλά, το παρακάτω label φαίνεται όμορφο
DATA_CSV  = os.path.join(BASE_DIR, "cases.csv")

# ───────── Γραμματοσειρές για σωστά Ελληνικά στο PDF (NotoSans → NotoSerif → fallback) ─────────
FONT_DIR = os.path.join(BASE_DIR, "assets", "fonts")
FONT_CANDIDATES = [
    ("NotoSans",  os.path.join(FONT_DIR, "NotoSans-Regular.ttf")),
    ("NotoSerif", os.path.join(FONT_DIR, "NotoSerif-Regular.ttf")),
]
PDF_FONT = "Helvetica"  # ασφαλές fallback

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
    base_adult = 537
    add_adult  = 269
    per_child  = 211
    if adults <= 0:
        adults = 1
    total = base_adult + max(adults-1,0)*add_adult + children*per_child
    return float(total)

def months_cap_from_age(age:int)->int:
    try: a = int(age)
    except: return 120
    if a <= 35:  return 240
    if a <= 50:  return 180
    if a <= 65:  return 120
    return 60

def available_income(total_income:float, edd_household:float, extra_medical:float, extra_students:float, extra_legal:float)->float:
    extras = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
    return max(0.0, float(total_income or 0) - float(edd_household or 0) - extras)

# ─────────────────────────────── RULE ENGINE ───────────────────────────────
def classify_debt(creditor:str, secured:bool)->str:
    if creditor in PUBLIC_CREDITORS: return "PUBLIC"
    if creditor in BANK_SERVICERS:
        return "BANK" if secured else "UNSECURED"
    return "DEFAULT"

def term_cap_for(creditor:str, age_cap:int, secured:bool)->int:
    cat = classify_debt(creditor, secured)
    base = POLICY["term_caps"].get(cat, POLICY["term_caps"]["DEFAULT"])
    return max(1, min(base, age_cap))

def security_floor(balance:float, secured:bool, collateral_value:float)->float:
    if not secured:
        return 0.0
    return max(0.0, float(balance or 0.0) - float(collateral_value or 0.0))

def split_available_proportional(avail:float, debts:list)->dict:
    total = sum(d["balance"] for d in debts if d["balance"]>0)
    if total <= 0: return {i:0.0 for i in range(len(debts))}
    return {i: avail * (d["balance"]/total) for i,d in enumerate(debts)}

def split_available_priority(avail:float, debts:list)->dict:
    out = {i:0.0 for i in range(len(debts))}
    groups = {"PUBLIC":[], "SECURED":[], "UNSECURED":[]}
    for i,d in enumerate(debts):
        if d.get("cat") == "PUBLIC" or d["creditor"] in PUBLIC_CREDITORS:
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
            out[i] += remaining * (debts[i]["balance"]/subtotal)
        remaining = 0.0
        if remaining <= 0:
            break
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
        st.error("Δεν έχει οριστεί DATABASE_URL στα Secrets.")
        st.stop()
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
        return pd.read_sql("SELECT * FROM cases", con=engine)
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

# ────────────────────────────── PDF EXPORT (layout) ──────────────────────────────
CONTACT_NAME   = "Γεώργιος Φ. Διονυσίου Οικονομολόγος BA, MSc"
CONTACT_PHONE  = "+30 2273081618"
CONTACT_EMAIL  = "info@bizboost.gr"
CONTACT_SITE   = "www.bizboost.gr"
CONTACT_ADDRESS= "Αγίου Νικολάου 1, Σάμος 83100"

def _available_width(doc):
    return doc.pagesize[0] - doc.leftMargin - doc.rightMargin

def _cm_list_to_points(widths_cm, doc):
    pts = [w*cm for w in widths_cm]
    total = sum(pts)
    avail = _available_width(doc)
    if total > avail and total > 0:
        scale = avail / total
        pts = [p*scale for p in pts]
    return pts

def _personalized_reasoning(case_dict):
    mi   = float(case_dict.get("monthly_income",0) or 0)
    edd  = float(case_dict.get("edd_household",0) or 0)
    extra= float(case_dict.get("extras_sum",0) or 0)
    avail= float(case_dict.get("avail",0) or 0)
    debts= case_dict.get("debts",[]) or []
    public_cnt  = sum(1 for d in debts if str(d.get("creditor","")) in PUBLIC_CREDITORS)
    secured_cnt = sum(1 for d in debts if bool(d.get("secured")))
    other_cnt   = max(0, len(debts) - public_cnt - secured_cnt)
    public_terms  = sorted({int(d.get("term_cap",0) or 0) for d in debts if str(d.get("creditor","")) in PUBLIC_CREDITORS and d.get("term_cap")})
    bank_terms    = sorted({int(d.get("term_cap",0) or 0) for d in debts if str(d.get("creditor","")) in BANK_SERVICERS and d.get("term_cap")})
    line1 = (
        f"Η πρόταση βασίζεται στο καθαρό διαθέσιμο **{avail:,.2f} €** "
        f"(εισόδημα **{mi:,.2f}** − ΕΔΔ **{edd:,.2f}** − πρόσθετα **{extra:,.2f}**)."
    )
    parts = []
    if public_cnt:
        cap_info = f"{max(public_terms) if public_terms else 240} μήνες"
        parts.append(f"Για Δημόσιο ({public_cnt}) χρησιμοποιήθηκε όριο έως **{cap_info}**.")
    if secured_cnt:
        parts.append("Για εξασφαλισμένες οφειλές εφαρμόστηκε κατώφλι εξασφάλισης (security floor).")
    if other_cnt:
        cap_bank = f"{max(bank_terms)} μήνες" if bank_terms else "420 μήνες"
        parts.append(f"Για λοιπές τραπεζικές/servicers ({other_cnt}) όριο έως **{cap_bank}**.")
    dist = "Κατανομή διαθέσιμου με προτεραιότητα: Δημόσιο → Εξασφαλισμένα → Λοιπά."
    return " ".join([line1, *parts, dist])

class HR(Flowable):
    def __init__(self, width=1, color=colors.HexColor("#DDD")):
        Flowable.__init__(self)
        self.width = width
        self.color = color
        self.height = 5
    def draw(self):
        c = self.canv
        w = c._pagesize[0] - c.leftMargin - c.rightMargin
        x0 = c.leftMargin
        y  = 2
        c.setStrokeColor(self.color)
        c.setLineWidth(self.width)
        c.line(x0, y, x0 + w, y)

def make_pdf(case_dict:dict)->bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.2*cm
    )
    styles = getSampleStyleSheet()
    base_font = PDF_FONT
    styles.add(ParagraphStyle(name="H1", fontName=base_font, fontSize=16, leading=20, spaceAfter=10, textColor=colors.HexColor("#0F4C81"), alignment=1))
    styles.add(ParagraphStyle(name="H2", fontName=base_font, fontSize=12, leading=16, spaceAfter=6, textColor=colors.HexColor("#333333")))
    styles.add(ParagraphStyle(name="P",  fontName=base_font, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="SmallCenter", fontName=base_font, fontSize=8, leading=11, alignment=1, textColor=colors.HexColor("#666")))

    story = []

    # Αν το logo δεν δείχνει καλά, βάζουμε "label"
    if os.path.exists(LOGO_PATH):
        try:
            img = Image(LOGO_PATH, width=150)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 6))
        except Exception:
            pass
    else:
        label = Paragraph("<b>The Bizboost</b> by G. Dionysiou", ParagraphStyle(name="LBL", fontName=base_font, fontSize=14, alignment=1, textColor=colors.HexColor("#0F4C81")))
        story.append(label)
        story.append(Spacer(1, 6))

    story.append(Paragraph("Bizboost – Πρόβλεψη Ρύθμισης", styles["H1"]))

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
    story.append(t)
    story.append(Spacer(1, 10))

    debts = case_dict.get("debts", [])
    if debts:
        story.append(Paragraph("Αναλυτικά ανά οφειλή (πρόβλεψη):", styles["H2"]))
        rows = [["Πιστωτής","Είδος","Υπόλοιπο (€)","Εξασφαλ.","Εξασφάλιση (€)","Οροφή μηνών","Δόση (€)","Διαγραφή (€)","Υπόλοιπο (€)","Κούρεμα (%)"]]
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
        tt = Table(rows, colWidths=_cm_list_to_points([2.6, 2.1, 2.2, 1.0, 1.8, 1.6, 2.1, 2.1, 2.2, 1.1], doc), repeatRows=1)
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
        story.append(tt)
        story.append(Spacer(1, 10))

    story.append(Paragraph("Σκεπτικό πρότασης", styles["H2"]))
    story.append(Paragraph(_personalized_reasoning(case_dict), styles["P"]))
    story.append(Spacer(1, 12))

    story.append(HR())
    story.append(Spacer(1, 4))
    footer_line1 = f"{CONTACT_NAME} • Τ: {CONTACT_PHONE} • E: {CONTACT_EMAIL} • {CONTACT_SITE}"
    footer_line2 = f"{CONTACT_ADDRESS}"
    story.append(Paragraph(footer_line1, styles["SmallCenter"]))
    story.append(Paragraph(footer_line2, styles["SmallCenter"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ────────────────────────────── ΣΥΓΚΡΙΣΗ (κείμενο) ──────────────────────────────
def comparison_text(forecast_debts:list, real_debts:list)->str:
    """Παράγει σύντομο κείμενο σύγκρισης με αποκλίσεις & ποσοστά."""
    if not forecast_debts:
        return "Δεν υπάρχουν δεδομένα πρόβλεψης."
    real_map = {}
    for r in (real_debts or []):
        key = (str(r.get("creditor","")).strip(), str(r.get("loan_type","")).strip(), bool(r.get("balance",0)>0))
        real_map[key] = r

    lines = []
    total_pred_monthly = 0.0
    total_real_monthly = 0.0
    for d in forecast_debts:
        key = (str(d.get("creditor","")).strip(), str(d.get("loan_type","")).strip(), bool(float(d.get("balance",0))>0))
        r = real_map.get(key)
        pm = float(d.get("predicted_monthly",0) or 0.0)
        total_pred_monthly += pm
        if r:
            rm = float(r.get("real_monthly",0) or 0.0)
            total_real_monthly += rm
            diff = rm - pm
            pct  = (diff/pm*100) if pm>0 else 0.0
            lines.append(f"- {d.get('creditor','')}/{d.get('loan_type','')}: δόση πρόβλ. {pm:,.2f}€, πραγματ. {rm:,.2f}€ (Διαφορά {diff:+.2f}€, {pct:+.1f}%).")
        else:
            lines.append(f"- {d.get('creditor','')}/{d.get('loan_type','')}: δεν υπάρχει ακόμη πραγματική καταχώριση.")
    total_diff = total_real_monthly - total_pred_monthly
    total_pct  = (total_diff/total_pred_monthly*100) if total_pred_monthly>0 else 0.0
    lines.append(f"Σύνολο δόσεων: πρόβλεψη {total_pred_monthly:,.2f}€, πραγματική {total_real_monthly:,.2f}€ (Διαφορά {total_diff:+.2f}€, {total_pct:+.1f}%).")
    return "\n".join(lines)

# ────────────────────────────── ΒΟΗΘ. UTILS ──────────────────────────────
def get_query_params():
    # streamlit 1.29+: st.query_params, αλλιώς experimental
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def set_query_params(**kwargs):
    try:
        st.query_params.clear()
        for k,v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

# ────────────────────────────── UI ──────────────────────────────
st.sidebar.image(LOGO_PATH, width=170, caption="Bizboost") if os.path.exists(LOGO_PATH) else st.sidebar.markdown("**Bizboost**")
params = get_query_params()

# Viewer Mode: ./?view=case&case_id=...
if params.get("view", [""])[0] == "case" and params.get("case_id"):
    case_id = params["case_id"][0] if isinstance(params["case_id"], list) else params["case_id"]
    df_all = load_data()
    row = df_all[df_all["case_id"]==case_id]
    if row.empty:
        st.error("Η υπόθεση δεν βρέθηκε.")
        st.stop()
    row = row.iloc[0].to_dict()

    st.title(f"📄 Υπόθεση {row.get('case_id','')}")
    st.caption(f"Οφειλέτης: {row.get('borrower','')} • Ημερομηνία πρόβλεψης: {row.get('predicted_at','')}")

    try:
        debts = json.loads(row.get("debts_json") or "[]")
    except Exception:
        debts = []

    if not debts:
        st.warning("Δεν υπάρχουν οφειλές στην πρόβλεψη.")
        st.stop()

    # ΠΡΟΒΛΕΨΗ: γρήγορη απεικόνιση πίνακα
    st.subheader("Πρόβλεψη ανά οφειλή")
    df_fore = pd.DataFrame([{
        "Πιστωτής": d.get("creditor",""),
        "Είδος": d.get("loan_type",""),
        "Υπόλοιπο (€)": float(d.get("balance",0) or 0.0),
        "Εξασφαλ.": "Ναι" if d.get("secured") else "Όχι",
        "Οροφή μηνών": d.get("term_cap",0),
        "Δόση (€)": float(d.get("predicted_monthly",0) or 0.0),
        "Διαγραφή (€)": float(d.get("predicted_writeoff",0) or 0.0),
        "Υπόλοιπο (€)": float(d.get("predicted_residual",0) or 0.0),
        "Κούρεμα (%)": float(d.get("predicted_haircut_pct",0) or 0.0),
    } for d in debts])
    st.dataframe(df_fore, use_container_width=True)

    # ΚΑΤΑΧΩΡΙΣΗ ΠΡΑΓΜΑΤΙΚΗΣ ΡΥΘΜΙΣΗΣ
    st.markdown("### Πραγματική ρύθμιση (καταχώριση)")
    real_list = []
    for i, d in enumerate(debts):
        with st.expander(f"Οφειλή #{i+1} – {d.get('creditor','')} / {d.get('loan_type','')} / Υπόλοιπο: {float(d.get('balance',0)):,.2f} €", expanded=False):
            col1,col2,col3,col4 = st.columns(4)
            real_term    = col1.number_input("Πραγμ. μήνες", 0, 1200, int(d.get("term_cap",0) or 0), key=f"rtv_{i}")
            real_monthly = col2.number_input("Πραγμ. δόση (€)", 0.0, 1e9, float(d.get("predicted_monthly",0) or 0.0), step=10.0, key=f"rmv_{i}")
            real_write   = col3.number_input("Διαγραφή (€)", 0.0, float(d.get("balance",0) or 0.0), float(d.get("predicted_writeoff",0) or 0.0), step=100.0, key=f"rwv_{i}")
            real_resid   = max(0.0, float(d.get("balance",0) or 0.0) - float(real_write or 0.0))
            col4.metric("Υπόλοιπο ρύθμισης (€)", f"{real_resid:,.2f}")
            haircut_pct = 0.0 if (float(d.get("balance",0) or 0.0) <= 0) else 100.0 * (float(real_write or 0.0) / max(float(d.get("balance") or 1.0),1.0))
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

    # ΠΑΝΤΑ ΕΜΦΑΝΙΖΕΙ ΣΥΓΚΡΙΣΗ ΑΝ ΥΠΑΡΧΟΥΝ REAL
    try:
        df_all = load_data()
        real_saved = json.loads(df_all[df_all["case_id"]==case_id].iloc[0].get("real_debts_json") or "[]")
    except Exception:
        real_saved = []

    st.markdown("### Σύγκριση πρόβλεψης με πραγματική ρύθμιση")
    if real_saved:
        txt = comparison_text(debts, real_saved)
        st.markdown(txt.replace("\n","  \n"))
        # και πίνακας σύγκρισης
        real_map = { (r.get("creditor",""), r.get("loan_type","")): r for r in real_saved }
        comp_rows = []
        for d in debts:
            r = real_map.get((d.get("creditor",""), d.get("loan_type","")))
            pm = float(d.get("predicted_monthly",0) or 0.0)
            rm = float((r or {}).get("real_monthly") or 0.0)
            diff = (rm - pm) if r else None
            pct  = (diff/pm*100) if (r and pm>0) else None
            comp_rows.append({
                "Πιστωτής": d.get("creditor",""),
                "Είδος": d.get("loan_type",""),
                "Πρόβλ. Δόση": pm,
                "Πραγμ. Δόση": rm if r else None,
                "Διαφορά (€)": diff,
                "Διαφορά (%)": round(pct,1) if pct is not None else None,
                "Πρόβλ. Διαγραφή": d.get("predicted_writeoff",0.0),
                "Πραγμ. Διαγραφή": (r or {}).get("real_writeoff"),
                "Πρόβλ. Υπόλοιπο": d.get("predicted_residual",0.0),
                "Πραγμ. Υπόλοιπο": (r or {}).get("real_residual"),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    else:
        st.info("Δεν υπάρχουν ακόμα καταχωρημένες πραγματικές ρυθμίσεις για σύγκριση.")

    st.stop()

# ────────────────────────────── ΚΥΡΙΟ ΜΕΝΟΥ ──────────────────────────────
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
        monthly_income_codes = 0.0
        edd_codes = 0.0
        for c in codebtors:
            monthly_income_codes += float(c.get("annual_income") or 0.0)/12.0
            cadults = int(c.get("adults") or 1)
            cchildren = int(c.get("children") or 0)
            edd_codes += compute_edd(cadults, cchildren)

        monthly_income = float(monthly_income_main + monthly_income_codes)
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

        per_debt_rows = []
        debts_to_store = []
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

        st.subheader("Αποτελέσματα ανά οφειλή")
        st.dataframe(pd.DataFrame(per_debt_rows), use_container_width=True)
        st.info("Κατανομή διαθέσιμου: Δημόσιο → Εξασφαλισμένα → Λοιπά.")

        # Αποθήκευση
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
            "debts_json": json.dumps(debts_to_store, ensure_ascii=False),
            "co_debtors_json": json.dumps(codebtors, ensure_ascii=False),
            "real_debts_json": json.dumps([], ensure_ascii=False),
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
        st.success(f"✅ Αποθηκεύτηκε η πρόβλεψη: {case_id}")

        # PDF λήψη
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
else:
    st.title("📁 Προβλέψεις & Πραγματικές Ρυθμίσεις")
    df_all = load_data()
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
        st.stop()

    st.markdown("### Αποθηκευμένες υποθέσεις")
    cols = st.columns(3)
    for i, (_, r) in enumerate(df_all.sort_values("predicted_at", ascending=False).iterrows()):
        with cols[i % 3]:
            st.markdown(f"**{r.get('case_id','')}**  \n{r.get('borrower','')}")
            # ανοίγει σε νέο παράθυρο: ./?view=case&case_id=...
            href = f"./?view=case&case_id={r.get('case_id','')}"
            st.markdown(f"<a href='{href}' target='_blank'>🔎 Άνοιγμα σε νέο παράθυρο</a>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            if c1.button("📄 PDF", key=f"pdf_{r['case_id']}"):
                # γρήγορο PDF από την αποθηκευμένη πρόβλεψη
                try:
                    debts = json.loads(r.get("debts_json") or "[]")
                except Exception:
                    debts = []
                case_for_pdf = {
                    "case_id": r.get("case_id",""),
                    "borrower": r.get("borrower",""),
                    "debtor_age": int(r.get("debtor_age") or 0),
                    "adults": int(r.get("adults") or 1),
                    "children": int(r.get("children") or 0),
                    "monthly_income": float(r.get("monthly_income") or 0.0),
                    "edd_household": float(r.get("edd_manual") or 0.0),
                    "extras_sum": float((r.get("extra_medical") or 0)+(r.get("extra_students") or 0)+(r.get("extra_legal") or 0)),
                    "avail": None,
                    "property_value": float(r.get("property_value") or 0.0),
                    "debts": debts,
                    "predicted_at": r.get("predicted_at","")
                }
                pdf_bytes = make_pdf(case_for_pdf)
                st.download_button(
                    "⬇️ Κατέβασε PDF",
                    data=pdf_bytes,
                    file_name=f"{r.get('case_id','')}_prediction.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_{r['case_id']}"
                )
            if c2.button("🗑️ Διαγραφή", key=f"del_{r['case_id']}"):
                engine = get_db_engine()
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM cases WHERE case_id=:cid"), {"cid": r["case_id"]})
                st.success("Η υπόθεση διαγράφηκε.")
                st.rerun()
