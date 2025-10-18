# app.py
# Bizboost - Εξωδικαστικός: Πρόβλεψη & Καταγραφή Ρυθμίσεων (Streamlit + Postgres + PDF)
# - Ελληνικό UI
# - Supabase Postgres μέσω SQLAlchemy + psycopg v3
# - PDF (WeasyPrint HTML/CSS) με καθαρό, επαγγελματικό layout:
#   • Header label ("The Bizboost by G. Dionysiou") — no logo image required
#   • Περίληψη σε 2 στήλες (dl)
#   • Οφειλές σε 2 στήλες "cards" (dl ανά οφειλή, χωρίς στριμώγματα)
#   • Footer pinned στο κάτω μέρος κάθε σελίδας με στοιχεία επικοινωνίας
# - Συνοφειλέτες: annual_income (ετήσιο) -> monthly, αφαίρεση ΕΔΔ ανά συνοφειλέτη
# - Κανόνες εξωδικαστικού: ΑΑΔΕ/ΕΦΚΑ 240μήνες, Τράπεζες/Servicers 420μήνες, κόφτης ηλικίας
# - Κατανομή διαθέσιμου: Δημόσιο -> Εξασφαλισμένα -> Λοιπά (priority)
# - Αποθήκευση πρόβλεψης + καταχώριση πραγματικών ρυθμίσεων ανά οφειλή

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# PDF (WeasyPrint)
from weasyprint import HTML, CSS

# ────────────────────────────── UI / PATHS ──────────────────────────────
st.set_page_config(page_title="Bizboost - Εξωδικαστικός", page_icon="💠", layout="wide")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")  # (δεν απαιτείται πλέον, κρατάμε για μελλοντική χρήση)
DATA_CSV  = os.path.join(BASE_DIR, "cases.csv")  # προαιρετικό αρχικό import

# ────────────────────────────── ΠΑΡΑΜΕΤΡΟΙ ΠΟΛΙΤΙΚΗΣ ──────────────────────────────
PUBLIC_CREDITORS = {"ΑΑΔΕ", "ΕΦΚΑ"}
BANK_SERVICERS   = {
    "DoValue","Intrum","Cepal","Qquant","APS","EOS","Veralitis",
    "Πειραιώς","Εθνική","Eurobank","Alpha"
}
CREDITORS = list(PUBLIC_CREDITORS) + list(BANK_SERVICERS)
LOAN_TYPES = ["Στεγαστικό","Καταναλωτικό","Επαγγελματικό"]

POLICY = {
    "priority": ["PUBLIC", "SECURED", "UNSECURED"],  # σειρά εξυπηρέτησης
    "term_caps": {"PUBLIC": 240, "BANK": 420, "DEFAULT": 240},
    "allocate": "priority_first",  # "priority_first" ή "proportional"
    "max_haircut": {"PUBLIC": None, "BANK": None, "DEFAULT": None},  # π.χ. 0.4 για 40%
}

# ─────────────────────── ΕΔΔ & ΔΙΑΘΕΣΙΜΑ ───────────────────────
def compute_edd(adults:int, children:int)->float:
    """ΕΔΔ: 1 ενήλικας 537€, κάθε επιπλέον ενήλικας +269€, κάθε ανήλικος +211€."""
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
    """Κατώφλι λόγω εξασφάλισης: δεν διαγράφεις κάτω από (balance - collateral)."""
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

# ────────────────────────────── PDF: HTML + CSS (WeasyPrint) ──────────────────────────────

CONTACT_NAME    = "Γεώργιος Φ. Διονυσίου Οικονομολόγος BA, MSc"
CONTACT_PHONE   = "+30 2273081618"
CONTACT_EMAIL   = "info@bizboost.gr"
CONTACT_SITE    = "www.bizboost.gr"
CONTACT_ADDRESS = "Αγίου Νικολάου 1, Σάμος 83100"

def _fmt_eur(x):
    try:
        return f"{float(x):,.2f} €"
    except:
        return str(x)

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
        f"Η πρόταση διαμορφώθηκε με βάση το καθαρό διαθέσιμο εισόδημα {avail:,.2f} € "
        f"(μηνιαίο εισόδημα {mi:,.2f} € − ΕΔΔ {edd:,.2f} € − πρόσθετες δαπάνες {extra:,.2f} €)."
    )
    parts = []
    if public_cnt:
        cap_info = f"με όριο {max(public_terms) if public_terms else 240} μήνες" if public_terms else "έως 240 μήνες"
        parts.append(f"Για τις απαιτήσεις Δημοσίου (ΑΑΔΕ/ΕΦΚΑ, {public_cnt} οφειλή/ές) χρησιμοποιήθηκε μέγιστη διάρκεια {cap_info}.")
    if secured_cnt:
        parts.append("Για τις εξασφαλισμένες οφειλές ελήφθη υπόψη η εξασφάλιση (security floor).")
    if other_cnt:
        cap_bank = f"{max(bank_terms)} μήνες" if bank_terms else "έως 420 μήνες"
        parts.append(f"Για τις λοιπές τραπεζικές/servicers οφειλές εφαρμόστηκε μέγιστη διάρκεια {cap_bank}.")
    dist = "Η κατανομή του διαθέσιμου έγινε με προτεραιότητα: Δημόσιο → Εξασφαλισμένα → Λοιπά."
    end = "Το υπόλοιπο ρύθμισης ανά οφειλή είναι Υπόλοιπο − Διαγραφή και το ποσοστό κουρέματος είναι Διαγραφή / Υπόλοιπο."
    return " ".join([line1, *parts, dist, end])

def _html_css(case):
    """Return (html, css) for WeasyPrint. Two-column summary + two-column debt cards + sticky footer."""
    debts = case.get("debts", []) or []

    # Summary (dl) two columns
    summary_rows = [
        ("Υπόθεση", case.get("case_id","")),
        ("Οφειλέτης", case.get("borrower","")),
        ("Ηλικία", str(case.get("debtor_age",""))),
        ("Μέλη νοικοκυριού", f"{case.get('adults',0)}/{case.get('children',0)}"),
        ("Συνολικό μηνιαίο εισόδημα", _fmt_eur(case.get("monthly_income",0))),
        ("ΕΔΔ νοικοκυριού", _fmt_eur(case.get("edd_household",0))),
        ("Πρόσθετες δαπάνες", _fmt_eur(case.get("extras_sum",0))),
        ("Καθαρό διαθέσιμο", _fmt_eur(case.get("avail",0))),
        ("Ακίνητη περιουσία", _fmt_eur(case.get("property_value",0))),
        ("Ημερομηνία", case.get("predicted_at","")),
    ]
    summary_html = "".join([
        f"""<div class="pair">
               <div class="k">{k}</div>
               <div class="v">{v}</div>
            </div>"""
        for k, v in summary_rows
    ])

    # Debts as cards (two columns, flow nicely, no squashed columns)
    debt_cards = []
    for d in debts:
        secured_txt = "Ναι" if d.get("secured") else "Όχι"
        coll_val = d.get("collateral_value", 0) if d.get("secured") else 0
        card = f"""
        <div class="debt-card">
          <div class="debt-title">{d.get('creditor','')} — {d.get('loan_type','')}</div>
          <dl>
            <div><dt>Υπόλοιπο</dt><dd>{_fmt_eur(d.get('balance',0))}</dd></div>
            <div><dt>Εξασφαλισμένο</dt><dd>{secured_txt}</dd></div>
            <div><dt>Εξασφάλιση</dt><dd>{_fmt_eur(coll_val)}</dd></div>
            <div><dt>Οροφή μηνών</dt><dd>{d.get('term_cap','')}</dd></div>
            <div><dt>Πρόταση δόσης</dt><dd>{_fmt_eur(d.get('predicted_monthly',0))}</dd></div>
            <div><dt>Διαγραφή</dt><dd>{_fmt_eur(d.get('predicted_writeoff',0))}</dd></div>
            <div><dt>Υπόλοιπο ρύθμισης</dt><dd>{_fmt_eur(d.get('predicted_residual',0))}</dd></div>
            <div><dt>Κούρεμα</dt><dd>{float(d.get('predicted_haircut_pct',0)):.1f}%</dd></div>
          </dl>
        </div>
        """
        debt_cards.append(card)
    debts_html = "".join(debt_cards) if debt_cards else "<p class='muted'>Δεν δηλώθηκαν οφειλές.</p>"

    html = f"""
<!DOCTYPE html>
<html lang="el">
<head>
  <meta charset="utf-8" />
  <title>Bizboost – Πρόβλεψη</title>
</head>
<body>
  <header class="header">
    <div class="brand">
      <div class="brand-top">The Bizboost</div>
      <div class="brand-sub">by G. Dionysiou</div>
    </div>
    <div class="doc-title">Πρόβλεψη Ρύθμισης</div>
  </header>

  <main class="content">
    <section class="card">
      <h1>Στοιχεία Περίληψης</h1>
      <div class="summary-grid">
        {summary_html}
      </div>
    </section>

    <section class="card">
      <h2>Αναλυτικά ανά οφειλή (πρόβλεψη)</h2>
      <div class="debt-grid">
        {debts_html}
      </div>
    </section>

    <section class="card">
      <h2>Σκεπτικό πρότασης</h2>
      <p class="reasoning">{_personalized_reasoning(case)}</p>
    </section>
  </main>

  <footer class="footer" id="footer">
    <div class="footer-line"></div>
    <div class="footer-text">
      {CONTACT_NAME} • Τ: {CONTACT_PHONE} • E: {CONTACT_EMAIL} • {CONTACT_SITE}<br/>
      {CONTACT_ADDRESS}
    </div>
  </footer>
</body>
</html>
"""
    css = """
/* ----------- Page setup ----------- */
@page {
  size: A4;
  margin: 20mm 16mm 26mm 16mm; /* extra bottom for footer */
}
html, body {
  font-family: "Inter", "Noto Sans", system-ui, -apple-system, Arial, sans-serif;
  color: #1a1a1a;
  font-size: 10.5pt;
  line-height: 1.45;
}
* { box-sizing: border-box; }

/* ----------- Header ----------- */
.header {
  border-bottom: 1px solid #e6e9ef;
  padding-bottom: 8pt;
  margin-bottom: 12pt;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.brand {
  display: flex;
  flex-direction: column;
  gap: 2pt;
}
.brand-top {
  font-weight: 700;
  letter-spacing: 0.6pt;
  text-transform: uppercase;
  color: #0F4C81;
  font-size: 12pt;
}
.brand-sub {
  color: #6b7280;
  font-size: 9pt;
}
.doc-title {
  font-size: 14pt;
  font-weight: 700;
  color: #0F4C81;
}

/* ----------- Content wrapper ----------- */
.content {
  display: block;
}

/* ----------- Cards ----------- */
.card {
  border: 1px solid #e6e9ef;
  border-radius: 6pt;
  padding: 10pt 12pt;
  margin: 10pt 0;
  background: #fff;
}
.card h1, .card h2 {
  margin: 0 0 8pt 0;
  color: #111827;
}
.card h1 { font-size: 12.5pt; }
.card h2 { font-size: 11.5pt; }

/* ----------- Summary grid (two columns) ----------- */
.summary-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6pt 18pt;
}
.summary-grid .pair {
  display: grid;
  grid-template-columns: 1.1fr 1.4fr;
  gap: 6pt;
}
.summary-grid .k {
  color: #374151;
  font-weight: 600;
}
.summary-grid .v {
  color: #111827;
  word-break: break-word;
}

/* ----------- Debts: two-column cards ----------- */
.debt-grid {
  columns: 2;            /* newspaper-style columns */
  column-gap: 12pt;
}
.debt-card {
  break-inside: avoid;   /* keep a card intact */
  border: 1px solid #e6e9ef;
  border-radius: 6pt;
  margin: 0 0 10pt 0;   /* space after each card */
  padding: 8pt 10pt;
  background: #fafbfc;
}
.debt-title {
  font-weight: 700;
  color: #0F4C81;
  margin-bottom: 6pt;
}
.debt-card dl {
  display: grid;
  grid-template-columns: 1.2fr 1.3fr;
  gap: 4pt 10pt;
}
.debt-card dt {
  color: #374151;
  font-weight: 600;
}
.debt-card dd {
  margin: 0;
  color: #111827;
  word-break: break-word;
}

/* ----------- Reasoning ----------- */
.reasoning {
  margin: 0;
  color: #1f2937;
}

/* ----------- Footer (pinned bottom) ----------- */
.footer {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  height: 22mm;            /* reserve height */
  padding: 6pt 16mm 0 16mm;
}
.footer-line {
  width: 100%;
  border-top: 1px solid #e6e9ef;
  margin-bottom: 6pt;
}
.footer-text {
  text-align: center;
  font-size: 9pt;
  color: #6b7280;
  line-height: 1.35;
}

/* ----------- Helpers ----------- */
.muted { color: #6b7280; }
"""
    return html, css

def make_pdf(case_dict:dict)->bytes:
    """Render A4 PDF with clean layout using WeasyPrint (HTML+CSS)."""
    # sanity default strings (avoid 'None' in fields)
    for k in ["case_id","borrower","predicted_at"]:
        if not case_dict.get(k):
            case_dict[k] = ""
    html, css = _html_css(case_dict)
    try:
        pdf = HTML(string=html, base_url=BASE_DIR).write_pdf(
            stylesheets=[CSS(string=css)]
        )
        return pdf
    except Exception as e:
        raise RuntimeError(
            "WeasyPrint failed. Ensure Homebrew libs (glib, cairo, pango, gdk-pixbuf, libffi, libxml2, libxslt) "
            "are installed and `pip install weasyprint` ran in your venv. "
            f"Original error: {e}"
        )

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
        # Συνοφειλέτες -> λίστα αντικειμένων
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

        # Συγκεντρωτικά / οφειλές
        debts = debts_df.fillna(0).to_dict(orient="records")
        total_debt  = sum([float(d["balance"] or 0) for d in debts])
        secured_amt = sum([float(d["collateral_value"] or 0) for d in debts if d.get("secured")])

        extras_sum = (extra_medical or 0) + (extra_students or 0) + (extra_legal or 0)
        avail = available_income(monthly_income, edd_total_house, extra_medical, extra_students, extra_legal)
        age_cap_months = months_cap_from_age(int(debtor_age))

        # Εμπλουτισμός οφειλών για κανόνες
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

        # Κατανομή διαθέσιμου
        shares = split_available_priority(avail, enriched) if POLICY["allocate"]=="priority_first" else split_available_proportional(avail, enriched)

        # Υπολογισμός ανά οφειλή
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
        st.info("Κατανομή διαθέσιμου: Δημόσιο → Εξασφαλισμένα → Λοιπά (προτεραιότητα).")

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
else:
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
