# app.py
# Bizboost - Εξωδικαστικός: Πρόβλεψη & Καταγραφή Ρυθμίσεων (Streamlit + Postgres + PDF)
# - Ελληνικό UI
# - Supabase Postgres μέσω SQLAlchemy + psycopg v3
# - PDF (WeasyPrint HTML/CSS) με embedded template/CSS & σύγχρονο layout
# - Συνοφειλέτες: annual_income (ετήσιο) -> monthly, αφαίρεση ΕΔΔ ανά συνοφειλέτη
# - Κανόνες εξωδικαστικού: ΑΑΔΕ/ΕΦΚΑ 240μήνες, Τράπεζες/Servicers 420μήνες, κόφτης ηλικίας
# - Κατανομή διαθέσιμου: Δημόσιο -> Εξασφαλισμένα -> Λοιπά (priority)
# - Αποθήκευση πρόβλεψης + καταχώριση πραγματικών ρυθμίσεων ανά οφειλή

import os, io, json, uuid, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from sqlalchemy import create_engine, text

# PDF via HTML/CSS
from jinja2 import Environment, DictLoader, select_autoescape
from weasyprint import HTML, CSS

# ────────────────────────────── UI / PATHS ──────────────────────────────
st.set_page_config(page_title="Bizboost - Εξωδικαστικός", page_icon="💠", layout="wide")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")  # δεν χρησιμοποιείται στο PDF πλέον, το κρατάμε για το sidebar
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

# ────────────────────────────── PDF (HTML/CSS) ──────────────────────────────
# Footer στοιχεία (εμφανίζονται sticky στο κάτω μέρος κάθε σελίδας)
CONTACT_NAME   = "Γεώργιος Φ. Διονυσίου Οικονομολόγος BA, MSc"
CONTACT_PHONE  = "+30 2273081618"
CONTACT_EMAIL  = "info@bizboost.gr"
CONTACT_SITE   = "www.bizboost.gr"
CONTACT_ADDRESS= "Αγίου Νικολάου 1, Σάμος 83100"

# Ενσωματωμένο HTML Template (Jinja2)
PREDICTION_HTML = r"""<!doctype html>
<html lang="el">
<head>
  <meta charset="utf-8" />
  <title>Bizboost – Πρόβλεψη Ρύθμισης</title>
  <style>{{ inline_css }}</style>
</head>
<body>

  <!-- HEADER: label αντί για logo -->
  <header class="header">
    <div class="brand">
      <div class="brand-top">The Bizboost</div>
      <div class="brand-sub">by G. Dionysiou</div>
    </div>
    <h1>Πρόβλεψη Ρύθμισης</h1>
  </header>

  <!-- META: tidy key/value card -->
  <section class="meta">
    <div class="kv">
      <div class="k">Υπόθεση</div>                   <div class="v">{{ case_id }}</div>
      <div class="k">Οφειλέτης</div>                  <div class="v">{{ borrower }}</div>
      <div class="k">Ηλικία</div>                     <div class="v">{{ debtor_age }}</div>
      <div class="k">Μέλη νοικοκυριού</div>           <div class="v">{{ adults }}/{{ children }} (ενήλ./ανήλ.)</div>
      <div class="k">Συνολικό μηνιαίο εισόδημα</div>  <div class="v">{{ monthly_income | format_eur }}</div>
      <div class="k">ΕΔΔ νοικοκυριού</div>            <div class="v">{{ edd_household | format_eur }}</div>
      <div class="k">Επιπλέον δαπάνες</div>           <div class="v">{{ extras_sum | format_eur }}</div>
      <div class="k">Καθαρό διαθέσιμο</div>           <div class="v">{{ avail | format_eur }}</div>
      <div class="k">Ακίνητη περιουσία</div>          <div class="v">{{ property_value | format_eur }}</div>
      <div class="k">Ημερομηνία</div>                 <div class="v">{{ predicted_at }}</div>
    </div>
  </section>

  <!-- DEBTS: two-column cards -->
  {% if debts and debts|length > 0 %}
  <section class="debts">
    <h2>Αναλυτικά ανά οφειλή (πρόβλεψη)</h2>

    <div class="debt-grid">
      {% for d in debts %}
      <article class="debt-card">
        <div class="row">
          <div class="k">Πιστωτής</div>            <div class="v">{{ d.creditor }}</div>
          <div class="k">Είδος</div>               <div class="v">{{ d.loan_type }}</div>
          <div class="k">Υπόλοιπο</div>            <div class="v">{{ d.balance | format_eur }}</div>
          <div class="k">Εξασφαλισμένο</div>       <div class="v">{{ "Ναι" if d.secured else "Όχι" }}</div>
          <div class="k">Εξασφάλιση</div>          <div class="v">{{ (d.collateral_value or 0) | format_eur }}</div>
          <div class="k">Οροφή μηνών</div>         <div class="v">{{ d.term_cap }}</div>
          <div class="k">Πρόταση δόσης</div>       <div class="v">{{ d.predicted_monthly | format_eur }}</div>
          <div class="k">Διαγραφή</div>            <div class="v">{{ d.predicted_writeoff | format_eur }}</div>
          <div class="k">Υπόλοιπο ρύθμισης</div>   <div class="v">{{ d.predicted_residual | format_eur }}</div>
          <div class="k">Κούρεμα</div>             <div class="v">{{ "%.1f%%"|format(d.predicted_haircut_pct or 0) }}</div>
        </div>
      </article>
      {% endfor %}
    </div>
  </section>
  {% endif %}

  <!-- REASONING -->
  <section class="reasoning">
    <h2>Σκεπτικό πρότασης</h2>
    <p>{{ personalized_reasoning }}</p>
  </section>

</body>
</html>
"""

# Ενσωματωμένο CSS (WeasyPrint υποστηρίζει @page για footer)
PREDICTION_CSS = r"""
@font-face {
  font-family: "Noto Sans";
  src: url("assets/fonts/NotoSans-Regular.ttf") format("truetype");
  font-weight: normal;
  font-style: normal;
}

:root{
  --ink:#1d2428;
  --muted:#5f6b75;
  --brand:#0F4C81;
  --card:#f7f9fb;
  --line:#e6eaee;
}

@page {
  size: A4;
  margin: 20mm 20mm 24mm 20mm; /* extra bottom for sticky footer */
  @bottom-center {
    content: "Γεώργιος Φ. Διονυσίου Οικονομολόγος BA, MSc • Τ: +30 2273081618 • E: info@bizboost.gr • www.bizboost.gr\AΑγίου Νικολάου 1, Σάμος 83100";
    white-space: pre;
    color: #666;
    font-size: 8pt;
    border-top: 0.6pt solid #ddd;
    padding-top: 6pt;
  }
}
*{ box-sizing:border-box; }

body{
  font-family: "Noto Sans", system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  color: var(--ink);
  font-size: 10.5pt;
  line-height: 1.35;
}

/* Header label */
.header{
  text-align:center;
  margin-bottom: 8mm;
}
.brand{
  display:inline-block;
  border:1pt solid var(--line);
  border-radius:10pt;
  padding:6pt 12pt;
}
.brand-top{
  font-size: 14pt;
  font-weight: 700;
  color: var(--brand);
  letter-spacing: .3pt;
}
.brand-sub{
  font-size: 9pt;
  color: #3a8be0;
  margin-top: -2pt;
}
h1{
  font-size: 16pt;
  margin: 8pt 0 0 0;
  color: var(--brand);
}
h2{
  font-size: 12.5pt;
  margin: 12pt 0 6pt 0;
  color: #2b3238;
}

/* Meta key/value summary */
.meta .kv{
  display:grid;
  grid-template-columns: 48% 52%;
  gap: 3pt 8pt;
  border:1pt solid var(--line);
  border-radius:8pt;
  padding:8pt 10pt;
  background:#fbfcfe;
}
.meta .k{ color: var(--muted); }
.meta .v{ text-align:right; }

/* Debts grid (two columns of cards) */
.debts .debt-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 8pt;
}
.debt-card{
  background: var(--card);
  border: 1pt solid var(--line);
  border-radius: 10pt;
  padding: 8pt 10pt;
  break-inside: avoid;
}
.debt-card .row{
  display:grid;
  grid-template-columns: 52% 48%;
  gap: 2pt 6pt;
}
.debt-card .k{ color:var(--muted); }
.debt-card .v{ text-align:right; font-weight:600; }

/* Reasoning */
.reasoning p{
  margin: 6pt 0 0 0;
}
"""

# Jinja2 περιβάλλον με embedded templates
env = Environment(
    loader=DictLoader({"prediction.html": PREDICTION_HTML}),
    autoescape=select_autoescape(["html"])
)

def _format_eur(x):
    try:
        return f"{float(x):,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"{x} €"

env.filters["format_eur"] = _format_eur

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
    line1 = (f"Η πρόταση διαμορφώθηκε με βάση το καθαρό διαθέσιμο εισόδημα **{_format_eur(avail)}** "
             f"(μηνιαίο εισόδημα **{_format_eur(mi)}** − ΕΔΔ **{_format_eur(edd)}** − πρόσθετες δαπάνες **{_format_eur(extra)}**).")
    parts = []
    if public_cnt:
        cap_info = f"με όριο **{max(public_terms) if public_terms else 240} μήνες**" if public_terms else "έως **240 μήνες**"
        parts.append(f"Για τις απαιτήσεις Δημοσίου (ΑΑΔΕ/ΕΦΚΑ, {public_cnt} οφειλή/ές) χρησιμοποιήθηκε μέγιστη διάρκεια {cap_info}.")
    if secured_cnt:
        parts.append("Για τις εξασφαλισμένες οφειλές ελήφθη υπόψη η εξασφάλιση (security floor).")
    if other_cnt:
        cap_bank = f"{max(bank_terms)} μήνες" if bank_terms else "έως **420 μήνες**"
        parts.append(f"Για τις λοιπές τραπεζικές/servicers οφειλές εφαρμόστηκε μέγιστη διάρκεια {cap_bank}.")
    dist = "Η κατανομή του διαθέσιμου έγινε με προτεραιότητα: **Δημόσιο → Εξασφαλισμένα → Λοιπά**."
    end = "Το υπόλοιπο προς ρύθμιση ισούται με Υπόλοιπο − Διαγραφή, ενώ το ποσοστό κουρέματος με Διαγραφή / Υπόλοιπο."
    return " ".join([line1, *parts, dist, end])

def make_pdf(case_dict: dict) -> bytes:
    """
    Render the prediction PDF using HTML/CSS (WeasyPrint) με:
    - Branded label header (χωρίς εικόνα)
    - Δύο στήλες ανά οφειλή (card layout)
    - Sticky footer στο κάτω μέρος κάθε σελίδας
    """
    context = dict(case_dict)
    context["personalized_reasoning"] = _personalized_reasoning(case_dict)
    # περνάμε το CSS inline στο <style> του template
    context["inline_css"] = PREDICTION_CSS

    # Render HTML
    tpl = env.get_template("prediction.html")
    html_str = tpl.render(**context)

    # WeasyPrint: base_url=BASE_DIR για να «δει» τη γραμματοσειρά assets/fonts/NotoSans-Regular.ttf
    pdf_bytes = HTML(string=html_str, base_url=BASE_DIR).write_pdf()
    return pdf_bytes

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
        # Υπολογισμός μηνιαίου εισοδήματος & ΕΔΔ συνοφειλετών
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
