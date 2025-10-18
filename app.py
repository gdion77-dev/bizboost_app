# ─────────────────────── ΠΡΟΒΛΕΨΕΙΣ & ΠΡΑΓΜΑΤΙΚΕΣ ΡΥΘΜΙΣΕΙΣ ───────────────────────
else:
    st.title("📁 Προβλέψεις & Πραγματικές Ρυθμίσεις")

    # ---------- helpers ----------
    def safe_json_loads(v):
        if isinstance(v, (list, dict)):
            return v
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            try:
                return json.loads(v)
            except Exception:
                # try to fix common double-encoding artifacts
                try:
                    return json.loads(v.encode("utf8").decode("unicode_escape"))
                except Exception:
                    return []
        return []

    def delete_case_db(case_id: str):
        try:
            engine = get_db_engine()
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM cases WHERE case_id = :cid"), {"cid": case_id})
        except Exception as e:
            st.error(f"Σφάλμα διαγραφής: {e}")

    # fresh read every time (so we see latest)
    df_all = load_data()
    if df_all.empty:
        st.info("Δεν υπάρχουν ακόμα υποθέσεις.")
        st.stop()

    # list view
    dfv = df_all[["case_id", "borrower", "predicted_at"]].sort_values("predicted_at", ascending=False)
    st.dataframe(dfv, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Άνοιγμα υπόθεσης")

    case_ids = df_all["case_id"].tolist()
    pick = st.selectbox("Διάλεξε Υπόθεση", case_ids, key="select_case_id")

    # actions row
    a1, a2, a3 = st.columns([1,1,6])
    with a1:
        if st.button("📂 Άνοιγμα", use_container_width=True, key="btn_open_case"):
            st.session_state["open_case_id"] = pick
            st.rerun()
    with a2:
        if st.button("🗑️ Διαγραφή", use_container_width=True, key="btn_delete_case"):
            delete_case_db(pick)
            st.session_state["open_case_id"] = None
            st.success(f"Διαγράφηκε η υπόθεση {pick}")
            st.rerun()

    open_id = st.session_state.get("open_case_id")
    if not open_id:
        st.info("Πάτησε **Άνοιγμα** για να επεξεργαστείς την υπόθεση.")
        st.stop()

    # load the selected case fresh
    row = df_all[df_all["case_id"] == open_id]
    if row.empty:
        st.warning("Δεν βρέθηκε η υπόθεση. Ίσως διαγράφηκε.")
        st.stop()
    row = row.iloc[0].to_dict()

    # parse JSON blobs defensively
    debts = safe_json_loads(row.get("debts_json"))
    real_debts_existing = safe_json_loads(row.get("real_debts_json"))

    st.markdown(f"**Υπόθεση:** `{row.get('case_id','')}`  •  **Οφειλέτης:** {row.get('borrower','')}  •  **Ημερομηνία πρόβλεψης:** {row.get('predicted_at','')}")

    if not debts:
        st.error("Δεν υπάρχουν οφειλές καταχωρημένες στην πρόβλεψη.")
        st.stop()

    st.markdown("### Πραγματική ρύθμιση ανά οφειλή")

    # build a per-debt form
    real_list = []
    for i, d in enumerate(debts):
        cred = d.get("creditor","")
        ltype = d.get("loan_type","")
        bal   = float(d.get("balance",0) or 0.0)

        # try to seed with previous saved real values if any
        seed = {}
        for old in real_debts_existing:
            if (old.get("creditor")==cred) and (old.get("loan_type")==ltype) and abs(float(old.get("balance",0))-bal) < 0.01:
                seed = old
                break

        with st.expander(f"Οφειλή #{i+1} – {cred} / {ltype} / Υπόλοιπο: {bal:,.2f} €", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            real_term    = c1.number_input("Πραγμ. μήνες", 0, 1200, int(seed.get("real_term_months") or 0), key=f"rt_{open_id}_{i}")
            real_monthly = c2.number_input("Πραγμ. δόση (€)", 0.0, 1e9, float(seed.get("real_monthly") or 0.0), step=10.0, key=f"rm_{open_id}_{i}")
            real_write   = c3.number_input("Διαγραφή (€)", 0.0, bal, float(seed.get("real_writeoff") or 0.0), step=100.0, key=f"rw_{open_id}_{i}")

            real_resid   = max(0.0, bal - float(real_write or 0.0))
            c4.metric("Υπόλοιπο ρύθμισης (€)", f"{real_resid:,.2f}")

            haircut_pct = 0.0 if bal <= 0 else 100.0 * (float(real_write or 0.0) / (bal if bal else 1.0))
            st.caption(f"Ποσοστό κουρέματος: **{haircut_pct:.1f}%**")

            real_list.append({
                "creditor": cred,
                "loan_type": ltype,
                "balance": bal,
                "real_term_months": int(real_term) if real_term else None,
                "real_monthly": float(real_monthly) if real_monthly else None,
                "real_writeoff": float(real_write) if real_write else None,
                "real_residual": float(real_resid),
                "real_haircut_pct": float(haircut_pct)
            })

    bL, bR = st.columns([1,1])
    with bL:
        if st.button("💾 Αποθήκευση πραγματικής ρύθμισης", type="primary", use_container_width=True, key="btn_save_real"):
            # rollups
            try:
                monthly_vals = [x.get("real_monthly") for x in real_list if x.get("real_monthly") is not None]
                real_monthly_avg = float(np.mean(monthly_vals)) if monthly_vals else None
                total_bal   = sum([x.get("balance",0.0) for x in real_list])
                total_write = sum([x.get("real_writeoff",0.0) or 0.0 for x in real_list])
                real_haircut_pct = (100.0*total_write/total_bal) if total_bal>0 else None
            except Exception:
                real_monthly_avg = None
                real_haircut_pct = None

            # update DB row
            row_update = row.copy()
            row_update["real_debts_json"] = json.dumps(real_list, ensure_ascii=False)
            row_update["real_monthly"] = real_monthly_avg
            row_update["real_haircut_pct"] = real_haircut_pct

            # keep all other columns intact and write back
            df_to_save = pd.DataFrame([row_update])
            save_data(df_to_save)

            st.success("✅ Αποθηκεύτηκε η πραγματική ρύθμιση για την υπόθεση.")
            st.rerun()

    with bR:
        if st.button("⬅️ Κλείσιμο υπόθεσης", use_container_width=True, key="btn_close_case"):
            st.session_state["open_case_id"] = None
            st.rerun()
