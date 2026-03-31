import pandas as pd
import streamlit as st
from pathlib import Path

DEFAULT_FILE = Path("FrankieDS.csv")
EXCLUDE_SP = [
    "Shazia",
    "Nancy",
    "Shahvar",
    "House",
    "BC Accounts",
    "Company Pool",
    "Dakota (Staff Account)",
    "Anas",
    "SMOKE ARSENAL BC",
    "Smoke Arsenal QC",
    "Bc Smoke Arsenal",
]
KEY_PRODUCT_LINES = ["BC10K", "GH20K", "Nuud 50K"]

st.set_page_config(page_title="Smoke Arsenal Incentive Tool", layout="wide")
st.title("Smoke Arsenal — Incentive Allocation")


@st.cache_data(show_spinner=False)
def load_csv_data(source):
    return pd.read_csv(source)


@st.cache_data(show_spinner=False)
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df["Customer"].astype(str).str.contains("smoke arsenal", case=False, na=False)]
    df["invoice_date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
    df = df.rename(columns={
        "Untaxed Total": "revenue",
        "Current Salesperson": "current_sp",
        "Product": "product",
        "Customer Tier": "customer_tier",
        "Account Type": "account_type",
    })
    df["quarter"] = df["invoice_date"].dt.to_period("Q").astype(str)

    # Normalize account type — blank/missing becomes Individual Store
    df["account_type"] = (
        df["account_type"].fillna("").str.strip()
        .apply(lambda v: "Franchise" if v == "Franchise" else "Individual Store")
    )

    # Spend Tier — based on each customer's average quarterly revenue across 2025
    df_2025 = df[df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    cust_2025_avg = (
        df_2025.groupby("Customer")["revenue"].sum() / n_2025_q
    ).rename("_avg_2025_q")

    def _spend_tier(avg: float) -> str:
        if avg >= 10000:
            return "High Tier"
        if avg >= 5000:
            return "Medium Tier"
        return "Low Tier"

    spend_tier_map = cust_2025_avg.apply(_spend_tier).rename("spend_tier")
    df = df.join(spend_tier_map, on="Customer")
    df["spend_tier"] = df["spend_tier"].fillna("Low Tier")

    return df


def fmt(value: float) -> str:
    return f"${value:,.0f}"


def compute_kpis(df: pd.DataFrame, current_q: str, prior_q: str | None) -> dict:
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    current_rev = cq_df["revenue"].sum()
    prior_rev = pq_df["revenue"].sum()
    growth_pct = (current_rev - prior_rev) / prior_rev * 100 if prior_rev > 0 else None

    cq_customers = set(cq_df["Customer"].dropna().unique())
    ever_before = set(df[df["quarter"] < current_q]["Customer"].dropna().unique())
    new_accounts = cq_customers - ever_before
    new_account_count = len(new_accounts)

    df_2025 = df[df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    cust_2025_avg = df_2025.groupby("Customer")["revenue"].sum() / n_2025_q

    cq_per_cust = cq_df.groupby("Customer")["revenue"].sum()
    existing_cq = cq_per_cust[~cq_per_cust.index.isin(new_accounts)]
    grew_count = sum(1 for c, r in existing_cq.items() if r > cust_2025_avg.get(c, 0))
    declined_count = sum(1 for c, r in existing_cq.items() if r < cust_2025_avg.get(c, 0))

    pq_customers = set(pq_df["Customer"].dropna().unique()) if not pq_df.empty else set()
    inactive_count = len(pq_customers - cq_customers)

    kpl = {}
    for line in KEY_PRODUCT_LINES:
        cq_mask = cq_df["product"].fillna("").str.contains(line, case=False)
        cq_rev = cq_df.loc[cq_mask, "revenue"].sum()
        pq_mask = pq_df["product"].fillna("").str.contains(line, case=False)
        pq_rev = pq_df.loc[pq_mask, "revenue"].sum() if not pq_df.empty else 0.0
        growth = (cq_rev - pq_rev) / pq_rev * 100 if pq_rev > 0 else None
        kpl[line] = {"current": cq_rev, "prior": pq_rev, "growth_pct": growth}

    return {
        "current_rev": current_rev,
        "prior_rev": prior_rev,
        "growth_pct": growth_pct,
        "new_account_count": new_account_count,
        "grew_count": grew_count,
        "declined_count": declined_count,
        "inactive_count": inactive_count,
        "kpl": kpl,
    }


def compute_sp_summary(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    cq_df = df[(df["quarter"] == current_q) & (~df["current_sp"].isin(EXCLUDE_SP))]
    pq_df = (
        df[(df["quarter"] == prior_q) & (~df["current_sp"].isin(EXCLUDE_SP))]
        if prior_q else pd.DataFrame(columns=df.columns)
    )

    cq_rev = cq_df.groupby("current_sp")["revenue"].sum().rename("cq_revenue")
    pq_rev = (
        pq_df.groupby("current_sp")["revenue"].sum().rename("pq_revenue")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_revenue")
    )

    summary = cq_rev.to_frame().join(pq_rev, how="outer").fillna(0.0)
    summary["growth_pct"] = summary.apply(
        lambda r: (r["cq_revenue"] - r["pq_revenue"]) / r["pq_revenue"] * 100
        if r["pq_revenue"] > 0 else None,
        axis=1,
    )

    for line in KEY_PRODUCT_LINES:
        mask = cq_df["product"].fillna("").str.contains(line, case=False)
        line_total = cq_df.loc[mask, "revenue"].sum()
        rep_line = cq_df[mask].groupby("current_sp")["revenue"].sum()
        summary[f"{line}_rev"] = rep_line
        summary[f"{line}_rev"] = summary[f"{line}_rev"].fillna(0.0)
        summary[f"{line}_pct"] = (
            summary[f"{line}_rev"] / line_total * 100 if line_total > 0 else 0.0
        )

    # Attribution: revenue where the raw Salesperson column matches this rep
    if "Salesperson" in df.columns:
        attr_rev = (
            df[df["quarter"] == current_q]
            .groupby("Salesperson")["revenue"].sum()
            .rename("attribution_rev")
        )
        summary = summary.join(attr_rev, how="left")
        summary["attribution_rev"] = summary["attribution_rev"].fillna(0.0)
    else:
        summary["attribution_rev"] = 0.0
    summary["attribution_pct"] = summary.apply(
        lambda r: r["attribution_rev"] / r["cq_revenue"] * 100 if r["cq_revenue"] > 0 else 0.0,
        axis=1,
    )

    # New accounts opened this quarter (by current_sp)
    ever_before = set(df[df["quarter"] < current_q]["Customer"].dropna().unique())
    new_custs_cq = cq_df[~cq_df["Customer"].isin(ever_before)]
    new_acct_count = new_custs_cq.groupby("current_sp")["Customer"].nunique().rename("new_accounts")
    summary = summary.join(new_acct_count, how="left")
    summary["new_accounts"] = summary["new_accounts"].fillna(0).astype(int)

    # Low spend growth: delta for Low Tier customers active in CQ (same universe both quarters)
    cq_low_customers = set(
        cq_df[cq_df["spend_tier"] == "Low Tier"]["Customer"].dropna().unique()
    )
    cq_low = (
        cq_df[cq_df["Customer"].isin(cq_low_customers)]
        .groupby("current_sp")["revenue"].sum().rename("cq_low_rev")
    )
    pq_low = (
        pq_df[pq_df["Customer"].isin(cq_low_customers)]
        .groupby("current_sp")["revenue"].sum().rename("pq_low_rev")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_low_rev")
    )
    low_tbl = cq_low.to_frame().join(pq_low, how="left").fillna(0.0)
    low_tbl["low_spend_growth"] = low_tbl["cq_low_rev"] - low_tbl["pq_low_rev"]
    summary = summary.join(low_tbl["low_spend_growth"], how="left")
    summary["low_spend_growth"] = summary["low_spend_growth"].fillna(0.0)

    # Reactivated accounts count (by current_sp): absent last quarter, seen before
    pq_customers = (
        set(df[df["quarter"] == prior_q]["Customer"].dropna().unique())
        if prior_q else set()
    )
    pre_pq_cutoff = prior_q if prior_q else current_q
    pre_pq_customers = set(df[df["quarter"] < pre_pq_cutoff]["Customer"].dropna().unique())
    cq_all_customers = set(cq_df["Customer"].dropna().unique())
    reactivated = (cq_all_customers - pq_customers) & pre_pq_customers
    react_count = (
        cq_df[cq_df["Customer"].isin(reactivated)]
        .groupby("current_sp")["Customer"].nunique()
        .rename("reactivated_count")
    )
    summary = summary.join(react_count, how="left")
    summary["reactivated_count"] = summary["reactivated_count"].fillna(0).astype(int)

    summary = summary.reset_index().rename(columns={"current_sp": "Salesperson"})
    return summary.sort_values("cq_revenue", ascending=False)


def compute_customer_table(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    # Exclude internal/admin accounts AND unassigned rows so the universe matches the SP table
    df = df[df["current_sp"].notna() & ~df["current_sp"].isin(EXCLUDE_SP)]
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    cq_rev = cq_df.groupby("Customer")["revenue"].sum().rename("cq_revenue")
    pq_rev = (
        pq_df.groupby("Customer")["revenue"].sum().rename("pq_revenue")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_revenue")
    )

    df_2025 = df[df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    avg_2025 = (df_2025.groupby("Customer")["revenue"].sum() / n_2025_q).rename("avg_2025_q")

    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier"]
    cq_meta = cq_df.sort_values("invoice_date").groupby("Customer")[meta_cols].last()

    table = (
        cq_rev.to_frame()
        .join(pq_rev, how="left")
        .join(avg_2025, how="left")
        .join(cq_meta, how="left")
        .fillna({"pq_revenue": 0.0, "avg_2025_q": 0.0})
        .reset_index()
    )
    table["growth_pct"] = table.apply(
        lambda r: (r["cq_revenue"] - r["pq_revenue"]) / r["pq_revenue"] * 100
        if r["pq_revenue"] > 0 else None,
        axis=1,
    )
    return table.sort_values("cq_revenue", ascending=False)


def compute_sp_table(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    sp_df = df[df["current_sp"].notna() & ~df["current_sp"].isin(EXCLUDE_SP)]
    cq_df = sp_df[sp_df["quarter"] == current_q]
    pq_df = sp_df[sp_df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    cq_rev = cq_df.groupby("current_sp")["revenue"].sum().rename("cq_revenue")

    # Only count prior-quarter revenue for customers active in the current quarter
    # so totals stay comparable to the customer table (which left-joins on CQ customers)
    cq_customers = set(cq_df["Customer"].dropna().unique())
    pq_rev = (
        pq_df[pq_df["Customer"].isin(cq_customers)]
        .groupby("current_sp")["revenue"].sum().rename("pq_revenue")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_revenue")
    )

    df_2025 = sp_df[sp_df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    avg_2025 = (df_2025.groupby("current_sp")["revenue"].sum() / n_2025_q).rename("avg_2025_q")

    table = (
        cq_rev.to_frame()
        .join(pq_rev, how="left")
        .join(avg_2025, how="left")
        .fillna(0.0)
        .reset_index()
        .rename(columns={"current_sp": "Salesperson"})
    )
    table["growth_pct"] = table.apply(
        lambda r: (r["cq_revenue"] - r["pq_revenue"]) / r["pq_revenue"] * 100
        if r["pq_revenue"] > 0 else None,
        axis=1,
    )
    return table.sort_values("cq_revenue", ascending=False)


def compute_new_accounts_table(df: pd.DataFrame, current_q: str) -> pd.DataFrame:
    """Customers whose very first invoice in the entire dataset falls in current_q."""
    base = df[df["current_sp"].notna() & ~df["current_sp"].isin(EXCLUDE_SP)]
    cq_df = base[base["quarter"] == current_q]
    ever_before = set(base[base["quarter"] < current_q]["Customer"].dropna().unique())
    new_customers = set(cq_df["Customer"].dropna().unique()) - ever_before

    cq_new = cq_df[cq_df["Customer"].isin(new_customers)]
    rev = cq_new.groupby("Customer")["revenue"].sum().rename("cq_revenue")
    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier"]
    meta = cq_new.sort_values("invoice_date").groupby("Customer")[meta_cols].last()
    table = rev.to_frame().join(meta).reset_index()
    return table.sort_values("cq_revenue", ascending=False)


def compute_reactivated_accounts_table(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    """Customers active in current_q, absent in prior_q, but with history before prior_q."""
    base = df[df["current_sp"].notna() & ~df["current_sp"].isin(EXCLUDE_SP)]
    cq_df = base[base["quarter"] == current_q]
    pq_customers = (
        set(base[base["quarter"] == prior_q]["Customer"].dropna().unique())
        if prior_q else set()
    )
    pre_pq_cutoff = prior_q if prior_q else current_q
    pre_pq_customers = set(
        base[base["quarter"] < pre_pq_cutoff]["Customer"].dropna().unique()
    )

    cq_customers = set(cq_df["Customer"].dropna().unique())
    # Active now, absent last quarter, but seen before last quarter
    reactivated = cq_customers - pq_customers
    reactivated = reactivated & pre_pq_customers

    cq_react = cq_df[cq_df["Customer"].isin(reactivated)]
    rev = cq_react.groupby("Customer")["revenue"].sum().rename("cq_revenue")
    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier"]
    meta = cq_react.sort_values("invoice_date").groupby("Customer")[meta_cols].last()

    # Last active quarter before current
    last_active = (
        base[base["Customer"].isin(reactivated) & (base["quarter"] < current_q)]
        .groupby("Customer")["quarter"].max()
        .rename("last_active_q")
    )

    table = rev.to_frame().join(meta).join(last_active).reset_index()
    return table.sort_values("cq_revenue", ascending=False)


WEIGHTS = {"new_accounts": 0.4, "low_spend_growth": 0.3, "reactivated_count": 0.3}


def compute_bonus_allocation(selected_reps: list, sp_metrics: pd.DataFrame, pool: float) -> pd.DataFrame:
    alloc = sp_metrics[sp_metrics["Salesperson"].isin(selected_reps)][
        ["Salesperson", "new_accounts", "low_spend_growth", "reactivated_count"]
    ].copy()

    # Negative low spend growth doesn't contribute
    alloc["low_spend_growth"] = alloc["low_spend_growth"].clip(lower=0)

    total_new = alloc["new_accounts"].sum()
    total_low = alloc["low_spend_growth"].sum()
    total_react = alloc["reactivated_count"].sum()

    alloc["new_share"] = alloc["new_accounts"] / total_new if total_new > 0 else 0.0
    alloc["low_share"] = alloc["low_spend_growth"] / total_low if total_low > 0 else 0.0
    alloc["react_share"] = alloc["reactivated_count"] / total_react if total_react > 0 else 0.0

    alloc["score"] = (
        WEIGHTS["new_accounts"] * alloc["new_share"]
        + WEIGHTS["low_spend_growth"] * alloc["low_share"]
        + WEIGHTS["reactivated_count"] * alloc["react_share"]
    )

    total_score = alloc["score"].sum()
    alloc["share"] = alloc["score"] / total_score if total_score > 0 else 0.0
    alloc["bonus"] = alloc["share"] * pool
    return alloc.sort_values("bonus", ascending=False)


def main():
    st.sidebar.header("Data source")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        source = uploaded_file
        st.sidebar.success("Using uploaded file")
    elif DEFAULT_FILE.exists():
        source = DEFAULT_FILE
        st.sidebar.info(f"Using local file: {DEFAULT_FILE.name}")
    else:
        st.info("Upload a CSV file using the sidebar to get started.")
        st.stop()

    with st.spinner("Loading data..."):
        raw_df = load_csv_data(source)

    with st.spinner("Processing..."):
        df = preprocess_dataset(raw_df)

    if df["invoice_date"].isna().all():
        st.error("No valid invoice dates found.")
        return

    quarter_list = sorted(df["quarter"].dropna().unique())
    current_q = quarter_list[-1]
    prior_quarters = [q for q in quarter_list if q < current_q]
    prior_q = prior_quarters[-1] if prior_quarters else None

    # ── Sidebar filters ────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("Filters")

    all_account_types = ["Franchise", "Individual Store"]
    sel_account_types = st.sidebar.multiselect(
        "Account type", options=all_account_types, default=all_account_types
    )

    all_spend_tiers = ["High Tier", "Medium Tier", "Low Tier"]
    sel_spend_tiers = st.sidebar.multiselect(
        "Spend tier", options=all_spend_tiers, default=all_spend_tiers
    )

    raw_customer_tiers = sorted(df["customer_tier"].dropna().unique().tolist())
    sel_customer_tiers = st.sidebar.multiselect(
        "Customer tier", options=raw_customer_tiers, default=raw_customer_tiers
    )

    tier_mask = df["customer_tier"].isin(sel_customer_tiers) | df["customer_tier"].isna()
    df = df[
        df["account_type"].isin(sel_account_types)
        & df["spend_tier"].isin(sel_spend_tiers)
        & tier_mask
    ]

    pq_label = prior_q or "Prior Q"
    cq_label = current_q

    # ── KPI Dashboard ──────────────────────────────────────────────────────────
    st.subheader(f"KPIs — {current_q}")
    kpis = compute_kpis(df, current_q, prior_q)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Revenue — {pq_label}", fmt(kpis["prior_rev"]))
    growth_str = f"{kpis['growth_pct']:+.1f}%" if kpis["growth_pct"] is not None else None
    col2.metric(f"Revenue — {cq_label}", fmt(kpis["current_rev"]), delta=growth_str)
    col3.metric("New accounts", kpis["new_account_count"], help="First sale ever recorded in this quarter")

    col4, col5, col6 = st.columns(3)
    col4.metric("Accounts grew", kpis["grew_count"], help="vs 2025 quarterly average")
    col5.metric("Accounts declined", kpis["declined_count"], help="vs 2025 quarterly average")
    col6.metric(
        "Inactive accounts",
        kpis["inactive_count"],
        help=f"Bought in {prior_q}, did not buy in {current_q}" if prior_q else "No prior quarter",
    )

    st.markdown("**Key product lines**")
    kpl_cols = st.columns(len(KEY_PRODUCT_LINES))
    for i, line in enumerate(KEY_PRODUCT_LINES):
        d = kpis["kpl"][line]
        delta_str = f"{d['growth_pct']:+.1f}%" if d["growth_pct"] is not None else None
        kpl_cols[i].metric(
            line,
            fmt(d["current"]),
            delta=delta_str,
            help=f"{pq_label}: {fmt(d['prior'])}",
        )

    st.markdown("---")

    # ── Salesperson Breakdown ──────────────────────────────────────────────────
    st.subheader("Salesperson breakdown")

    sp_df = compute_sp_summary(df, current_q, prior_q)

    sp_display = sp_df[[
        "Salesperson", "cq_revenue", "growth_pct",
        "attribution_rev", "attribution_pct",
        "new_accounts", "low_spend_growth", "reactivated_count",
    ]].copy()
    sp_display["cq_revenue"] = sp_display["cq_revenue"].apply(fmt)
    sp_display["growth_pct"] = sp_display["growth_pct"].apply(
        lambda v: f"{v:+.1f}%" if v is not None else "—"
    )
    sp_display["attribution_rev"] = sp_display["attribution_rev"].apply(fmt)
    sp_display["attribution_pct"] = sp_display["attribution_pct"].apply(lambda v: f"{v:.1f}%")
    sp_display["low_spend_growth"] = sp_display["low_spend_growth"].apply(
        lambda v: f"+{fmt(v)}" if v >= 0 else f"-${abs(v):,.0f}"
    )
    sp_display = sp_display.rename(columns={
        "cq_revenue": f"{cq_label} Revenue",
        "growth_pct": "Growth",
        "attribution_rev": "Attribution ($)",
        "attribution_pct": "Attribution (%)",
        "new_accounts": "New Accounts",
        "low_spend_growth": "Low Spend Growth",
        "reactivated_count": "Reactivated",
    })

    st.dataframe(sp_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Customer Detail ────────────────────────────────────────────────────────
    st.subheader("Customer detail")

    # Master salesperson filter — drives all three tables below
    all_sp_options = sorted(
        df[df["current_sp"].notna() & ~df["current_sp"].isin(EXCLUDE_SP)]["current_sp"].unique().tolist()
    )
    selected_detail_sps = st.multiselect(
        "Filter by salesperson",
        options=all_sp_options,
        default=all_sp_options,
        key="detail_sp_filter",
    )

    def apply_sp_filter(table: pd.DataFrame) -> pd.DataFrame:
        return table[table["current_sp"].isin(selected_detail_sps)] if selected_detail_sps else table

    # ── By salesperson ────────────────────────────────────────────────────────
    st.markdown("**By salesperson**")
    sp_table = compute_sp_table(df, current_q, prior_q)
    sp_table_filtered = sp_table[sp_table["Salesperson"].isin(selected_detail_sps)] if selected_detail_sps else sp_table

    sp_tbl_display = sp_table_filtered.copy()
    sp_total_pq = sp_tbl_display["pq_revenue"].sum()
    sp_total_cq = sp_tbl_display["cq_revenue"].sum()
    sp_total_growth = (sp_total_cq - sp_total_pq) / sp_total_pq * 100 if sp_total_pq > 0 else None

    sp_tbl_display["avg_2025_q"] = sp_tbl_display["avg_2025_q"].apply(fmt)
    sp_tbl_display["pq_revenue"] = sp_tbl_display["pq_revenue"].apply(fmt)
    sp_tbl_display["cq_revenue"] = sp_tbl_display["cq_revenue"].apply(fmt)
    sp_tbl_display["growth_pct"] = sp_tbl_display["growth_pct"].apply(
        lambda v: f"{v:+.1f}%" if v is not None else "—"
    )
    sp_tbl_display = sp_tbl_display.rename(columns={
        "avg_2025_q": "2025 Avg/Quarter",
        "pq_revenue": f"{pq_label} Revenue",
        "cq_revenue": f"{cq_label} Revenue",
        "growth_pct": "Growth",
    })
    st.dataframe(sp_tbl_display, use_container_width=True, hide_index=True)
    sp_growth_str = f"{sp_total_growth:+.1f}%" if sp_total_growth is not None else "—"
    st.dataframe(
        pd.DataFrame([{
            "Salesperson": f"TOTAL ({len(sp_tbl_display)} reps)",
            "2025 Avg/Quarter": "—",
            f"{pq_label} Revenue": fmt(sp_total_pq),
            f"{cq_label} Revenue": fmt(sp_total_cq),
            "Growth": sp_growth_str,
        }]),
        use_container_width=True, hide_index=True,
    )

    # ── New accounts ──────────────────────────────────────────────────────────
    st.markdown("**New accounts this quarter**")
    new_tbl = apply_sp_filter(compute_new_accounts_table(df, current_q))
    if new_tbl.empty:
        st.caption("No new accounts this quarter.")
    else:
        new_display = new_tbl[["Customer", "current_sp", "spend_tier", "account_type", "customer_tier", "cq_revenue"]].copy()
        new_display["cq_revenue"] = new_display["cq_revenue"].apply(fmt)
        new_display = new_display.rename(columns={
            "current_sp": "Salesperson", "spend_tier": "Spend Tier",
            "account_type": "Account Type", "customer_tier": "Customer Tier",
            "cq_revenue": f"{cq_label} Revenue",
        })
        st.dataframe(new_display, use_container_width=True, hide_index=True)
        st.dataframe(
            pd.DataFrame([{
                "Customer": f"TOTAL ({len(new_display)} accounts)",
                "Salesperson": "—", "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
                f"{cq_label} Revenue": fmt(new_tbl["cq_revenue"].sum()),
            }]),
            use_container_width=True, hide_index=True,
        )

    # ── Reactivated accounts ──────────────────────────────────────────────────
    st.markdown("**Reactivated accounts** *(absent last quarter, back this quarter)*")
    react_tbl = apply_sp_filter(compute_reactivated_accounts_table(df, current_q, prior_q))
    if react_tbl.empty:
        st.caption("No reactivated accounts this quarter.")
    else:
        react_display = react_tbl[["Customer", "current_sp", "spend_tier", "account_type", "customer_tier", "last_active_q", "cq_revenue"]].copy()
        react_display["cq_revenue"] = react_display["cq_revenue"].apply(fmt)
        react_display = react_display.rename(columns={
            "current_sp": "Salesperson", "spend_tier": "Spend Tier",
            "account_type": "Account Type", "customer_tier": "Customer Tier",
            "last_active_q": "Last Active Quarter",
            "cq_revenue": f"{cq_label} Revenue",
        })
        st.dataframe(react_display, use_container_width=True, hide_index=True)
        st.dataframe(
            pd.DataFrame([{
                "Customer": f"TOTAL ({len(react_display)} accounts)",
                "Salesperson": "—", "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
                "Last Active Quarter": "—",
                f"{cq_label} Revenue": fmt(react_tbl["cq_revenue"].sum()),
            }]),
            use_container_width=True, hide_index=True,
        )

    # ── All accounts ──────────────────────────────────────────────────────────
    st.markdown("**All accounts**")
    cust_table = apply_sp_filter(compute_customer_table(df, current_q, prior_q))
    cust_display = cust_table[
        ["Customer", "current_sp", "spend_tier", "account_type", "customer_tier",
         "avg_2025_q", "pq_revenue", "cq_revenue", "growth_pct"]
    ].copy()

    total_pq = cust_display["pq_revenue"].sum()
    total_cq = cust_display["cq_revenue"].sum()
    total_growth = (total_cq - total_pq) / total_pq * 100 if total_pq > 0 else None

    cust_display["avg_2025_q"] = cust_display["avg_2025_q"].apply(fmt)
    cust_display["pq_revenue"] = cust_display["pq_revenue"].apply(fmt)
    cust_display["cq_revenue"] = cust_display["cq_revenue"].apply(fmt)
    cust_display["growth_pct"] = cust_display["growth_pct"].apply(
        lambda v: f"{v:+.1f}%" if v is not None else "—"
    )
    cust_display = cust_display.rename(columns={
        "current_sp": "Salesperson", "spend_tier": "Spend Tier",
        "account_type": "Account Type", "customer_tier": "Customer Tier",
        "avg_2025_q": "2025 Avg/Quarter",
        "pq_revenue": f"{pq_label} Revenue",
        "cq_revenue": f"{cq_label} Revenue",
        "growth_pct": "Growth",
    })
    st.dataframe(cust_display, use_container_width=True, hide_index=True)
    cust_growth_str = f"{total_growth:+.1f}%" if total_growth is not None else "—"
    st.dataframe(
        pd.DataFrame([{
            "Customer": f"TOTAL ({len(cust_display)} accounts)",
            "Salesperson": "—", "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
            "2025 Avg/Quarter": "—",
            f"{pq_label} Revenue": fmt(total_pq),
            f"{cq_label} Revenue": fmt(total_cq),
            "Growth": cust_growth_str,
        }]),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Bonus Allocation ───────────────────────────────────────────────────────
    st.subheader("Bonus allocation")
    st.caption(
        f"Weighted score: New Accounts {int(WEIGHTS['new_accounts']*100)}% · "
        f"Low Spend Growth {int(WEIGHTS['low_spend_growth']*100)}% · "
        f"Reactivated {int(WEIGHTS['reactivated_count']*100)}%"
    )

    all_reps = sp_df["Salesperson"].tolist()

    selected_reps = st.multiselect(
        "Salespeople in bonus pool",
        options=all_reps,
        default=all_reps,
    )

    pool = st.number_input(
        "Total bonus pool ($)",
        min_value=0.0,
        value=200000.0,
        step=1000.0,
        format="%.2f",
    )

    if not selected_reps:
        st.warning("Select at least one salesperson.")
        return

    alloc_df = compute_bonus_allocation(selected_reps, sp_df, pool)

    a1, a2 = st.columns(2)
    a1.metric("Reps in pool", len(selected_reps))
    a2.metric("Total allocated", fmt(alloc_df["bonus"].sum()))

    display = alloc_df.copy()
    display["low_spend_growth"] = display["low_spend_growth"].apply(fmt)
    display["score"] = (display["score"] * 100).round(1).astype(str) + "%"
    display["share"] = (display["share"] * 100).round(1).astype(str) + "%"
    display["bonus"] = display["bonus"].apply(fmt)
    st.dataframe(
        display.rename(columns={
            "new_accounts": "New Accounts",
            "low_spend_growth": "Low Spend Growth",
            "reactivated_count": "Reactivated",
            "score": "Weighted Score",
            "share": "Pool Share",
            "bonus": "Bonus",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download allocation as CSV",
        data=alloc_df.to_csv(index=False).encode("utf-8"),
        file_name=f"bonus_allocation_{current_q}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
