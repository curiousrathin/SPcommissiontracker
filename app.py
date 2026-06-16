import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional

DEFAULT_FILE = next(
    (p for p in [Path("FrankieDS.csv"), Path("FrankieDS.csv.gz")] if p.exists()),
    Path("FrankieDS.csv"),
)
OTHER_SP_THRESHOLD = 30_000  # SPs with < this in current quarter → grouped as "Other"
TIER_ORDER = ["Low Tier", "Medium Tier", "High Tier"]
TIER_RATE = {"Low Tier": 0.02, "Medium Tier": 0.01, "High Tier": 0.01}
NEW_INCENTIVE = 50.0
REACTIVATED_INCENTIVE = 100.0

st.set_page_config(page_title="Smoke Arsenal Incentive Tool", layout="wide")
st.title("Smoke Arsenal — Incentive Allocation")


REQUIRED_COLUMNS = {
    "Customer", "Invoice Date", "Untaxed Total",
    "Current Salesperson", "Salesperson", "Product",
    "Customer Tier", "Account Type", "Order",
}
OPTIONAL_COLUMNS = {"Franchise", "Delivery Address", "Zip", "Brand"}


@st.cache_data(show_spinner=False)
def load_csv_data(source):
    all_cols = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    raw = pd.read_csv(source, dtype=str, usecols=lambda c: c.strip("﻿") in all_cols)
    return raw


@st.cache_data(show_spinner=False)
def preprocess_dataset(df: pd.DataFrame, exclude_sa: bool = True) -> pd.DataFrame:
    df = df.copy()
    if exclude_sa:
        df = df[~df["Customer"].str.contains("smoke arsenal", case=False, na=False)]

    df["invoice_date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["Untaxed Total"], errors="coerce").fillna(0.0).astype("float32")
    df = df.drop(columns=["Invoice Date", "Untaxed Total"])

    df = df.rename(columns={
        "Current Salesperson": "current_sp",
        "Product": "product",
        "Customer Tier": "customer_tier",
        "Account Type": "account_type",
        "Franchise": "franchise",
    })
    # Preserved pre-grouping, so new-account eligibility (current_sp == Salesperson on
    # the first invoice) still works correctly after apply_sp_grouping relabels small
    # reps' current_sp to "Other".
    df["original_sp"] = df["current_sp"]
    if "franchise" not in df.columns:
        df["franchise"] = ""
    else:
        df["franchise"] = df["franchise"].fillna("").str.strip()

    for addr_col in ["Delivery Address", "Zip"]:
        if addr_col not in df.columns:
            df[addr_col] = ""
        else:
            df[addr_col] = df[addr_col].fillna("").str.strip()

    if "Brand" not in df.columns:
        df["Brand"] = ""
    else:
        df["Brand"] = df["Brand"].fillna("").str.strip()

    df["Customer"] = df["Customer"].str.strip().str.title()
    df["quarter"] = df["invoice_date"].dt.to_period("Q").astype(str)

    df["account_type"] = np.where(
        df["account_type"].fillna("").str.strip() == "Franchise",
        "Franchise",
        "Individual Store",
    )

    for col in ["current_sp", "Salesperson", "product", "customer_tier", "account_type", "Customer"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # A Franchise value is only trustworthy when Customer Tier is also populated —
    # rows with a Franchise but no Customer Tier are data errors, so treat them as
    # non-franchise (individual) for every downstream calculation.
    df.loc[df["customer_tier"].astype(str).str.strip() == "", "franchise"] = ""

    for col in ["current_sp", "Salesperson", "product", "customer_tier", "account_type", "Customer"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Spend Tier — based on each customer's average quarterly revenue across 2025
    df_2025 = df[df["quarter"].astype(str).str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    cust_2025_avg = (
        df_2025.groupby("Customer", observed=True)["revenue"].sum() / n_2025_q
    ).rename("_avg_2025_q")

    spend_tier_map = pd.cut(
        cust_2025_avg,
        bins=[-np.inf, 5000.0, 10000.0, np.inf],
        labels=["Low Tier", "Medium Tier", "High Tier"],
    ).rename("spend_tier")
    df = df.join(spend_tier_map, on="Customer")
    df["spend_tier"] = df["spend_tier"].fillna("Low Tier").astype("category")

    # For franchise accounts: override spend tier using the franchise group's
    # combined 2025 quarterly average across all stores in the same franchise.
    # One store may hold the PO and distribute internally — the group should be
    # judged together, not penalised for how the purchase happens to flow.
    fr_customers = df[df["franchise"] != ""][["Customer", "franchise"]].drop_duplicates(subset="Customer", keep="first")
    if not fr_customers.empty:
        cust_to_fr = fr_customers.set_index("Customer")["franchise"]
        fr_2025 = df_2025[df_2025["Customer"].isin(fr_customers["Customer"])].copy()
        fr_2025["_franchise"] = fr_2025["Customer"].map(cust_to_fr)
        fr_agg = fr_2025.groupby("_franchise")["revenue"].sum() / n_2025_q
        fr_tier = pd.cut(
            fr_agg,
            bins=[-np.inf, 5000.0, 10000.0, np.inf],
            labels=["Low Tier", "Medium Tier", "High Tier"],
        )
        # Map franchise-level tier back to each franchise customer row
        fr_tier_per_cust = cust_to_fr.map(fr_tier)
        fr_idx = df[df["franchise"] != ""].index
        df["spend_tier"] = df["spend_tier"].astype(str)
        df.loc[fr_idx, "spend_tier"] = df.loc[fr_idx, "Customer"].map(fr_tier_per_cust).fillna("Low Tier").values
        df["spend_tier"] = df["spend_tier"].astype("category")

    return df


@st.cache_data(show_spinner=False)
def apply_sp_grouping(df: pd.DataFrame, current_q: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Relabel current_sp to 'Other' for any SP whose total revenue in current_q
    is below OTHER_SP_THRESHOLD. Returns the modified df and a sorted list of
    the original SP names that were moved into 'Other'.
    """
    df = df.copy()
    cq_sp_rev = (
        df[df["quarter"] == current_q]
        .groupby("current_sp", observed=True)["revenue"]
        .sum()
    )
    qualified = set(cq_sp_rev[cq_sp_rev >= OTHER_SP_THRESHOLD].index.tolist())
    # Collect names that go into Other (skip blank)
    all_sps = set(cq_sp_rev.index.tolist())
    other_sps = sorted(sp for sp in (all_sps - qualified) if sp != "")

    def _remap(sp: str) -> str:
        return sp if sp in qualified else "Other"

    df["current_sp"] = df["current_sp"].astype(str).apply(_remap).astype("category")
    return df, other_sps


def fmt(value: float) -> str:
    return f"${value:,.0f}"


def fmt_quarter(q: str) -> str:
    if not q or q in ("", "—", "nan"):
        return "—"
    try:
        year, qnum = q.split("Q")
        return f"Q{qnum} {year}"
    except Exception:
        return q


def _bold_last_row(df: pd.DataFrame):
    """Return a Styler with the last row bolded (for total rows)."""
    last_idx = df.index[-1]
    def row_style(row):
        weight = "font-weight: bold" if row.name == last_idx else ""
        return [weight] * len(row)
    return df.style.apply(row_style, axis=1)


@st.cache_data(show_spinner=False)
def get_new_accounts_df(df: pd.DataFrame, current_q: str) -> pd.DataFrame:
    """
    Returns a DataFrame of genuinely new accounts in current_q.

    A new account qualifies only when BOTH conditions hold:
      1. The customer's very first invoice in the entire dataset falls in current_q.
      2. The original_sp on that first invoice is also the Salesperson on that invoice
         (i.e. the rep who owns the account today was the one who made the first sale).
         original_sp (not current_sp) is used here because current_sp may already have
         been relabelled to "Other" by apply_sp_grouping for smaller reps — that bucketing
         is for display/rollup purposes only and shouldn't affect new-account eligibility.

    Returns one row per customer with column: Customer.
    """
    cq_df = df[df["quarter"] == current_q]
    ever_before = set(df[df["quarter"] < current_q]["Customer"].dropna().unique())
    new_custs = cq_df[~cq_df["Customer"].isin(ever_before)]

    if new_custs.empty:
        return pd.DataFrame(columns=["Customer"])

    # For each new customer, grab their first invoice in current_q
    first_inv = (
        new_custs.sort_values("invoice_date")
        .groupby("Customer", observed=True)[["original_sp", "Salesperson"]]
        .first()
        .reset_index()
    )

    # Only credit if original_sp == Salesperson on the first invoice
    credited = first_inv[
        first_inv["original_sp"].astype(str) == first_inv["Salesperson"].astype(str)
    ][["Customer"]]

    return credited


@st.cache_data(show_spinner=False)
def build_master_table(df: pd.DataFrame, current_q: str, prior_q: Optional[str]) -> pd.DataFrame:
    """
    One row per customer active in current_q. All facts resolved from the same
    source so every downstream table can be derived consistently.

    Columns:
      Customer, current_sp, spend_tier, account_type, customer_tier,
      cq_revenue, pq_revenue, avg_2025_q, growth_pct,
      is_new, is_reactivated
    """
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    # Revenue per customer
    cq_rev = cq_df.groupby("Customer", observed=True)["revenue"].sum().rename("cq_revenue")
    pq_rev = (
        pq_df.groupby("Customer", observed=True)["revenue"].sum().rename("pq_revenue")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_revenue")
    )

    # 2025 avg quarterly revenue per customer
    df_2025 = df[df["quarter"].astype(str).str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    avg_2025 = (
        df_2025.groupby("Customer", observed=True)["revenue"].sum() / n_2025_q
    ).rename("avg_2025_q")

    # Yearly revenue totals
    def _year_rev(year):
        return (
            df[df["invoice_date"].dt.year == year]
            .groupby("Customer", observed=True)["revenue"].sum()
            .rename(f"sale_{year}")
        )
    sale_2024 = _year_rev(2024)
    sale_2025 = _year_rev(2025)
    sale_2026 = _year_rev(2026)

    # STLTH-related columns (2025 only)
    stlth_2025 = (
        df[
            (df["invoice_date"].dt.year == 2025) &
            df["Brand"].str.contains("STLTH", case=False, na=False)
        ]
        .groupby("Customer", observed=True)["revenue"].sum()
        .rename("stlth_2025")
    )

    # Attributes: CQ takes priority, then PQ, then most recent invoice ever
    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier", "franchise"]
    addr_cols = ["Delivery Address", "Zip"]
    def _last_meta(frame, cols):
        return (
            frame.sort_values("invoice_date")
            .groupby("Customer", observed=True)[cols].last()
            .astype(str)
        )
    cq_meta = _last_meta(cq_df, meta_cols)
    all_meta = _last_meta(df, meta_cols)
    if not pq_df.empty:
        combined_meta = cq_meta.combine_first(_last_meta(pq_df, meta_cols)).combine_first(all_meta)
    else:
        combined_meta = cq_meta.combine_first(all_meta)

    # Address fields: most recent invoice ever per customer
    addr_meta = _last_meta(df, addr_cols)

    # All customers ever seen — base index so no one is dropped
    all_customers = pd.Index(
        df["Customer"].dropna().astype(str).unique(), name="Customer"
    )
    master = (
        pd.DataFrame(index=all_customers)
        .join(cq_rev, how="left")
        .join(pq_rev, how="left")
        .join(avg_2025, how="left")
        .join(sale_2024, how="left")
        .join(sale_2025, how="left")
        .join(sale_2026, how="left")
        .join(stlth_2025, how="left")
        .join(combined_meta, how="left")
        .join(addr_meta, how="left")
        .fillna({"cq_revenue": 0.0, "pq_revenue": 0.0, "avg_2025_q": 0.0,
                 "sale_2024": 0.0, "sale_2025": 0.0, "sale_2026": 0.0,
                 "stlth_2025": 0.0})
        .reset_index()
    )
    master["sale_2025_ex_stlth"] = master["sale_2025"] - master["stlth_2025"]
    master["avg_2025_q_ex_stlth"] = master["sale_2025_ex_stlth"] / n_2025_q

    master["growth_pct"] = np.where(
        master["pq_revenue"] > 0,
        (master["cq_revenue"] - master["pq_revenue"]) / master["pq_revenue"] * 100,
        None,
    )

    # New: first invoice ever is in current_q AND current_sp == Salesperson on that invoice
    # (matches get_new_accounts_df — same rule used by KPIs), and must have CQ revenue.
    new_acct_customers = set(get_new_accounts_df(df, current_q)["Customer"].unique())
    master["is_new"] = (
        master["Customer"].isin(new_acct_customers)
        & (master["cq_revenue"] > 0)
    )

    # Reactivated: not in PQ, seen before PQ, AND placed >= 2 orders or ordered in >= 2 months in CQ
    pq_customers = set(pq_df["Customer"].dropna().unique()) if not pq_df.empty else set()
    pre_pq_q = prior_q if prior_q else current_q
    pre_pq_customers = set(df[df["quarter"] < pre_pq_q]["Customer"].dropna().unique())
    cq_activity = cq_df.groupby("Customer", observed=True).agg(
        _order_count=("Order", "nunique"),
        _month_count=("invoice_date", lambda x: x.dt.month.nunique()),
    )
    cq_min_activity = set(
        cq_activity[
            (cq_activity["_order_count"] >= 2) | (cq_activity["_month_count"] >= 2)
        ].index.tolist()
    )
    master["is_reactivated"] = (
        (master["cq_revenue"] > 0)
        & ~master["Customer"].isin(pq_customers)
        & master["Customer"].isin(pre_pq_customers)
        & master["Customer"].isin(cq_min_activity)
    )

    # Last quarter with purchases, populated only for reactivated accounts
    last_q_per_cust = (
        df[df["quarter"] < current_q]
        .groupby("Customer", observed=True)["quarter"]
        .max()
    )
    master["last_purchase_quarter"] = (
        master["Customer"].map(last_q_per_cust)
        .where(master["is_reactivated"], other="")
        .fillna("")
    )

    # Growth amount = CQ revenue − 2025 avg/Q ex-STLTH, floored at 0 (shrunken accounts earn nothing)
    master["growth_amount"] = (master["cq_revenue"] - master["avg_2025_q_ex_stlth"]).clip(lower=0)
    master.loc[master["cq_revenue"] == 0, "growth_amount"] = 0.0

    # Growth commission: Low Tier 2%, Medium/High Tier 1%
    _rates = master["spend_tier"].astype(str).map(TIER_RATE).fillna(0.01)
    master["growth_commission"] = (master["growth_amount"] * _rates).round(2)

    # Flat incentives
    master["new_incentive"] = np.where(master["is_new"], NEW_INCENTIVE, 0.0)
    master["reactivated_incentive"] = np.where(master["is_reactivated"], REACTIVATED_INCENTIVE, 0.0)

    return master.sort_values("cq_revenue", ascending=False)


def compute_growth_groups(master: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate growth at the franchise-group level (or per-customer when there is no
    franchise), split by current_sp. A salesperson can't control which exact store in
    their franchise places an order — product gets moved around internally — so growth
    is judged for the group as a whole rather than rewarding store-to-store reshuffling.
    """
    m = master.copy()
    m["group_key"] = np.where(m["franchise"] != "", m["franchise"], m["Customer"])
    grp = (
        m.groupby(["group_key", "current_sp"], observed=True)
        .agg(
            cq_revenue=("cq_revenue", "sum"),
            avg_2025_q_ex_stlth=("avg_2025_q_ex_stlth", "sum"),
            spend_tier=("spend_tier", "first"),
            is_franchise=("franchise", lambda x: (x != "").any()),
        )
        .reset_index()
    )
    grp["growth_amount"] = (grp["cq_revenue"] - grp["avg_2025_q_ex_stlth"]).clip(lower=0)
    grp.loc[grp["cq_revenue"] == 0, "growth_amount"] = 0.0
    grp["rate"] = grp["spend_tier"].astype(str).map(TIER_RATE).fillna(0.01)
    grp["growth_commission"] = (grp["growth_amount"] * grp["rate"]).round(2)
    return grp


def render_summary(df: pd.DataFrame, current_q: str, prior_q: Optional[str], pq_label: str, cq_label: str) -> None:
    st.subheader("Summary")
    st.caption(
        "High-level view, aggregated from the master table. Franchise growth and "
        "commission are calculated at the franchise-group level so internal stock "
        "movement between stores doesn't get rewarded as growth."
    )

    master = build_master_table(df, current_q, prior_q)

    sp_options = sorted(master["current_sp"].dropna().unique().tolist())
    sel_sp = st.multiselect("Current Salesperson", options=sp_options, default=sp_options, key="summary_sp")
    filtered = master[master["current_sp"].isin(sel_sp)]

    if filtered.empty:
        st.warning("No data for the selected salespeople.")
        return

    grp = compute_growth_groups(filtered)
    tier_growth = (
        grp.groupby("spend_tier", observed=True)["growth_amount"]
        .sum().reindex(TIER_ORDER).fillna(0.0)
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{cq_label} Revenue", fmt(filtered["cq_revenue"].sum()))
    k2.metric(f"{pq_label} Revenue", fmt(filtered["pq_revenue"].sum()))
    k3.metric("New Accounts", int(filtered["is_new"].sum()))
    k4.metric("Reactivations", int(filtered["is_reactivated"].sum()))

    t1, t2, t3 = st.columns(3)
    t1.metric("Low Tier Growth", fmt(tier_growth["Low Tier"]))
    t2.metric("Medium Tier Growth", fmt(tier_growth["Medium Tier"]))
    t3.metric("High Tier Growth", fmt(tier_growth["High Tier"]))

    st.markdown("---")

    # ── By salesperson ───────────────────────────────────────────────────────
    st.markdown("**By salesperson**")
    by_sp_tier = (
        grp.groupby(["current_sp", "spend_tier"], observed=True)["growth_amount"]
        .sum().unstack(fill_value=0.0).reindex(columns=TIER_ORDER, fill_value=0.0)
    )
    commission_by_sp = grp.groupby("current_sp", observed=True)["growth_commission"].sum()
    new_by_sp = filtered.groupby("current_sp", observed=True)["is_new"].sum()
    react_by_sp = filtered.groupby("current_sp", observed=True)["is_reactivated"].sum()

    sp_table = by_sp_tier.copy()
    sp_table["growth_commission"] = commission_by_sp
    sp_table["new_accounts"] = new_by_sp
    sp_table["reactivated"] = react_by_sp
    sp_table = sp_table.fillna(0.0).reset_index().rename(columns={"current_sp": "Salesperson"})
    sp_table["new_incentive"] = sp_table["new_accounts"] * NEW_INCENTIVE
    sp_table["reactivated_incentive"] = sp_table["reactivated"] * REACTIVATED_INCENTIVE
    sp_table["total_commission"] = (
        sp_table["growth_commission"] + sp_table["new_incentive"] + sp_table["reactivated_incentive"]
    )
    sp_table = sp_table.sort_values("total_commission", ascending=False)

    sp_display = pd.DataFrame({
        "Salesperson": sp_table["Salesperson"],
        "Low Tier Growth": sp_table["Low Tier"].apply(fmt),
        "Medium Tier Growth": sp_table["Medium Tier"].apply(fmt),
        "High Tier Growth": sp_table["High Tier"].apply(fmt),
        "New Accounts": sp_table["new_accounts"].astype(int),
        "Reactivated": sp_table["reactivated"].astype(int),
        "Total Commission": sp_table["total_commission"].apply(fmt),
    })
    sp_total_row = pd.DataFrame([{
        "Salesperson": f"TOTAL ({len(sp_table)} reps)",
        "Low Tier Growth": fmt(sp_table["Low Tier"].sum()),
        "Medium Tier Growth": fmt(sp_table["Medium Tier"].sum()),
        "High Tier Growth": fmt(sp_table["High Tier"].sum()),
        "New Accounts": int(sp_table["new_accounts"].sum()),
        "Reactivated": int(sp_table["reactivated"].sum()),
        "Total Commission": fmt(sp_table["total_commission"].sum()),
    }])
    st.dataframe(
        _bold_last_row(pd.concat([sp_display, sp_total_row], ignore_index=True)),
        use_container_width=True, hide_index=True,
    )

    with st.expander("View commission math for a rep"):
        rep_pick = st.selectbox("Salesperson", options=sp_table["Salesperson"].tolist(), key="summary_rep_pick")
        row = sp_table[sp_table["Salesperson"] == rep_pick].iloc[0]
        st.markdown(
            f"- Low Tier growth: {fmt(row['Low Tier'])} × 2% = **{fmt(row['Low Tier'] * 0.02)}**\n"
            f"- Medium Tier growth: {fmt(row['Medium Tier'])} × 1% = **{fmt(row['Medium Tier'] * 0.01)}**\n"
            f"- High Tier growth: {fmt(row['High Tier'])} × 1% = **{fmt(row['High Tier'] * 0.01)}**\n"
            f"- New Accounts: {int(row['new_accounts'])} × $50 = **{fmt(row['new_incentive'])}**\n"
            f"- Reactivated: {int(row['reactivated'])} × $100 = **{fmt(row['reactivated_incentive'])}**\n"
            f"- **Total = {fmt(row['total_commission'])}**"
        )

    st.markdown("---")

    # ── By customer / franchise ──────────────────────────────────────────────
    st.markdown("**By customer / franchise** *(click a franchise to drill in)*")

    fr_totals = (
        grp[grp["is_franchise"]]
        .groupby("group_key", observed=True)
        .agg(
            cq_revenue=("cq_revenue", "sum"),
            growth_amount=("growth_amount", "sum"),
            growth_commission=("growth_commission", "sum"),
        )
        .sort_values("cq_revenue", ascending=False)
    )
    for fr_name, fr_row in fr_totals.iterrows():
        fr_stores = filtered[filtered["franchise"] == fr_name].sort_values("cq_revenue", ascending=False)
        n_stores = len(fr_stores)
        header = (
            f"{fr_name}  —  {fmt(fr_row['cq_revenue'])}  |  "
            f"Growth {fmt(fr_row['growth_amount'])}  |  Commission {fmt(fr_row['growth_commission'])}  |  {n_stores} stores"
        )
        with st.expander(header):
            store_display = fr_stores[[
                "Customer", "current_sp", "cq_revenue", "pq_revenue",
                "avg_2025_q", "avg_2025_q_ex_stlth", "is_new", "is_reactivated",
            ]].copy()
            store_display[cq_label] = store_display["cq_revenue"].apply(fmt)
            store_display[pq_label] = store_display["pq_revenue"].apply(fmt)
            store_display["2025 Avg/Q"] = store_display["avg_2025_q"].apply(fmt)
            store_display["2025 Avg/Q (ex-STLTH)"] = store_display["avg_2025_q_ex_stlth"].apply(fmt)
            store_display["New"] = store_display["is_new"].map({True: "Yes", False: ""})
            store_display["Reactivated"] = store_display["is_reactivated"].map({True: "Yes", False: ""})
            store_display = store_display.rename(columns={"current_sp": "Salesperson"})
            cols = [
                "Customer", "Salesperson", pq_label, cq_label,
                "2025 Avg/Q", "2025 Avg/Q (ex-STLTH)", "New", "Reactivated",
            ]
            st.dataframe(store_display[cols], use_container_width=True, hide_index=True)

    individual = filtered[filtered["franchise"] == ""].sort_values("cq_revenue", ascending=False)
    if not individual.empty:
        st.markdown("**Individual stores**")
        ind_display = individual.copy()
        ind_display[cq_label] = ind_display["cq_revenue"].apply(fmt)
        ind_display[pq_label] = ind_display["pq_revenue"].apply(fmt)
        ind_display["2025 Avg/Q"] = ind_display["avg_2025_q"].apply(fmt)
        ind_display["2025 Avg/Q (ex-STLTH)"] = ind_display["avg_2025_q_ex_stlth"].apply(fmt)
        ind_display["Growth Amount"] = ind_display["growth_amount"].apply(lambda v: fmt(v) if v > 0 else "—")
        ind_display["Growth Commission"] = ind_display["growth_commission"].apply(lambda v: fmt(v) if v > 0 else "—")
        ind_display["New"] = ind_display["is_new"].map({True: "Yes", False: ""})
        ind_display["Reactivated"] = ind_display["is_reactivated"].map({True: "Yes", False: ""})
        ind_display["Spend Tier"] = ind_display["spend_tier"]
        ind_display = ind_display.rename(columns={"current_sp": "Salesperson"})
        cols = [
            "Customer", "Salesperson", "Spend Tier", pq_label, cq_label,
            "2025 Avg/Q", "2025 Avg/Q (ex-STLTH)", "Growth Amount", "Growth Commission",
            "New", "Reactivated",
        ]
        ind_total = pd.DataFrame([{
            "Customer": f"TOTAL ({len(individual)} stores)",
            "Salesperson": "—", "Spend Tier": "—",
            pq_label: fmt(individual["pq_revenue"].sum()),
            cq_label: fmt(individual["cq_revenue"].sum()),
            "2025 Avg/Q": fmt(individual["avg_2025_q"].sum()),
            "2025 Avg/Q (ex-STLTH)": fmt(individual["avg_2025_q_ex_stlth"].sum()),
            "Growth Amount": fmt(individual["growth_amount"].sum()),
            "Growth Commission": fmt(individual["growth_commission"].sum()),
            "New": str(int(individual["is_new"].sum())),
            "Reactivated": str(int(individual["is_reactivated"].sum())),
        }])
        st.dataframe(
            _bold_last_row(pd.concat([ind_display[cols], ind_total], ignore_index=True)),
            use_container_width=True, hide_index=True,
        )


def render_master_table(df: pd.DataFrame, current_q: str, prior_q: Optional[str], pq_label: str, cq_label: str, key_prefix: str = "lab", show_excluded_col: bool = False) -> None:
    st.subheader("Master Table")
    st.caption(
        "Single source of truth: one row per customer, all facts resolved the same way. "
        "Every other tab aggregates from this table."
    )

    master = build_master_table(df, current_q, prior_q)
    n_2025_q = df[df["quarter"].astype(str).str.startswith("2025")]["quarter"].nunique() or 1

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    sp_options = sorted(master["current_sp"].dropna().unique().tolist())
    sel_sp = f1.multiselect("Salesperson", options=sp_options, default=sp_options, key=f"{key_prefix}_sp")
    tier_options = ["High Tier", "Medium Tier", "Low Tier"]
    sel_tiers = f2.multiselect("Spend tier", options=tier_options, default=tier_options, key=f"{key_prefix}_tier")

    filtered = master[
        master["current_sp"].isin(sel_sp) & master["spend_tier"].isin(sel_tiers)
    ]

    total_cq = filtered["cq_revenue"].sum()
    total_pq = filtered["pq_revenue"].sum()
    total_growth = (total_cq - total_pq) / total_pq * 100 if total_pq > 0 else None

    cust_display = filtered.copy()
    cust_display[cq_label] = cust_display["cq_revenue"].apply(fmt)
    cust_display[pq_label] = cust_display["pq_revenue"].apply(fmt)
    cust_display["2025 Avg/Q"] = cust_display["avg_2025_q"].apply(fmt)
    cust_display["Growth"] = cust_display["growth_pct"].apply(lambda v: f"{v:+.1f}%" if v is not None else "—")
    cust_display["New"] = cust_display["is_new"].map({True: "Yes", False: ""})
    cust_display["New Incentive"] = cust_display["new_incentive"].apply(lambda v: fmt(v) if v > 0 else "—")
    cust_display["Reactivated"] = cust_display["is_reactivated"].map({True: "Yes", False: ""})
    cust_display["Last Purchase Q"] = cust_display["last_purchase_quarter"].apply(fmt_quarter)
    cust_display["Reactivated Incentive"] = cust_display["reactivated_incentive"].apply(lambda v: fmt(v) if v > 0 else "—")
    cust_display["2024 Sale"] = cust_display["sale_2024"].apply(fmt)
    cust_display["2025 Sale"] = cust_display["sale_2025"].apply(fmt)
    cust_display["2026 Sale"] = cust_display["sale_2026"].apply(fmt)
    cust_display["STLTH Purchase"] = cust_display["stlth_2025"].apply(fmt)
    cust_display["2025 Sale (ex-STLTH)"] = cust_display["sale_2025_ex_stlth"].apply(fmt)
    cust_display["2025 Avg/Q (ex-STLTH)"] = cust_display["avg_2025_q_ex_stlth"].apply(fmt)
    cust_display["Growth Amount"] = cust_display["growth_amount"].apply(lambda v: fmt(v) if v > 0 else "—")
    cust_display["Growth Commission"] = cust_display["growth_commission"].apply(lambda v: fmt(v) if v > 0 else "—")
    cust_display = cust_display.rename(columns={
        "current_sp": "Salesperson", "spend_tier": "Spend Tier",
        "account_type": "Account Type", "customer_tier": "Customer Tier",
        "franchise": "Franchise",
    })
    cust_display["Franchise"] = cust_display.apply(
        lambda r: r["Franchise"] if (r["Customer Tier"] not in ("", "nan", "—") and r["Franchise"] not in ("", "nan")) else "—",
        axis=1,
    )
    for ac in ["Delivery Address", "Zip"]:
        cust_display[ac] = cust_display[ac].replace("", "—")
    if show_excluded_col:
        cust_display["Excluded"] = cust_display["Customer"].str.contains(
            "smoke arsenal", case=False, na=False
        ).map({True: "Yes", False: ""})
    cols = [
        "Customer",
        *( ["Excluded"] if show_excluded_col else []),
        "Salesperson", "Franchise", "Delivery Address", "Zip",
        "Spend Tier", "Account Type", "Customer Tier",
        "2025 Avg/Q", pq_label, cq_label, "Growth",
        "2024 Sale", "2025 Sale", "2026 Sale",
        "STLTH Purchase", "2025 Sale (ex-STLTH)", "2025 Avg/Q (ex-STLTH)",
        "Growth Amount", "Growth Commission",
        "New", "New Incentive", "Reactivated", "Last Purchase Q", "Reactivated Incentive",
    ]
    cust_total = pd.DataFrame([{
        "Customer": f"TOTAL ({len(cust_display)} accounts)",
        **( {"Excluded": "—"} if show_excluded_col else {}),
        "Salesperson": "—", "Franchise": "—", "Delivery Address": "—", "Zip": "—",
        "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
        "2025 Avg/Q": "—",
        pq_label: fmt(filtered["pq_revenue"].sum()),
        cq_label: fmt(filtered["cq_revenue"].sum()),
        "Growth": f"{total_growth:+.1f}%" if total_growth is not None else "—",
        "2024 Sale": fmt(filtered["sale_2024"].sum()),
        "2025 Sale": fmt(filtered["sale_2025"].sum()),
        "2026 Sale": fmt(filtered["sale_2026"].sum()),
        "STLTH Purchase": fmt(filtered["stlth_2025"].sum()),
        "2025 Sale (ex-STLTH)": fmt(filtered["sale_2025_ex_stlth"].sum()),
        "2025 Avg/Q (ex-STLTH)": fmt(filtered["sale_2025_ex_stlth"].sum() / n_2025_q),
        "Growth Amount": fmt(filtered["growth_amount"].sum()),
        "Growth Commission": fmt(filtered["growth_commission"].sum()),
        "New": str(int(filtered["is_new"].sum())),
        "New Incentive": fmt(filtered["new_incentive"].sum()),
        "Reactivated": str(int(filtered["is_reactivated"].sum())),
        "Last Purchase Q": "—",
        "Reactivated Incentive": fmt(filtered["reactivated_incentive"].sum()),
    }])
    st.dataframe(
        _bold_last_row(pd.concat([cust_display[cols], cust_total], ignore_index=True)),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download master table as CSV",
        data=cust_display[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"master_table_{current_q}.csv",
        mime="text/csv",
    )


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

    try:
        with st.spinner("Loading data..."):
            raw_df = load_csv_data(source)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    try:
        with st.spinner("Processing..."):
            df = preprocess_dataset(raw_df, exclude_sa=True)
            df_full = preprocess_dataset(raw_df, exclude_sa=False)
    except Exception as e:
        st.error(f"Failed to process data: {e}")
        st.write("**Columns found in your CSV:**", list(raw_df.columns))
        return

    if df["invoice_date"].isna().all():
        st.error("No valid invoice dates found.")
        return

    all_quarters = sorted(df["quarter"].dropna().unique())
    if not all_quarters:
        st.error("No quarters found in the dataset.")
        return

    this_q = f"{pd.Period.now('Q').year}Q{pd.Period.now('Q').quarter}"
    completed = [q for q in all_quarters if q < this_q]
    default_q = completed[-1] if completed else all_quarters[-1]

    st.sidebar.markdown("---")
    st.sidebar.header("Quarter")
    current_q = st.sidebar.selectbox(
        "Quarter being assessed",
        options=all_quarters,
        index=all_quarters.index(default_q),
        format_func=fmt_quarter,
    )
    prior_qs = [q for q in all_quarters if q < current_q]
    prior_q = prior_qs[-1] if prior_qs else None

    # Apply SP grouping: any SP with < $30k in current_q becomes "Other"
    df, _ = apply_sp_grouping(df, current_q)
    df_full, _ = apply_sp_grouping(df_full, current_q)

    pq_label = fmt_quarter(prior_q) if prior_q else "Prior Q"
    cq_label = fmt_quarter(current_q)

    tab_summary, tab_master, tab_full = st.tabs(
        ["Summary", "Master Table", "Full Dataset (incl. excluded customers)"]
    )

    with tab_summary:
        render_summary(df, current_q, prior_q, pq_label, cq_label)

    with tab_master:
        # ── Sidebar filters ────────────────────────────────────────────────────
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
        df_filtered = df[
            df["account_type"].isin(sel_account_types)
            & df["spend_tier"].isin(sel_spend_tiers)
            & tier_mask
        ]

        render_master_table(df_filtered, current_q, prior_q, pq_label, cq_label, key_prefix="lab")

    with tab_full:
        st.caption(
            "Unfiltered view — includes all customers, including internal Smoke Arsenal rows "
            "excluded from the Master Table. Use this to verify the full dataset row counts and totals."
        )
        render_master_table(df_full, current_q, prior_q, pq_label, cq_label, key_prefix="full", show_excluded_col=True)


if __name__ == "__main__":
    main()
