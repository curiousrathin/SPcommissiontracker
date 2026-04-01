import pandas as pd
import streamlit as st
from pathlib import Path

DEFAULT_FILE = Path("FrankieDS.csv.gz")
KEY_PRODUCT_LINES = ["BC10K", "GH20K", "Nuud 50K"]
OTHER_SP_THRESHOLD = 30_000  # SPs with < this in current quarter → grouped as "Other"

st.set_page_config(page_title="Smoke Arsenal Incentive Tool", layout="wide")
st.title("Smoke Arsenal — Incentive Allocation")


REQUIRED_COLUMNS = {
    "Customer", "Invoice Date", "Untaxed Total",
    "Current Salesperson", "Salesperson", "Product",
    "Customer Tier", "Account Type",
}


@st.cache_data(show_spinner=False)
def load_csv_data(source):
    raw = pd.read_csv(source, dtype=str)
    keep = [c for c in raw.columns if c in REQUIRED_COLUMNS]
    return raw[keep]


@st.cache_data(show_spinner=False)
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Only exclusion: rows where Customer contains "smoke arsenal" (any casing)
    df = df[~df["Customer"].str.contains("smoke arsenal", case=False, na=False)]

    df["invoice_date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
    df["revenue"] = pd.to_numeric(df["Untaxed Total"], errors="coerce").fillna(0.0).astype("float32")
    df = df.drop(columns=["Invoice Date", "Untaxed Total"])

    df = df.rename(columns={
        "Current Salesperson": "current_sp",
        "Product": "product",
        "Customer Tier": "customer_tier",
        "Account Type": "account_type",
    })

    df["Customer"] = df["Customer"].str.strip().str.title()
    df["quarter"] = df["invoice_date"].dt.to_period("Q").astype(str)

    df["account_type"] = (
        df["account_type"].fillna("").str.strip()
        .apply(lambda v: "Franchise" if v == "Franchise" else "Individual Store")
    )

    for col in ["current_sp", "Salesperson", "product", "customer_tier", "account_type", "Customer"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    for col in ["current_sp", "Salesperson", "product", "customer_tier", "account_type", "Customer"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Spend Tier — based on each customer's average quarterly revenue across 2025
    df_2025 = df[df["quarter"].astype(str).str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    cust_2025_avg = (
        df_2025.groupby("Customer", observed=True)["revenue"].sum() / n_2025_q
    ).rename("_avg_2025_q")

    def _spend_tier(avg: float) -> str:
        if avg >= 10000:
            return "High Tier"
        if avg >= 5000:
            return "Medium Tier"
        return "Low Tier"

    spend_tier_map = cust_2025_avg.apply(_spend_tier).rename("spend_tier")
    df = df.join(spend_tier_map, on="Customer")
    df["spend_tier"] = df["spend_tier"].fillna("Low Tier").astype("category")

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


def _bold_last_row(df: pd.DataFrame):
    """Return a Styler with the last row bolded (for total rows)."""
    last_idx = df.index[-1]
    def row_style(row):
        weight = "font-weight: bold" if row.name == last_idx else ""
        return [weight] * len(row)
    return df.style.apply(row_style, axis=1)


def get_new_accounts_df(df: pd.DataFrame, current_q: str) -> pd.DataFrame:
    """
    Returns a DataFrame of genuinely new accounts in current_q.

    A new account qualifies only when BOTH conditions hold:
      1. The customer's very first invoice in the entire dataset falls in current_q.
      2. The current_sp on that first invoice is also the Salesperson on that invoice
         (i.e. the rep who owns the account today was the one who made the first sale).

    Returns one row per customer with columns: Customer, current_sp.
    """
    cq_df = df[df["quarter"] == current_q]
    ever_before = set(df[df["quarter"] < current_q]["Customer"].dropna().unique())
    new_custs = cq_df[~cq_df["Customer"].isin(ever_before)]

    if new_custs.empty:
        return pd.DataFrame(columns=["Customer", "current_sp"])

    # For each new customer, grab their first invoice in current_q
    first_inv = (
        new_custs.sort_values("invoice_date")
        .groupby("Customer", observed=True)[["current_sp", "Salesperson"]]
        .first()
        .reset_index()
    )

    # Only credit if current_sp == Salesperson on the first invoice
    credited = first_inv[
        first_inv["current_sp"].astype(str) == first_inv["Salesperson"].astype(str)
    ][["Customer", "current_sp"]]

    return credited


def compute_kpis(df: pd.DataFrame, current_q: str, prior_q: str | None) -> dict:
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    current_rev = cq_df["revenue"].sum()
    prior_rev = pq_df["revenue"].sum()
    growth_pct = (current_rev - prior_rev) / prior_rev * 100 if prior_rev > 0 else None

    new_accounts_df = get_new_accounts_df(df, current_q)
    new_accounts = set(new_accounts_df["Customer"].unique())
    new_account_count = len(new_accounts)

    df_2025 = df[df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    cust_2025_avg = df_2025.groupby("Customer")["revenue"].sum() / n_2025_q

    cq_per_cust = cq_df.groupby("Customer")["revenue"].sum()
    existing_cq = cq_per_cust[~cq_per_cust.index.isin(new_accounts)]
    grew_count = sum(1 for c, r in existing_cq.items() if r > cust_2025_avg.get(c, 0))
    declined_count = sum(1 for c, r in existing_cq.items() if r < cust_2025_avg.get(c, 0))

    cq_customers = set(cq_df["Customer"].dropna().unique())
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
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    cq_rev = cq_df.groupby("current_sp", observed=True)["revenue"].sum().rename("cq_revenue")
    pq_rev = (
        pq_df.groupby("current_sp", observed=True)["revenue"].sum().rename("pq_revenue")
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
        rep_line = cq_df[mask].groupby("current_sp", observed=True)["revenue"].sum()
        summary[f"{line}_rev"] = rep_line
        summary[f"{line}_rev"] = summary[f"{line}_rev"].fillna(0.0)
        summary[f"{line}_pct"] = (
            summary[f"{line}_rev"] / line_total * 100 if line_total > 0 else 0.0
        )

    # Attribution: revenue where the raw Salesperson column matches this rep
    if "Salesperson" in df.columns:
        attr_rev = (
            df[df["quarter"] == current_q]
            .groupby("Salesperson", observed=True)["revenue"].sum()
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

    # New accounts: customer first appeared this quarter AND current_sp was the first Salesperson
    new_acct_df = get_new_accounts_df(df, current_q)
    new_acct_count = (
        new_acct_df.groupby("current_sp", observed=True)["Customer"].nunique().rename("new_accounts")
    )
    summary = summary.join(new_acct_count, how="left")
    summary["new_accounts"] = summary["new_accounts"].fillna(0).astype(int)

    # Low spend growth: delta for Low Tier customers active in CQ (same universe both quarters)
    # PQ revenue is attributed to each customer's CQ salesperson so the comparison is consistent.
    cq_low_customers = set(
        cq_df[cq_df["spend_tier"] == "Low Tier"]["Customer"].dropna().unique()
    )
    cq_low = (
        cq_df[cq_df["Customer"].isin(cq_low_customers)]
        .groupby("current_sp", observed=True)["revenue"].sum().rename("cq_low_rev")
    )
    if not pq_df.empty:
        pq_cust_rev = (
            pq_df[pq_df["Customer"].isin(cq_low_customers)]
            .groupby("Customer", observed=True)["revenue"].sum().rename("pq_rev")
        )
        cq_low_sp = (
            cq_df[cq_df["Customer"].isin(cq_low_customers)][["Customer", "current_sp"]]
            .drop_duplicates("Customer")
            .set_index("Customer")
        )
        pq_low = (
            cq_low_sp.join(pq_cust_rev, how="left")
            .assign(pq_rev=lambda x: x["pq_rev"].fillna(0.0))
            .groupby("current_sp", observed=True)["pq_rev"].sum()
            .rename("pq_low_rev")
        )
    else:
        pq_low = pd.Series(dtype=float, name="pq_low_rev")
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
        .groupby("current_sp", observed=True)["Customer"].nunique()
        .rename("reactivated_count")
    )
    summary = summary.join(react_count, how="left")
    summary["reactivated_count"] = summary["reactivated_count"].fillna(0).astype(int)

    summary = summary.reset_index().rename(columns={"current_sp": "Salesperson"})
    return summary.sort_values("cq_revenue", ascending=False)


def compute_customer_table(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    cq_rev = cq_df.groupby("Customer", observed=True)["revenue"].sum().rename("cq_revenue")
    pq_rev = (
        pq_df.groupby("Customer", observed=True)["revenue"].sum().rename("pq_revenue")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_revenue")
    )

    df_2025 = df[df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    avg_2025 = (df_2025.groupby("Customer", observed=True)["revenue"].sum() / n_2025_q).rename("avg_2025_q")

    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier"]
    cq_meta = cq_df.sort_values("invoice_date").groupby("Customer", observed=True)[meta_cols].last()

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
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    cq_rev = cq_df.groupby("current_sp", observed=True)["revenue"].sum().rename("cq_revenue")

    pq_rev = (
        pq_df.groupby("current_sp", observed=True)["revenue"].sum().rename("pq_revenue")
        if not pq_df.empty else pd.Series(dtype=float, name="pq_revenue")
    )

    df_2025 = df[df["quarter"].str.startswith("2025")]
    n_2025_q = df_2025["quarter"].nunique() or 1
    avg_2025 = (df_2025.groupby("current_sp", observed=True)["revenue"].sum() / n_2025_q).rename("avg_2025_q")

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
    credited = get_new_accounts_df(df, current_q)
    if credited.empty:
        return pd.DataFrame(columns=["Customer", "current_sp", "spend_tier", "account_type", "customer_tier", "cq_revenue"])

    new_customers = set(credited["Customer"].unique())
    cq_df = df[df["quarter"] == current_q]
    cq_new = cq_df[cq_df["Customer"].isin(new_customers)]

    rev = cq_new.groupby("Customer", observed=True)["revenue"].sum().rename("cq_revenue")
    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier"]
    meta = cq_new.sort_values("invoice_date").groupby("Customer", observed=True)[meta_cols].last()
    table = rev.to_frame().join(meta).reset_index()
    return table.sort_values("cq_revenue", ascending=False)


def compute_reactivated_accounts_table(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    cq_df = df[df["quarter"] == current_q]
    pq_customers = (
        set(df[df["quarter"] == prior_q]["Customer"].dropna().unique())
        if prior_q else set()
    )
    pre_pq_cutoff = prior_q if prior_q else current_q
    pre_pq_customers = set(df[df["quarter"] < pre_pq_cutoff]["Customer"].dropna().unique())

    cq_customers = set(cq_df["Customer"].dropna().unique())
    reactivated = (cq_customers - pq_customers) & pre_pq_customers

    cq_react = cq_df[cq_df["Customer"].isin(reactivated)]
    rev = cq_react.groupby("Customer", observed=True)["revenue"].sum().rename("cq_revenue")
    meta_cols = ["current_sp", "spend_tier", "account_type", "customer_tier"]
    meta = cq_react.sort_values("invoice_date").groupby("Customer", observed=True)[meta_cols].last()

    last_active = (
        df[df["Customer"].isin(reactivated) & (df["quarter"] < current_q)]
        .groupby("Customer", observed=True)["quarter"].max()
        .rename("last_active_q")
    )

    table = rev.to_frame().join(meta).join(last_active).reset_index()
    return table.sort_values("cq_revenue", ascending=False)


WEIGHTS = {"new_accounts": 0.4, "low_spend_growth": 0.3, "reactivated_count": 0.3}


def compute_bonus_allocation(selected_reps: list, sp_metrics: pd.DataFrame, pool: float) -> pd.DataFrame:
    alloc = sp_metrics[sp_metrics["Salesperson"].isin(selected_reps)][
        ["Salesperson", "new_accounts", "low_spend_growth", "reactivated_count"]
    ].copy()

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


def compute_account_summary(df: pd.DataFrame, current_q: str, prior_q: str | None) -> pd.DataFrame:
    """Build a management summary grouped by Account Type > Customer Tier > Spend Tier.

    Uses groupby on the actual data so no rows are ever silently dropped.
    The grand total is computed directly from the full quarter slices to match KPIs exactly.
    """
    cq_df = df[df["quarter"] == current_q]
    pq_df = df[df["quarter"] == prior_q] if prior_q else pd.DataFrame(columns=df.columns)

    GROUP_COLS = ["account_type", "customer_tier", "spend_tier"]

    # Aggregate current quarter
    cq_agg = (
        cq_df.groupby(GROUP_COLS, observed=True)
        .agg(cq_count=("Customer", "nunique"), cq_rev=("revenue", "sum"))
        .reset_index()
    )

    # Aggregate prior quarter
    if not pq_df.empty:
        pq_agg = (
            pq_df.groupby(GROUP_COLS, observed=True)["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"revenue": "pq_rev"})
        )
    else:
        pq_agg = pd.DataFrame(columns=GROUP_COLS + ["pq_rev"])

    merged = cq_agg.merge(pq_agg, on=GROUP_COLS, how="left").fillna({"pq_rev": 0.0})

    # Sort order: Franchise first, then Individual Store;
    # within Franchise: Bronze → Silver → Gold → anything else;
    # within each group: Low Tier → Medium Tier → High Tier
    ACCT_ORDER = {"Franchise": 0, "Individual Store": 1}
    TIER_ORDER = {"Bronze": 0, "Silver": 1, "Gold": 2}
    SPEND_ORDER = {"Low Tier": 0, "Medium Tier": 1, "High Tier": 2}

    merged["_acct_sort"] = merged["account_type"].astype(str).map(ACCT_ORDER).fillna(99).astype(int)
    merged["_tier_sort"] = merged["customer_tier"].astype(str).map(TIER_ORDER).fillna(99).astype(int)
    merged["_spend_sort"] = merged["spend_tier"].astype(str).map(SPEND_ORDER).fillna(99).astype(int)
    merged = merged.sort_values(["_acct_sort", "_tier_sort", "_spend_sort"]).drop(
        columns=["_acct_sort", "_tier_sort", "_spend_sort"]
    )

    # Build display rows with label-suppression for repeated account/tier values
    rows = []
    prev_acct = prev_tier = None
    for _, r in merged.iterrows():
        acct = r["account_type"]
        tier = r["customer_tier"]
        growth = (r["cq_rev"] - r["pq_rev"]) / r["pq_rev"] * 100 if r["pq_rev"] > 0 else None
        rows.append({
            "Account Type": acct if acct != prev_acct else "",
            "Customer Tier": tier if (acct, tier) != (prev_acct, prev_tier) else "",
            "Spend Tier": r["spend_tier"],
            "Customer Count": int(r["cq_count"]),
            "_pq_rev": float(r["pq_rev"]),
            "_cq_rev": float(r["cq_rev"]),
            "_growth": growth,
        })
        prev_acct = acct
        prev_tier = tier

    # Grand total — computed directly from quarter slices so it matches KPIs exactly
    grand_cq = float(cq_df["revenue"].sum())
    grand_pq = float(pq_df["revenue"].sum()) if not pq_df.empty else 0.0
    grand_growth = (grand_cq - grand_pq) / grand_pq * 100 if grand_pq > 0 else None
    rows.append({
        "Account Type": "Total",
        "Customer Tier": "",
        "Spend Tier": "",
        "Customer Count": int(cq_df["Customer"].nunique()),
        "_pq_rev": grand_pq,
        "_cq_rev": grand_cq,
        "_growth": grand_growth,
    })

    return pd.DataFrame(rows)


def render_dashboard(df, current_q, prior_q, pq_label, cq_label):
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

    # ── Bonus Allocation (inline, right of KPIs) ───────────────────────────────
    st.subheader("Bonus allocation")
    st.caption(
        f"Weighted score: New Accounts {int(WEIGHTS['new_accounts']*100)}% · "
        f"Low Spend Growth {int(WEIGHTS['low_spend_growth']*100)}% · "
        f"Reactivated {int(WEIGHTS['reactivated_count']*100)}%"
    )

    sp_df = compute_sp_summary(df, current_q, prior_q)
    all_reps = sp_df["Salesperson"].tolist()

    bonus_col1, bonus_col2 = st.columns(2)
    with bonus_col1:
        selected_reps = st.multiselect(
            "Salespeople in bonus pool",
            options=all_reps,
            default=all_reps,
        )
    with bonus_col2:
        pool = st.number_input(
            "Total bonus pool ($)",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            format="%.2f",
        )

    if not selected_reps:
        st.warning("Select at least one salesperson.")
        return

    alloc_df = compute_bonus_allocation(selected_reps, sp_df, pool)

    a1, a2 = st.columns(2)
    a1.metric("Reps in pool", len(selected_reps))
    a2.metric("Total allocated", fmt(alloc_df["bonus"].sum()))

    bonus_display = alloc_df.copy()
    bonus_display["low_spend_growth"] = bonus_display["low_spend_growth"].apply(fmt)
    bonus_display["score"] = (bonus_display["score"] * 100).round(1).astype(str) + "%"
    bonus_display["share"] = (bonus_display["share"] * 100).round(1).astype(str) + "%"
    bonus_display["bonus"] = bonus_display["bonus"].apply(fmt)
    bonus_display = bonus_display.rename(columns={
        "new_accounts": "New Accounts",
        "low_spend_growth": "Low Spend Growth",
        "reactivated_count": "Reactivated",
        "score": "Weighted Score",
        "share": "Pool Share",
        "bonus": "Bonus",
    })
    bonus_total = pd.DataFrame([{
        "Salesperson": f"TOTAL ({len(alloc_df)} reps)",
        "New Accounts": int(alloc_df["new_accounts"].sum()),
        "Low Spend Growth": fmt(alloc_df["low_spend_growth"].sum()),
        "Reactivated": int(alloc_df["reactivated_count"].sum()),
        "Weighted Score": "—",
        "Pool Share": "100.0%",
        "Bonus": fmt(alloc_df["bonus"].sum()),
    }])
    st.dataframe(
        _bold_last_row(pd.concat([bonus_display, bonus_total], ignore_index=True)),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download allocation as CSV",
        data=alloc_df.to_csv(index=False).encode("utf-8"),
        file_name=f"bonus_allocation_{current_q}.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Salesperson Breakdown ──────────────────────────────────────────────────
    st.subheader("Salesperson breakdown")

    sp_display = sp_df[[
        "Salesperson", "pq_revenue", "cq_revenue", "growth_pct",
        "attribution_rev", "attribution_pct",
        "new_accounts", "low_spend_growth", "reactivated_count",
    ]].copy()

    sp_total_pq = sp_display["pq_revenue"].sum()
    sp_total_cq = sp_display["cq_revenue"].sum()
    sp_total_growth = (sp_total_cq - sp_total_pq) / sp_total_pq * 100 if sp_total_pq > 0 else None

    sp_display["pq_revenue"] = sp_display["pq_revenue"].apply(fmt)
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
        "pq_revenue": f"{pq_label} Revenue",
        "cq_revenue": f"{cq_label} Revenue",
        "growth_pct": "Growth",
        "attribution_rev": "Attribution ($)",
        "attribution_pct": "Attribution (%)",
        "new_accounts": "New Accounts",
        "low_spend_growth": "Low Spend Growth",
        "reactivated_count": "Reactivated",
    })

    sp_totals_row = pd.DataFrame([{
        "Salesperson": f"TOTAL ({len(sp_display)} reps)",
        f"{pq_label} Revenue": fmt(sp_total_pq),
        f"{cq_label} Revenue": fmt(sp_total_cq),
        "Growth": f"{sp_total_growth:+.1f}%" if sp_total_growth is not None else "—",
        "Attribution ($)": "—",
        "Attribution (%)": "—",
        "New Accounts": sp_df["new_accounts"].sum(),
        "Low Spend Growth": "—",
        "Reactivated": sp_df["reactivated_count"].sum(),
    }])

    st.dataframe(
        _bold_last_row(pd.concat([sp_display, sp_totals_row], ignore_index=True)),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Customer Detail ────────────────────────────────────────────────────────
    st.subheader("Customer detail")

    all_sp_options = sorted(df["current_sp"].dropna().unique().tolist())
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
    sp_growth_str = f"{sp_total_growth:+.1f}%" if sp_total_growth is not None else "—"
    sp_tbl_total = pd.DataFrame([{
        "Salesperson": f"TOTAL ({len(sp_tbl_display)} reps)",
        "2025 Avg/Quarter": "—",
        f"{pq_label} Revenue": fmt(sp_total_pq),
        f"{cq_label} Revenue": fmt(sp_total_cq),
        "Growth": sp_growth_str,
    }])
    st.dataframe(
        _bold_last_row(pd.concat([sp_tbl_display, sp_tbl_total], ignore_index=True)),
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
        new_total = pd.DataFrame([{
            "Customer": f"TOTAL ({len(new_display)} accounts)",
            "Salesperson": "—", "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
            f"{cq_label} Revenue": fmt(new_tbl["cq_revenue"].sum()),
        }])
        st.dataframe(
            _bold_last_row(pd.concat([new_display, new_total], ignore_index=True)),
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
        react_total = pd.DataFrame([{
            "Customer": f"TOTAL ({len(react_display)} accounts)",
            "Salesperson": "—", "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
            "Last Active Quarter": "—",
            f"{cq_label} Revenue": fmt(react_tbl["cq_revenue"].sum()),
        }])
        st.dataframe(
            _bold_last_row(pd.concat([react_display, react_total], ignore_index=True)),
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
    cust_growth_str = f"{total_growth:+.1f}%" if total_growth is not None else "—"
    cust_total = pd.DataFrame([{
        "Customer": f"TOTAL ({len(cust_display)} accounts)",
        "Salesperson": "—", "Spend Tier": "—", "Account Type": "—", "Customer Tier": "—",
        "2025 Avg/Quarter": "—",
        f"{pq_label} Revenue": fmt(total_pq),
        f"{cq_label} Revenue": fmt(total_cq),
        "Growth": cust_growth_str,
    }])
    st.dataframe(
        _bold_last_row(pd.concat([cust_display, cust_total], ignore_index=True)),
        use_container_width=True, hide_index=True,
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
            df = preprocess_dataset(raw_df)
    except Exception as e:
        st.error(f"Failed to process data: {e}")
        st.write("**Columns found in your CSV:**", list(raw_df.columns))
        return

    if df["invoice_date"].isna().all():
        st.error("No valid invoice dates found.")
        return

    quarter_list = sorted(df["quarter"].dropna().unique())
    current_q = quarter_list[-1]
    prior_quarters = [q for q in quarter_list if q < current_q]
    prior_q = prior_quarters[-1] if prior_quarters else None

    # Apply SP grouping: any SP with < $30k in current_q becomes "Other"
    df, other_sps = apply_sp_grouping(df, current_q)

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

    # ── Page tabs ──────────────────────────────────────────────────────────────
    tab_summary, tab_dash = st.tabs(["Account Summary", "Dashboard"])

    with tab_summary:
        st.subheader(f"Account Summary — {current_q}")
        st.caption(f"Revenue comparison: {pq_label} vs {cq_label}. Sidebar filters apply.")

        summary_df = compute_account_summary(df, current_q, prior_q)

        display = summary_df.copy()
        display[pq_label] = display["_pq_rev"].apply(fmt)
        display[cq_label] = display["_cq_rev"].apply(fmt)
        display["Growth"] = display["_growth"].apply(
            lambda v: f"{v:+.1f}%" if v is not None else "—"
        )
        display["Customer Count"] = display["Customer Count"].apply(
            lambda v: str(v) if v > 0 else "—"
        )
        display = display.drop(columns=["_pq_rev", "_cq_rev", "_growth"])
        st.dataframe(_bold_last_row(display), use_container_width=True, hide_index=True)

        # Download
        export = summary_df[["Account Type", "Customer Tier", "Spend Tier", "Customer Count"]].copy()
        export[pq_label] = summary_df["_pq_rev"]
        export[cq_label] = summary_df["_cq_rev"]
        export["Growth %"] = summary_df["_growth"].apply(
            lambda v: round(v, 1) if v is not None else None
        )
        st.download_button(
            "Download summary as CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name=f"account_summary_{current_q}.csv",
            mime="text/csv",
        )

        # ── "Other" SP disclosure ──────────────────────────────────────────────
        st.markdown("---")
        with st.expander(f"Salespeople grouped as 'Other' ({len(other_sps)}) — under ${OTHER_SP_THRESHOLD:,} in {current_q}"):
            if other_sps:
                st.write(", ".join(other_sps))
            else:
                st.caption("No salespeople fell below the threshold.")

    with tab_dash:
        render_dashboard(df, current_q, prior_q, pq_label, cq_label)


if __name__ == "__main__":
    main()
