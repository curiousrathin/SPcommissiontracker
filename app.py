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

st.set_page_config(page_title="Smoke Arsenal Incentive Tool", layout="wide")
st.title("Smoke Arsenal — Incentive Allocation")


REQUIRED_COLUMNS = {
    "Customer", "Invoice Date", "Untaxed Total",
    "Current Salesperson", "Salesperson", "Product",
    "Customer Tier", "Account Type",
}
OPTIONAL_COLUMNS = {"Franchise", "Delivery Address", "Zip", "Brand"}


@st.cache_data(show_spinner=False)
def load_csv_data(source):
    all_cols = REQUIRED_COLUMNS | OPTIONAL_COLUMNS
    raw = pd.read_csv(source, dtype=str, usecols=lambda c: c.strip("\ufeff") in all_cols)
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
    # (matches get_new_accounts_df — same rule used by KPIs and bonus allocation)
    new_acct_customers = set(get_new_accounts_df(df, current_q)["Customer"].unique())
    master["is_new"] = master["Customer"].isin(new_acct_customers)

    # Reactivated: not in PQ, seen before PQ, AND placed >= 2 orders or ordered in >= 2 months in CQ
    pq_customers = set(pq_df["Customer"].dropna().unique()) if not pq_df.empty else set()
    pre_pq_q = prior_q if prior_q else current_q
    pre_pq_customers = set(df[df["quarter"] < pre_pq_q]["Customer"].dropna().unique())
    cq_activity = cq_df.groupby("Customer", observed=True).agg(
        _order_count=("invoice_date", "count"),
        _month_count=("invoice_date", lambda x: x.dt.month.nunique()),
    )
    cq_min_activity = set(
        cq_activity[
            (cq_activity["_order_count"] >= 2) | (cq_activity["_month_count"] >= 2)
        ].index.tolist()
    )
    master["is_reactivated"] = (
        ~master["Customer"].isin(pq_customers)
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
    _rate_map = {"Low Tier": 0.02, "Medium Tier": 0.01, "High Tier": 0.01}
    _rates = master["spend_tier"].astype(str).map(_rate_map).fillna(0.01)
    master["growth_commission"] = (master["growth_amount"] * _rates).round(2)

    # Flat incentives
    master["new_incentive"] = np.where(master["is_new"], 50.0, 0.0)
    master["reactivated_incentive"] = np.where(master["is_reactivated"], 100.0, 0.0)

    return master.sort_values("cq_revenue", ascending=False)


def render_data_lab(df: pd.DataFrame, current_q: str, prior_q: Optional[str], pq_label: str, cq_label: str, key_prefix: str = "lab", show_excluded_col: bool = False) -> None:
    st.subheader("Data Lab — Master Table Verification")
    st.caption(
        "Single source of truth: one row per customer, all facts resolved the same way. "
        "Use this to cross-check numbers against the other tabs."
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

    # ── SP rollup ─────────────────────────────────────────────────────────────
    st.markdown("**By salesperson**")
    sp_rollup = (
        filtered.groupby("current_sp", observed=True)
        .agg(
            cq_revenue=("cq_revenue", "sum"),
            pq_revenue=("pq_revenue", "sum"),
            low_spend_growth=("cq_revenue", lambda x: (
                x.values.sum() - filtered.loc[
                    filtered["current_sp"] == x.name, "pq_revenue"
                ].sum()
                if filtered.loc[filtered["current_sp"] == x.name, "spend_tier"].eq("Low Tier").any()
                else 0.0
            )),
            new_accounts=("is_new", "sum"),
            reactivated=("is_reactivated", "sum"),
            customer_count=("Customer", "nunique"),
        )
        .reset_index()
        .rename(columns={"current_sp": "Salesperson"})
        .sort_values("cq_revenue", ascending=False)
    )

    # Recompute low_spend_growth properly from filtered master
    low_by_sp = (
        filtered[filtered["spend_tier"] == "Low Tier"]
        .groupby("current_sp", observed=True)
        .apply(lambda g: (g["cq_revenue"] - g["pq_revenue"]).sum(), include_groups=False)
        .rename("low_spend_growth")
        .reset_index()
        .rename(columns={"current_sp": "Salesperson"})
    )
    sp_rollup = sp_rollup.drop(columns=["low_spend_growth"]).merge(low_by_sp, on="Salesperson", how="left")
    sp_rollup["low_spend_growth"] = sp_rollup["low_spend_growth"].fillna(0.0)

    sp_rollup["growth_pct"] = np.where(
        sp_rollup["pq_revenue"] > 0,
        (sp_rollup["cq_revenue"] - sp_rollup["pq_revenue"]) / sp_rollup["pq_revenue"] * 100,
        None,
    )

    sp_display = sp_rollup.copy()
    sp_total_cq = sp_display["cq_revenue"].sum()
    sp_total_pq = sp_display["pq_revenue"].sum()
    sp_display[cq_label] = sp_display["cq_revenue"].apply(fmt)
    sp_display[pq_label] = sp_display["pq_revenue"].apply(fmt)
    sp_display["Growth"] = sp_display["growth_pct"].apply(lambda v: f"{v:+.1f}%" if v is not None else "—")
    sp_display["Low Spend Growth"] = sp_display["low_spend_growth"].apply(
        lambda v: f"+{fmt(v)}" if v >= 0 else f"-${abs(v):,.0f}"
    )
    sp_display["New Accounts"] = sp_display["new_accounts"].astype(int)
    sp_display["Reactivated"] = sp_display["reactivated"].astype(int)
    sp_display["Customers"] = sp_display["customer_count"].astype(int)
    sp_display = sp_display[["Salesperson", pq_label, cq_label, "Growth", "Low Spend Growth", "New Accounts", "Reactivated", "Customers"]]

    total_growth = (sp_total_cq - sp_total_pq) / sp_total_pq * 100 if sp_total_pq > 0 else None
    total_low_growth = low_by_sp["low_spend_growth"].sum() if not low_by_sp.empty else 0.0
    sp_total_row = pd.DataFrame([{
        "Salesperson": f"TOTAL ({len(sp_display)} reps)",
        pq_label: fmt(sp_total_pq),
        cq_label: fmt(sp_total_cq),
        "Growth": f"{total_growth:+.1f}%" if total_growth is not None else "—",
        "Low Spend Growth": f"+{fmt(total_low_growth)}" if total_low_growth >= 0 else f"-${abs(total_low_growth):,.0f}",
        "New Accounts": int(filtered["is_new"].sum()),
        "Reactivated": int(filtered["is_reactivated"].sum()),
        "Customers": int(filtered["Customer"].nunique()),
    }])
    st.dataframe(
        _bold_last_row(pd.concat([sp_display, sp_total_row], ignore_index=True)),
        use_container_width=True, hide_index=True,
    )

    # ── Low spend growth breakdown ────────────────────────────────────────────
    st.markdown("**Low spend growth** *(click to expand)*")
    low_df = filtered[filtered["spend_tier"] == "Low Tier"]
    for acct_type in ["Franchise", "Individual Store"]:
        acct_low = low_df[low_df["account_type"] == acct_type]
        if acct_low.empty:
            continue
        total_growth = float((acct_low["cq_revenue"] - acct_low["pq_revenue"]).sum())
        n = acct_low["Customer"].nunique()
        growth_str = f"+{fmt(total_growth)}" if total_growth >= 0 else f"-${abs(total_growth):,.0f}"
        header = f"{acct_type}  —  {growth_str}  |  {n} stores"

        with st.expander(header):
            if acct_type == "Franchise":
                # Level 1 — franchise group
                fg_order = (
                    acct_low.groupby("franchise", observed=True)
                    .apply(lambda g: (g["cq_revenue"] - g["pq_revenue"]).sum(), include_groups=False)
                    .sort_values(ascending=False).index.tolist()
                )
                for fg_val in fg_order:
                    fg_data = acct_low[acct_low["franchise"] == fg_val]
                    fg_label = fg_val if fg_val else "Unassigned"
                    fg_growth = float((fg_data["cq_revenue"] - fg_data["pq_revenue"]).sum())
                    fg_growth_str = f"+{fmt(fg_growth)}" if fg_growth >= 0 else f"-${abs(fg_growth):,.0f}"
                    fg_n = fg_data["Customer"].nunique()

                    with st.expander(f"{fg_label}  —  {fg_growth_str}  |  {fg_n} stores"):
                        # Level 2 — individual customers
                        cust = fg_data[["Customer", "avg_2025_q", "pq_revenue", "cq_revenue", "growth_pct"]].copy()
                        cust = cust.sort_values("cq_revenue", ascending=False)
                        cust["2025 Avg/Q"] = cust["avg_2025_q"].apply(fmt)
                        cust[cq_label] = cust["cq_revenue"].apply(fmt)
                        cust[pq_label] = cust["pq_revenue"].apply(fmt)
                        cust["Growth"] = cust["growth_pct"].apply(lambda v: f"{v:+.1f}%" if v is not None else "—")
                        cust["Low Spend Δ"] = (fg_data["cq_revenue"] - fg_data["pq_revenue"]).apply(
                            lambda v: f"+{fmt(v)}" if v >= 0 else f"-${abs(v):,.0f}"
                        )
                        cust_total = pd.DataFrame([{
                            "Customer": f"TOTAL ({fg_n} stores)",
                            "2025 Avg/Q": "—",
                            pq_label: fmt(fg_data["pq_revenue"].sum()),
                            cq_label: fmt(fg_data["cq_revenue"].sum()),
                            "Growth": "—",
                            "Low Spend Δ": fg_growth_str,
                        }])
                        display_cols = ["Customer", "2025 Avg/Q", pq_label, cq_label, "Growth", "Low Spend Δ"]
                        st.dataframe(
                            _bold_last_row(pd.concat([cust[display_cols], cust_total], ignore_index=True)),
                            use_container_width=True, hide_index=True,
                        )
            else:
                # Individual stores — straight to customer table
                cust = acct_low[["Customer", "avg_2025_q", "pq_revenue", "cq_revenue", "growth_pct"]].copy()
                cust = cust.sort_values("cq_revenue", ascending=False)
                cust["2025 Avg/Q"] = cust["avg_2025_q"].apply(fmt)
                cust[cq_label] = cust["cq_revenue"].apply(fmt)
                cust[pq_label] = cust["pq_revenue"].apply(fmt)
                cust["Growth"] = cust["growth_pct"].apply(lambda v: f"{v:+.1f}%" if v is not None else "—")
                cust["Low Spend Δ"] = (acct_low["cq_revenue"] - acct_low["pq_revenue"]).apply(
                    lambda v: f"+{fmt(v)}" if v >= 0 else f"-${abs(v):,.0f}"
                )
                cust_total = pd.DataFrame([{
                    "Customer": f"TOTAL ({n} stores)",
                    "2025 Avg/Q": "—",
                    pq_label: fmt(acct_low["pq_revenue"].sum()),
                    cq_label: fmt(acct_low["cq_revenue"].sum()),
                    "Growth": "—",
                    "Low Spend Δ": growth_str,
                }])
                display_cols = ["Customer", "2025 Avg/Q", pq_label, cq_label, "Growth", "Low Spend Δ"]
                st.dataframe(
                    _bold_last_row(pd.concat([cust[display_cols], cust_total], ignore_index=True)),
                    use_container_width=True, hide_index=True,
                )

    # ── Account type rollup with drill-in ────────────────────────────────────
    st.markdown("**By account type** *(click to expand)*")
    with st.expander("Tier definitions"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "**Customer Tier** *(store count per franchise)*\n\n"
                "| Tier | Stores |\n|---|---|\n"
                "| Bronze | 2 – 4 |\n"
                "| Silver | 5 – 9 |\n"
                "| Gold | 10+ |"
            )
        with c2:
            st.markdown(
                "**Spend Tier** *(2025 avg quarterly revenue)*\n\n"
                "| Tier | Revenue |\n|---|---|\n"
                "| High | > $10,000 |\n"
                "| Medium | $5,000 – $10,000 |\n"
                "| Low | < $5,000 |\n\n"
                "*Franchise accounts are classified by their combined group spend, not per store.*"
            )
    for acct_type in ["Franchise", "Individual Store"]:
        subset = filtered[filtered["account_type"] == acct_type]
        if subset.empty:
            continue

        grp_cq = float(subset["cq_revenue"].sum())
        grp_pq = float(subset["pq_revenue"].sum())
        grp_growth = (grp_cq - grp_pq) / grp_pq * 100 if grp_pq > 0 else None
        grp_growth_str = f"{grp_growth:+.1f}%" if grp_growth is not None else "—"
        n_cust = subset["Customer"].nunique()

        header = f"{acct_type}  —  {fmt(grp_cq)}  |  {grp_growth_str}  |  {n_cust} customers"
        with st.expander(header):
            if acct_type == "Franchise":
                # Level 1 — one expander per customer tier
                for ct in ["Bronze", "Silver", "Gold", ""]:
                    ct_data = subset[subset["customer_tier"] == ct]
                    if ct_data.empty:
                        continue
                    ct_label = ct if ct else "Untiered"
                    ct_cq = float(ct_data["cq_revenue"].sum())
                    ct_pq = float(ct_data["pq_revenue"].sum())
                    ct_growth = (ct_cq - ct_pq) / ct_pq * 100 if ct_pq > 0 else None
                    ct_growth_str = f"{ct_growth:+.1f}%" if ct_growth is not None else "—"
                    ct_n = ct_data["Customer"].nunique()
                    ct_header = f"{ct_label}  —  {fmt(ct_cq)}  |  {ct_growth_str}  |  {ct_n} customers"

                    with st.expander(ct_header):
                        # Level 2 — one expander per spend tier
                        _tier_sort = {"High Tier": 0, "Medium Tier": 1, "Low Tier": 2}
                        spend_tiers = sorted(
                            ct_data["spend_tier"].dropna().unique().tolist(),
                            key=lambda x: _tier_sort.get(x, 99),
                        )
                        for st_val in spend_tiers:
                            st_data = ct_data[ct_data["spend_tier"] == st_val]
                            if st_data.empty:
                                continue
                            st_cq = float(st_data["cq_revenue"].sum())
                            st_pq = float(st_data["pq_revenue"].sum())
                            st_growth = (st_cq - st_pq) / st_pq * 100 if st_pq > 0 else None
                            st_growth_str = f"{st_growth:+.1f}%" if st_growth is not None else "—"
                            st_n = st_data["Customer"].nunique()
                            st_header = f"{st_val}  —  {fmt(st_cq)}  |  {st_growth_str}  |  {st_n} customers"

                            with st.expander(st_header):
                                # Level 3 — one expander per franchise group
                                fg_order = (
                                    st_data.groupby("franchise", observed=True)["cq_revenue"]
                                    .sum().sort_values(ascending=False).index.tolist()
                                )
                                for fg_val in fg_order:
                                    fg_data = st_data[st_data["franchise"] == fg_val]
                                    fg_label = fg_val if fg_val else "Unassigned"
                                    fg_cq = float(fg_data["cq_revenue"].sum())
                                    fg_pq = float(fg_data["pq_revenue"].sum())
                                    fg_growth = (fg_cq - fg_pq) / fg_pq * 100 if fg_pq > 0 else None
                                    fg_growth_str = f"{fg_growth:+.1f}%" if fg_growth is not None else "—"
                                    fg_n = fg_data["Customer"].nunique()
                                    fg_header = f"{fg_label}  —  {fmt(fg_cq)}  |  {fg_growth_str}  |  {fg_n} customers"

                                    with st.expander(fg_header):
                                        # Level 4 — individual customers
                                        cust = fg_data[["Customer", "avg_2025_q", "pq_revenue", "cq_revenue", "growth_pct"]].copy()
                                        cust = cust.sort_values("cq_revenue", ascending=False)
                                        cust["2025 Avg/Q"] = cust["avg_2025_q"].apply(fmt)
                                        cust[cq_label] = cust["cq_revenue"].apply(fmt)
                                        cust[pq_label] = cust["pq_revenue"].apply(fmt)
                                        cust["Growth"] = cust["growth_pct"].apply(lambda v: f"{v:+.1f}%" if v is not None else "—")
                                        cust_total = pd.DataFrame([{
                                            "Customer": f"TOTAL ({len(cust)} stores)",
                                            "2025 Avg/Q": fmt(fg_data["avg_2025_q"].sum()),
                                            pq_label: fmt(fg_pq),
                                            cq_label: fmt(fg_cq),
                                            "Growth": fg_growth_str,
                                        }])
                                        display_cols = ["Customer", "2025 Avg/Q", pq_label, cq_label, "Growth"]
                                        st.dataframe(
                                            _bold_last_row(pd.concat([cust[display_cols], cust_total], ignore_index=True)),
                                            use_container_width=True, hide_index=True,
                                        )
                # Skip the generic st.dataframe at the bottom for Franchise
                continue
            else:
                # Level 1 — one expander per spend tier
                _tier_sort = {"High Tier": 0, "Medium Tier": 1, "Low Tier": 2}
                spend_tiers = sorted(
                    subset["spend_tier"].dropna().unique().tolist(),
                    key=lambda x: _tier_sort.get(x, 99),
                )
                for st_val in spend_tiers:
                    st_data = subset[subset["spend_tier"] == st_val]
                    if st_data.empty:
                        continue
                    st_cq = float(st_data["cq_revenue"].sum())
                    st_pq = float(st_data["pq_revenue"].sum())
                    st_growth = (st_cq - st_pq) / st_pq * 100 if st_pq > 0 else None
                    st_growth_str = f"{st_growth:+.1f}%" if st_growth is not None else "—"
                    st_n = st_data["Customer"].nunique()
                    st_header = f"{st_val}  —  {fmt(st_cq)}  |  {st_growth_str}  |  {st_n} stores"

                    with st.expander(st_header):
                        # Level 2 — individual customers
                        cust = st_data[["Customer", "avg_2025_q", "pq_revenue", "cq_revenue", "growth_pct"]].copy()
                        cust = cust.sort_values("cq_revenue", ascending=False)
                        cust["2025 Avg/Q"] = cust["avg_2025_q"].apply(fmt)
                        cust[cq_label] = cust["cq_revenue"].apply(fmt)
                        cust[pq_label] = cust["pq_revenue"].apply(fmt)
                        cust["Growth"] = cust["growth_pct"].apply(lambda v: f"{v:+.1f}%" if v is not None else "—")
                        cust_total = pd.DataFrame([{
                            "Customer": f"TOTAL ({st_n} stores)",
                            "2025 Avg/Q": fmt(st_data["avg_2025_q"].sum()),
                            pq_label: fmt(st_pq),
                            cq_label: fmt(st_cq),
                            "Growth": st_growth_str,
                        }])
                        display_cols = ["Customer", "2025 Avg/Q", pq_label, cq_label, "Growth"]
                        st.dataframe(
                            _bold_last_row(pd.concat([cust[display_cols], cust_total], ignore_index=True)),
                            use_container_width=True, hide_index=True,
                        )

    # ── Full customer list ────────────────────────────────────────────────────
    st.markdown("**All customers (raw master table)**")
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

    # ── Bonus Allocation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Bonus allocation")
    st.caption(
        f"Weighted score: New Accounts {int(WEIGHTS['new_accounts']*100)}% · "
        f"Low Spend Growth {int(WEIGHTS['low_spend_growth']*100)}% · "
        f"Reactivated {int(WEIGHTS['reactivated_count']*100)}%"
    )
    st.caption("All figures derived from the master table — consistent with every metric shown above.")

    all_reps = sp_rollup["Salesperson"].tolist()
    bonus_col1, bonus_col2 = st.columns(2)
    with bonus_col1:
        selected_reps = st.multiselect(
            "Salespeople in bonus pool",
            options=all_reps,
            default=all_reps,
            key=f"{key_prefix}_bonus_reps",
        )
    with bonus_col2:
        pool = st.number_input(
            "Total bonus pool ($)",
            min_value=0.0,
            value=2000.0,
            step=100.0,
            format="%.2f",
            key=f"{key_prefix}_bonus_pool",
        )

    if not selected_reps:
        st.warning("Select at least one salesperson.")
    else:
        # rename 'reactivated' → 'reactivated_count' to match compute_bonus_allocation's expected schema
        sp_for_bonus = sp_rollup.rename(columns={"reactivated": "reactivated_count"})
        alloc_df = compute_bonus_allocation(selected_reps, sp_for_bonus, pool)

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

    this_q = f"{pd.Period.now('Q').year}Q{pd.Period.now('Q').quarter}"
    quarter_list = sorted(q for q in df["quarter"].dropna().unique() if q < this_q)
    if not quarter_list:
        st.error("No completed quarters found in the dataset.")
        return
    current_q = quarter_list[-1]
    prior_quarters = [q for q in quarter_list if q < current_q]
    prior_q = prior_quarters[-1] if prior_quarters else None

    # Apply SP grouping: any SP with < $30k in current_q becomes "Other"
    df, _ = apply_sp_grouping(df, current_q)
    df_full, _ = apply_sp_grouping(df_full, current_q)

    pq_label = prior_q or "Prior Q"
    cq_label = current_q

    tab_lab, tab_full = st.tabs(["Data Lab", "Full Dataset (incl. excluded customers)"])

    with tab_lab:
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

        render_data_lab(df_filtered, current_q, prior_q, pq_label, cq_label, key_prefix="lab")

    with tab_full:
        st.caption(
            "Unfiltered view — includes all customers, including internal Smoke Arsenal rows "
            "excluded from the Data Lab. Use this to verify the full dataset row counts and totals."
        )
        render_data_lab(df_full, current_q, prior_q, pq_label, cq_label, key_prefix="full", show_excluded_col=True)


if __name__ == "__main__":
    main()
