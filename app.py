import io
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

DEFAULT_FILE = Path("FrankieDS.xlsx")
DEFAULT_SHEET = "Sheet 1"
VALID_PAYMENT_STATUS = ["Paid", "In Payment", "Partially Paid"]
SMALL_REP_THRESHOLD = 30000
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

st.set_page_config(
    page_title="Smoke Arsenal Incentive Tool",
    layout="wide",
)

st.title("Smoke Arsenal — Incentive Allocation Starter")
st.markdown(
    """
    This app loads `FrankieDS.xlsx`, applies the first load-time filters, and provides a data preview.
    Use the sidebar to swap files and explore the cleaned dataset.
    """
)


@st.cache_data(show_spinner=False)
def load_excel_data(source):
    return pd.read_excel(source, sheet_name=DEFAULT_SHEET, engine="openpyxl")


@st.cache_data(show_spinner=False)
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Step 1 — Smoke Arsenal internal customer exclusion
    df = df[~df["Customer"].astype(str).str.contains("smoke arsenal", case=False, na=False)]

    # Step 2 — Date conversion (handles Excel serial int and date strings)
    invoice_values = df["Invoice Date"]
    if pd.api.types.is_numeric_dtype(invoice_values):
        df["invoice_date"] = pd.to_datetime(
            invoice_values, origin="1899-12-30", unit="D", errors="coerce"
        )
    else:
        df["invoice_date"] = pd.to_datetime(invoice_values, errors="coerce")

    # Step 3 — Rename key fields
    df = df.rename(columns={
        "Untaxed Total": "revenue",
        "Current Salesperson": "current_sp",
        "Sales Team": "sales_team",
        "Customer Tier": "customer_tier",
        "Account Type": "account_type",
        "Product Category": "product_category",
        "Qty Ordered": "qty_ordered",
        "Product": "product",
        "Channel": "channel",
    })

    # Step 4 — Spending tier (computed from full dataset before any period filtering)
    # Uses average quarterly spend per customer across all history in the file.
    df["quarter_period"] = df["invoice_date"].dt.to_period("Q")
    cust_q = (
        df.groupby(["Customer", "quarter_period"])["revenue"]
        .sum()
        .reset_index()
    )
    cust_avg = (
        cust_q.groupby("Customer")["revenue"]
        .mean()
        .reset_index()
        .rename(columns={"revenue": "avg_quarterly_spend"})
    )

    def _spending_tier(v: float) -> str:
        if v > 20000:
            return "High"
        if v >= 5000:
            return "Medium"
        return "Low"

    cust_avg["spending_tier"] = cust_avg["avg_quarterly_spend"].apply(_spending_tier)
    df = df.merge(cust_avg[["Customer", "spending_tier"]], on="Customer", how="left")
    df["spending_tier"] = df["spending_tier"].fillna("Low")

    return df


def format_currency(value):
    return f"${value:,.2f}"


def get_quarter_label(series: pd.Series):
    return series.dt.to_period("Q").astype(str)


def aggregate_current_sp_revenue(df: pd.DataFrame, threshold: float = SMALL_REP_THRESHOLD) -> pd.DataFrame:
    totals = (
        df.groupby("current_sp", dropna=False)["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"current_sp": "Current Salesperson"})
    )
    totals["Current Salesperson"] = totals["Current Salesperson"].fillna("Unassigned")
    small_mask = totals["revenue"] < threshold
    if small_mask.any():
        other_total = totals.loc[small_mask, "revenue"].sum()
        totals = totals.loc[~small_mask].copy()
        totals = pd.concat(
            [
                totals,
                pd.DataFrame(
                    [{"Current Salesperson": "Other", "revenue": other_total}]
                ),
            ],
            ignore_index=True,
        )
    return totals.sort_values("revenue", ascending=False)


def parse_key_line_keywords(raw_text: str) -> list[str]:
    return [term.strip() for term in str(raw_text).split(",") if term.strip()]


def flag_key_line_products(series: pd.Series, keywords: list[str]) -> pd.Series:
    if not keywords:
        return pd.Series(False, index=series.index)

    mask = pd.Series(False, index=series.index)
    values = series.fillna("").astype(str)
    for keyword in keywords:
        mask |= values.str.contains(keyword, case=False, na=False)
    return mask


def compute_rep_performance(
    df: pd.DataFrame,
    current_quarter: str,
    prior_quarter: str | None,
    key_line_keywords: list[str],
) -> pd.DataFrame:
    quarter_df = df[df["quarter"] == current_quarter].copy()
    prior_df = df[df["quarter"] == prior_quarter].copy() if prior_quarter else pd.DataFrame(columns=df.columns)

    prior_revenue_by_customer = prior_df.groupby("Customer")["revenue"].sum()
    quarter_df["is_new_customer"] = ~quarter_df["Customer"].isin(prior_revenue_by_customer.index)
    quarter_df["key_line_revenue"] = quarter_df["revenue"].where(
        flag_key_line_products(quarter_df["product"], key_line_keywords), 0.0
    )

    # Use most-recent metadata per customer in CQ for tier/channel
    cust_meta = (
        quarter_df.sort_values("invoice_date")
        .groupby("Customer")[["spending_tier", "channel", "customer_tier"]]
        .last()
    )

    customer_summary = (
        quarter_df.groupby(["current_sp", "Customer"], dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            key_line_revenue=("key_line_revenue", "sum"),
            is_new_customer=("is_new_customer", "any"),
        )
        .reset_index()
    )
    customer_summary = customer_summary.join(cust_meta, on="Customer")
    customer_summary["prior_revenue"] = customer_summary["Customer"].map(prior_revenue_by_customer).fillna(0.0)
    customer_summary["new_customer_revenue"] = customer_summary["revenue"].where(customer_summary["is_new_customer"], 0.0)
    customer_summary["existing_customer_revenue"] = customer_summary["revenue"].where(~customer_summary["is_new_customer"], 0.0)
    customer_summary["raw_growth"] = (customer_summary["revenue"] - customer_summary["prior_revenue"]).clip(lower=0)

    # Emphasis multipliers (Section 3.3 of plan) — stacked multiplicatively
    customer_summary["growth_mult"] = 1.0
    customer_summary.loc[customer_summary["spending_tier"] == "Low", "growth_mult"] *= 1.5
    customer_summary.loc[customer_summary["channel"] == "C&G", "growth_mult"] *= 1.3
    customer_summary.loc[customer_summary["channel"] == "Hybrid Vape Store", "growth_mult"] *= 1.3
    customer_summary.loc[customer_summary["customer_tier"] == "Bronze", "growth_mult"] *= 1.3
    customer_summary["existing_growth"] = customer_summary["raw_growth"] * customer_summary["growth_mult"]

    rep_summary = (
        customer_summary.groupby("current_sp", dropna=False)
        .agg(
            revenue=("revenue", "sum"),
            coverage=("Customer", "nunique"),
            new_customer_count=("is_new_customer", "sum"),
            new_customer_revenue=("new_customer_revenue", "sum"),
            existing_customer_revenue=("existing_customer_revenue", "sum"),
            key_line_revenue=("key_line_revenue", "sum"),
            existing_growth=("existing_growth", "sum"),
            low_spend_customers=(
                "spending_tier",
                lambda s: (s == "Low").sum(),
            ),
            medium_spend_customers=(
                "spending_tier",
                lambda s: (s == "Medium").sum(),
            ),
            high_spend_customers=(
                "spending_tier",
                lambda s: (s == "High").sum(),
            ),
        )
        .reset_index()
        .rename(columns={"current_sp": "Current Salesperson"})
    )
    rep_summary["Current Salesperson"] = rep_summary["Current Salesperson"].fillna("Unassigned")
    return rep_summary.sort_values("revenue", ascending=False)


def compute_rep_details(
    df: pd.DataFrame,
    current_quarter: str,
    prior_quarter: str | None,
    key_line_keywords: list[str],
) -> dict:
    """Return a dict keyed by rep name with account-level detail for the explanation card."""
    quarter_df = df[df["quarter"] == current_quarter].copy()
    prior_df = df[df["quarter"] == prior_quarter].copy() if prior_quarter else pd.DataFrame(columns=df.columns)

    # Customers who bought under each rep last quarter
    prior_customers_by_rep = (
        prior_df.groupby("current_sp")["Customer"]
        .apply(set)
        .to_dict()
    ) if not prior_df.empty else {}

    # All customers who bought from anyone last quarter (used for "new to company" check)
    prior_all_customers = set(prior_df["Customer"].dropna().unique()) if not prior_df.empty else set()

    prior_revenue_by_customer = prior_df.groupby("Customer")["revenue"].sum() if not prior_df.empty else pd.Series(dtype=float)

    quarter_df["is_key_line"] = flag_key_line_products(quarter_df["product"], key_line_keywords)

    details = {}
    for rep, rep_df in quarter_df.groupby("current_sp", dropna=False):
        rep_name = rep if pd.notna(rep) else "Unassigned"
        current_customers = set(rep_df["Customer"].dropna().unique())
        prior_rep_customers = prior_customers_by_rep.get(rep, set())

        # New accounts: in this quarter, not anywhere in prior quarter
        new_accounts = []
        existing_accounts = []
        for customer, cust_df in rep_df.groupby("Customer", dropna=False):
            rev = cust_df["revenue"].sum()
            kl_rev = cust_df.loc[cust_df["is_key_line"], "revenue"].sum()
            prior_rev = prior_revenue_by_customer.get(customer, 0.0)
            tier = cust_df["spending_tier"].iloc[-1] if "spending_tier" in cust_df.columns else "Low"
            ch = cust_df["channel"].iloc[-1] if "channel" in cust_df.columns else None
            if customer not in prior_all_customers:
                new_accounts.append({
                    "Customer": customer,
                    "Spend tier": tier,
                    "Revenue": rev,
                    "Key-line revenue": kl_rev,
                })
            else:
                raw_growth = max(rev - prior_rev, 0.0)
                mult = 1.0
                if tier == "Low":
                    mult *= 1.5
                if ch == "C&G":
                    mult *= 1.3
                if ch == "Hybrid Vape Store":
                    mult *= 1.3
                existing_accounts.append({
                    "Customer": customer,
                    "Spend tier": tier,
                    "Channel": ch,
                    "Revenue": rev,
                    "Prior revenue": prior_rev,
                    "Raw growth": raw_growth,
                    "Emphasis mult": mult,
                    "Adj growth": raw_growth * mult,
                    "Key-line revenue": kl_rev,
                })

        # Inactive accounts: were under this rep last quarter, absent this quarter
        inactive_accounts = []
        for customer in prior_rep_customers - current_customers:
            prior_rev = prior_revenue_by_customer.get(customer, 0.0)
            inactive_accounts.append({"Customer": customer, "Prior revenue": prior_rev})

        # Key-line line items for this rep
        kl_df = rep_df[rep_df["is_key_line"]].copy()
        key_line_items = []
        if not kl_df.empty:
            for product, prod_df in kl_df.groupby("product", dropna=False):
                key_line_items.append({"Product": product, "Revenue": prod_df["revenue"].sum()})

        details[rep_name] = {
            "new_accounts": sorted(new_accounts, key=lambda r: r["Revenue"], reverse=True),
            "existing_accounts": sorted(existing_accounts, key=lambda r: r["Revenue"], reverse=True),
            "inactive_accounts": sorted(inactive_accounts, key=lambda r: r["Prior revenue"], reverse=True),
            "key_line_items": sorted(key_line_items, key=lambda r: r["Revenue"], reverse=True),
        }
    return details


def assign_bonus_tier(revenue: float, tier_1_min: float, tier_2_min: float) -> str:
    if revenue >= tier_1_min:
        return "Tier 1"
    if revenue >= tier_2_min:
        return "Tier 2"
    return "Tier 3"


def get_tier_weight(tier_name: str, tier_weights: dict) -> float:
    return tier_weights.get(tier_name, 1.0)


def compute_bonus_allocation(rep_df: pd.DataFrame, pool: float, tier_1_min: float, tier_2_min: float, tier_weights: dict) -> pd.DataFrame:
    alloc = rep_df.copy()
    alloc["tier"] = alloc["revenue"].apply(lambda v: assign_bonus_tier(v, tier_1_min, tier_2_min))
    alloc["weight"] = alloc["tier"].apply(lambda t: get_tier_weight(t, tier_weights))
    alloc["weighted_revenue"] = alloc["revenue"] * alloc["weight"]
    total_weighted = alloc["weighted_revenue"].sum()
    if total_weighted > 0:
        alloc["allocation_pct"] = alloc["weighted_revenue"] / total_weighted
    else:
        alloc["allocation_pct"] = 0.0
    alloc["bonus"] = alloc["allocation_pct"] * pool
    return alloc.sort_values("bonus", ascending=False)


def main():
    st.sidebar.header("Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload an Excel file", type=["xlsx"], help="Upload a dataset or leave blank to use FrankieDS.xlsx"
    )

    if uploaded_file is not None:
        source = uploaded_file
        st.sidebar.success("Using uploaded file")
    elif DEFAULT_FILE.exists():
        source = DEFAULT_FILE
        st.sidebar.info(f"Using local file: {DEFAULT_FILE.name}")
    else:
        st.error("Could not find FrankieDS.xlsx in the workspace. Upload a file to continue.")
        return

    with st.spinner("Loading workbook..."):
        raw_df = load_excel_data(source)

    st.subheader("Raw worksheet snapshot")
    st.write(raw_df.head(10))

    with st.spinner("Applying load-time filters..."):
        df = preprocess_dataset(raw_df)

    st.markdown("---")
    st.subheader("Dataset summary")
    st.write("All payment statuses are retained. Only Smoke Arsenal customer rows are excluded.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows after filters", len(df))
    c2.metric("Unique customers", df["Customer"].nunique(dropna=True))
    c3.metric("Unique current salespersons", df["current_sp"].nunique(dropna=True))
    c4.metric("Total revenue", format_currency(df["revenue"].sum()))

    min_date = df["invoice_date"].min()
    max_date = df["invoice_date"].max()
    st.write(
        f"Invoice date range: {min_date.date() if pd.notna(min_date) else 'N/A'} — "
        f"{max_date.date() if pd.notna(max_date) else 'N/A'}"
    )

    if df["invoice_date"].isna().any():
        st.warning(
            "Some Invoice Date values could not be converted. Review the raw data if you expect missing or malformed dates."
        )

    st.markdown("---")
    st.subheader("Quarter selection")

    if df["invoice_date"].isna().all():
        st.error("No valid invoice dates are available for quarter selection.")
        return

    df["quarter"] = get_quarter_label(df["invoice_date"])
    quarter_list = sorted(df["quarter"].dropna().unique())
    selected_quarter = st.selectbox("Select Current Quarter", quarter_list, index=len(quarter_list) - 1)

    quarter_df = df[df["quarter"] == selected_quarter]
    prior_quarters = [q for q in quarter_list if q < selected_quarter]
    selected_prior = prior_quarters[-1] if prior_quarters else None

    st.write(f"**Selected quarter:** {selected_quarter}")
    if selected_prior:
        st.write(f"**Prior quarter:** {selected_prior}")
    else:
        st.write("**Prior quarter:** none available")

    st.markdown("---")
    st.subheader("Current quarter revenue leaderboard")

    top_customers = (
        quarter_df.groupby("Customer")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_customers["revenue"] = top_customers["revenue"].apply(format_currency)
    st.table(top_customers)

    st.markdown("---")
    st.subheader("Current salesperson breakdown")

    kl_col1, kl_col2 = st.columns([2, 1])
    key_line_input = kl_col1.text_input(
        "Key product lines (comma-separated)",
        value="",
        help="Enter key product line keywords to evaluate Tier 2 performance.",
    )
    min_key_line_rev = kl_col2.number_input(
        "Min key-line revenue for eligibility",
        min_value=0.0,
        value=0.0,
        step=500.0,
        help="Reps below this key-line threshold are excluded from the bonus pool.",
    )
    key_line_keywords = parse_key_line_keywords(key_line_input)

    perf_df = compute_rep_performance(
        df,
        selected_quarter,
        selected_prior,
        key_line_keywords,
    )

    if perf_df.empty:
        st.warning("No current-quarter data is available for the selected quarter.")
    else:
        fig = px.bar(
            perf_df,
            x="Current Salesperson",
            y="revenue",
            hover_data={
                "coverage": True,
                "new_customer_count": True,
                "new_customer_revenue": True,
                "existing_customer_revenue": True,
                "existing_growth": True,
                "key_line_revenue": True,
                "revenue": ":.2f",
            },
            title="Current salesperson performance (hover for detail)",
        )
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

        top_reps = aggregate_current_sp_revenue(quarter_df)
        top_reps["revenue"] = top_reps["revenue"].apply(format_currency)
        st.table(top_reps)

    st.markdown("---")
    st.subheader("Bonus allocation setup")

    all_reps = sorted(perf_df["Current Salesperson"].dropna().unique())

    # Build per-rep eligibility map before the multiselect so we can show reasons.
    kl_revenue_by_rep = perf_df.set_index("Current Salesperson")["key_line_revenue"].to_dict()

    def _eligibility_reason(rep: str) -> str | None:
        if rep in EXCLUDE_SP:
            return "excluded (internal/admin)"
        if rep == "Unassigned":
            return "excluded (unassigned)"
        if min_key_line_rev > 0 and kl_revenue_by_rep.get(rep, 0.0) < min_key_line_rev:
            return f"below key-line minimum ({format_currency(kl_revenue_by_rep.get(rep, 0.0))})"
        return None

    eligible_reps = [r for r in all_reps if _eligibility_reason(r) is None]
    ineligible_reps = [(r, _eligibility_reason(r)) for r in all_reps if _eligibility_reason(r) is not None]

    if ineligible_reps and min_key_line_rev > 0:
        kl_excluded = [(r, reason) for r, reason in ineligible_reps if "key-line" in reason]
        if kl_excluded:
            st.info(
                f"{len(kl_excluded)} rep(s) excluded by key-line threshold "
                f"(< {format_currency(min_key_line_rev)}): "
                + ", ".join(r for r, _ in kl_excluded)
            )

    selected_reps = st.multiselect(
        "Select salespeople eligible for this bonus pool",
        options=eligible_reps,
        default=eligible_reps,
        help="Choose which current salespeople should share the bonus pool.",
    )

    pool = st.number_input(
        "Total bonus pool",
        min_value=0.0,
        value=200000.0,
        step=1000.0,
        format="%.2f",
        help="Enter the total dollar amount available for allocation.",
    )

    st.markdown("#### Allocation tiers")
    tier_col1, tier_col2, tier_col3 = st.columns(3)
    tier_1_min = tier_col1.number_input(
        "Tier 1 minimum revenue",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        help="Salespeople above this revenue become Tier 1.",
    )
    tier_1_weight = tier_col2.number_input(
        "Tier 1 weight",
        min_value=0.1,
        value=1.4,
        step=0.1,
    )
    tier_2_min = tier_col1.number_input(
        "Tier 2 minimum revenue",
        min_value=0.0,
        max_value=tier_1_min,
        value=30000.0,
        step=1000.0,
        help="Salespeople above this revenue and below Tier 1 become Tier 2.",
    )
    tier_2_weight = tier_col2.number_input(
        "Tier 2 weight",
        min_value=0.1,
        value=1.1,
        step=0.1,
    )
    tier_3_weight = tier_col3.number_input(
        "Tier 3 weight",
        min_value=0.1,
        value=1.0,
        step=0.1,
        help="Remaining eligible salespeople receive this weight.",
    )

    tier_weights = {
        "Tier 1": tier_1_weight,
        "Tier 2": tier_2_weight,
        "Tier 3": tier_3_weight,
    }

    if len(selected_reps) == 0:
        st.warning("Select at least one eligible salesperson to compute allocations.")
    else:
        selected_perf = perf_df[perf_df["Current Salesperson"].isin(selected_reps)].copy()
        if selected_perf.empty:
            st.warning("None of the selected reps have current-quarter performance data.")
        else:
            alloc_base = selected_perf[["Current Salesperson", "revenue"]].copy()
            alloc_df = compute_bonus_allocation(alloc_base, pool, tier_1_min, tier_2_min, tier_weights)
            alloc_df = alloc_df.merge(
                selected_perf.drop(columns=["revenue"]),
                on="Current Salesperson",
                how="left",
            )

            alloc_df_display = alloc_df.copy()
            alloc_df_display["revenue"] = alloc_df_display["revenue"].apply(format_currency)
            alloc_df_display["weighted_revenue"] = alloc_df_display["weighted_revenue"].apply(format_currency)
            alloc_df_display["bonus"] = alloc_df_display["bonus"].apply(format_currency)
            alloc_df_display["allocation_pct"] = (alloc_df_display["allocation_pct"] * 100).round(2).astype(str) + "%"
            alloc_df_display["new_customer_revenue"] = alloc_df_display["new_customer_revenue"].apply(format_currency)
            alloc_df_display["existing_customer_revenue"] = alloc_df_display["existing_customer_revenue"].apply(format_currency)
            alloc_df_display["key_line_revenue"] = alloc_df_display["key_line_revenue"].apply(format_currency)

            st.markdown("---")
            st.subheader("Bonus allocation result")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Eligible reps", len(selected_reps))
            summary_col2.metric("Selected revenue", format_currency(selected_perf["revenue"].sum()))
            summary_col3.metric("Allocated pool", format_currency(alloc_df["bonus"].sum()))

            st.write(
                "Bonus is distributed by weighted revenue within the selected eligible reps. "
                "Tier weights adjust the allocation share by relative revenue band."
            )
            st.dataframe(
                alloc_df_display[
                    [
                        "Current Salesperson",
                        "tier",
                        "weight",
                        "revenue",
                        "weighted_revenue",
                        "allocation_pct",
                        "bonus",
                        "coverage",
                        "new_customer_count",
                        "new_customer_revenue",
                        "existing_customer_revenue",
                        "existing_growth",
                        "key_line_revenue",
                        "low_spend_customers",
                        "medium_spend_customers",
                        "high_spend_customers",
                    ]
                ].rename(columns={
                    "tier": "Tier",
                    "weight": "Tier weight",
                    "revenue": "Revenue",
                    "weighted_revenue": "Weighted revenue",
                    "allocation_pct": "Share",
                    "bonus": "Bonus allocation",
                    "coverage": "Coverage (customers)",
                    "new_customer_count": "New customer count",
                    "new_customer_revenue": "New customer revenue",
                    "existing_customer_revenue": "Existing account revenue",
                    "existing_growth": "Adj growth (w/ emphasis)",
                    "key_line_revenue": "Key line revenue",
                    "low_spend_customers": "Low-spend accounts",
                    "medium_spend_customers": "Medium-spend accounts",
                    "high_spend_customers": "High-spend accounts",
                }),
                use_container_width=True,
            )

            with st.expander("Tier impact breakdown"):
                st.write(
                    "The pool is split by weighted revenue. Higher tiers receive more weight, "
                    "which increases their share of the same revenue base."
                )
                tier_summary = (
                    alloc_df.groupby("tier")
                    .agg(
                        reps=("Current Salesperson", "count"),
                        revenue=("revenue", "sum"),
                        weighted=("weighted_revenue", "sum"),
                        bonus=("bonus", "sum"),
                    )
                    .reset_index()
                )
                tier_summary["revenue"] = tier_summary["revenue"].apply(format_currency)
                tier_summary["weighted"] = tier_summary["weighted"].apply(format_currency)
                tier_summary["bonus"] = tier_summary["bonus"].apply(format_currency)
                st.table(tier_summary.rename(columns={
                    "tier": "Tier",
                    "reps": "Reps",
                    "revenue": "Total revenue",
                    "weighted": "Total weighted revenue",
                    "bonus": "Bonus allocated",
                }))

            # ── Export ────────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Export")
            export_col1, export_col2 = st.columns(2)

            csv_bytes = alloc_df.to_csv(index=False).encode("utf-8")
            export_col1.download_button(
                label="Download allocation as CSV",
                data=csv_bytes,
                file_name=f"bonus_allocation_{selected_quarter}.csv",
                mime="text/csv",
            )

            audit_rows = []
            for _, row in alloc_df.iterrows():
                audit_rows.append({
                    "Quarter": selected_quarter,
                    "Current Salesperson": row["Current Salesperson"],
                    "Revenue": row["revenue"],
                    "Key line revenue": row.get("key_line_revenue", 0.0),
                    "Min key-line threshold": min_key_line_rev,
                    "Key-line eligible": row.get("key_line_revenue", 0.0) >= min_key_line_rev,
                    "Tier": row["tier"],
                    "Tier weight": row["weight"],
                    "Weighted revenue": row["weighted_revenue"],
                    "Share": row["allocation_pct"],
                    "Bonus allocation": row["bonus"],
                    "Pool": pool,
                    "Tier 1 min": tier_1_min,
                    "Tier 1 weight": tier_1_weight,
                    "Tier 2 min": tier_2_min,
                    "Tier 2 weight": tier_2_weight,
                    "Tier 3 weight": tier_3_weight,
                    "Key-line keywords": ", ".join(key_line_keywords) if key_line_keywords else "",
                })
            for rep, _ in ineligible_reps:
                audit_rows.append({
                    "Quarter": selected_quarter,
                    "Current Salesperson": rep,
                    "Revenue": perf_df.set_index("Current Salesperson")["revenue"].get(rep, 0.0),
                    "Key line revenue": kl_revenue_by_rep.get(rep, 0.0),
                    "Min key-line threshold": min_key_line_rev,
                    "Key-line eligible": False,
                    "Tier": "Excluded",
                    "Tier weight": 0.0,
                    "Weighted revenue": 0.0,
                    "Share": 0.0,
                    "Bonus allocation": 0.0,
                    "Pool": pool,
                    "Tier 1 min": tier_1_min,
                    "Tier 1 weight": tier_1_weight,
                    "Tier 2 min": tier_2_min,
                    "Tier 2 weight": tier_2_weight,
                    "Tier 3 weight": tier_3_weight,
                    "Key-line keywords": ", ".join(key_line_keywords) if key_line_keywords else "",
                })
            audit_df = pd.DataFrame(audit_rows)
            audit_csv = audit_df.to_csv(index=False).encode("utf-8")
            export_col2.download_button(
                label="Download full audit log as CSV",
                data=audit_csv,
                file_name=f"bonus_audit_{selected_quarter}.csv",
                mime="text/csv",
            )

            # ── Audit view ────────────────────────────────────────────────
            with st.expander("Audit log — all reps (eligible and excluded)"):
                st.write(
                    f"**Quarter:** {selected_quarter} | "
                    f"**Prior quarter:** {selected_prior or 'none'} | "
                    f"**Pool:** {format_currency(pool)}"
                )
                st.write(
                    f"**Tier 1:** revenue ≥ {format_currency(tier_1_min)}, weight = {tier_1_weight} | "
                    f"**Tier 2:** ≥ {format_currency(tier_2_min)}, weight = {tier_2_weight} | "
                    f"**Tier 3:** weight = {tier_3_weight}"
                )
                if key_line_keywords:
                    st.write(
                        f"**Key-line keywords:** {', '.join(key_line_keywords)} | "
                        f"**Min key-line revenue:** {format_currency(min_key_line_rev)}"
                    )
                else:
                    st.write("**Key-line keywords:** none defined")

                audit_display = audit_df[[
                    "Current Salesperson", "Tier", "Key-line eligible",
                    "Revenue", "Key line revenue", "Weighted revenue", "Share", "Bonus allocation",
                ]].copy()
                audit_display["Revenue"] = audit_display["Revenue"].apply(format_currency)
                audit_display["Key line revenue"] = audit_display["Key line revenue"].apply(format_currency)
                audit_display["Weighted revenue"] = audit_display["Weighted revenue"].apply(format_currency)
                audit_display["Share"] = (audit_display["Share"] * 100).round(2).astype(str) + "%"
                audit_display["Bonus allocation"] = audit_display["Bonus allocation"].apply(format_currency)
                st.dataframe(audit_display, use_container_width=True)

            # ── Per-rep explanation card ───────────────────────────────────
            st.markdown("---")
            st.subheader("Per-rep explanation")
            st.write(
                "Select a salesperson to see a detailed breakdown they can review and dispute if needed."
            )

            rep_details = compute_rep_details(df, selected_quarter, selected_prior, key_line_keywords)

            card_rep = st.selectbox(
                "Select salesperson",
                options=sorted(alloc_df["Current Salesperson"].unique()),
                key="card_rep_select",
            )

            if card_rep:
                rep_row = alloc_df[alloc_df["Current Salesperson"] == card_rep].iloc[0]
                detail = rep_details.get(card_rep, {})

                # Header metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tier", rep_row["tier"])
                m2.metric("Revenue", format_currency(rep_row["revenue"]))
                m3.metric("Share", f"{rep_row['allocation_pct'] * 100:.1f}%")
                m4.metric("Bonus", format_currency(rep_row["bonus"]))

                # Tier explanation
                if rep_row["tier"] == "Tier 1":
                    st.info(
                        f"**Tier 1** — your revenue of {format_currency(rep_row['revenue'])} "
                        f"met the Tier 1 threshold of {format_currency(tier_1_min)}. "
                        f"Your revenue was multiplied by {rep_row['weight']}× when calculating your share of the pool."
                    )
                elif rep_row["tier"] == "Tier 2":
                    st.info(
                        f"**Tier 2** — your revenue of {format_currency(rep_row['revenue'])} "
                        f"was above the Tier 2 threshold of {format_currency(tier_2_min)} "
                        f"but below the Tier 1 threshold of {format_currency(tier_1_min)}. "
                        f"Your revenue was multiplied by {rep_row['weight']}× when calculating your share of the pool."
                    )
                else:
                    st.info(
                        f"**Tier 3** — your revenue of {format_currency(rep_row['revenue'])} "
                        f"was below the Tier 2 threshold of {format_currency(tier_2_min)}. "
                        f"Your revenue was used at face value (weight = {rep_row['weight']}×) to calculate your share."
                    )

                # Pool share explanation
                st.write(
                    f"Your weighted revenue of {format_currency(rep_row['weighted_revenue'])} "
                    f"represented **{rep_row['allocation_pct'] * 100:.2f}%** of the total weighted revenue "
                    f"across all {len(selected_reps)} eligible reps, "
                    f"giving you **{format_currency(rep_row['bonus'])}** from the "
                    f"{format_currency(pool)} pool."
                )

                # New accounts
                new_accounts = detail.get("new_accounts", [])
                st.markdown(f"#### New accounts ({len(new_accounts)})")
                if new_accounts:
                    new_df = pd.DataFrame(new_accounts)
                    new_df["Revenue"] = new_df["Revenue"].apply(format_currency)
                    new_df["Key-line revenue"] = new_df["Key-line revenue"].apply(format_currency)
                    st.dataframe(new_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No new accounts this quarter.")

                # Inactive accounts
                inactive_accounts = detail.get("inactive_accounts", [])
                st.markdown(f"#### Inactive accounts — bought last quarter, not this quarter ({len(inactive_accounts)})")
                if inactive_accounts:
                    inactive_df = pd.DataFrame(inactive_accounts)
                    inactive_df["Prior revenue"] = inactive_df["Prior revenue"].apply(format_currency)
                    st.dataframe(inactive_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No accounts went inactive this quarter.")

                # Existing accounts — spend tier breakdown
                existing_accounts = detail.get("existing_accounts", [])
                st.markdown(f"#### Existing accounts ({len(existing_accounts)})")
                if existing_accounts:
                    exist_df = pd.DataFrame(existing_accounts)

                    # Spend tier summary strip
                    tier_counts = exist_df["Spend tier"].value_counts()
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Low-spend accounts", tier_counts.get("Low", 0))
                    tc2.metric("Medium-spend accounts", tier_counts.get("Medium", 0))
                    tc3.metric("High-spend accounts", tier_counts.get("High", 0))

                    low_growth = exist_df.loc[exist_df["Spend tier"] == "Low", "Adj growth"].sum()
                    total_adj_growth = exist_df["Adj growth"].sum()
                    if total_adj_growth > 0:
                        st.write(
                            f"Low-spend account growth (1.5× emphasis): **{format_currency(low_growth)}** "
                            f"— {low_growth / total_adj_growth * 100:.1f}% of your total emphasis-adjusted growth"
                        )

                    # Format for display
                    display_exist = exist_df.copy()
                    for col in ["Revenue", "Prior revenue", "Raw growth", "Adj growth", "Key-line revenue"]:
                        display_exist[col] = display_exist[col].apply(format_currency)
                    display_exist["Emphasis mult"] = display_exist["Emphasis mult"].apply(lambda v: f"{v:.2f}×")
                    st.dataframe(display_exist, use_container_width=True, hide_index=True)
                else:
                    st.write("No returning accounts this quarter.")

                # Key-line items
                if key_line_keywords:
                    key_line_items = detail.get("key_line_items", [])
                    st.markdown(f"#### Key-line sales ({len(key_line_items)} products)")
                    if key_line_items:
                        kl_df = pd.DataFrame(key_line_items)
                        kl_df["Revenue"] = kl_df["Revenue"].apply(format_currency)
                        st.dataframe(kl_df, use_container_width=True, hide_index=True)
                    else:
                        st.write("No key-line products sold this quarter.")

                # Per-rep export
                st.markdown("##### Export this rep's breakdown")
                rep_export_rows = []
                for acct in new_accounts:
                    rep_export_rows.append({"Type": "New account", **acct})
                for acct in existing_accounts:
                    rep_export_rows.append({"Type": "Existing account", **acct})
                for acct in inactive_accounts:
                    rep_export_rows.append({"Type": "Inactive account", **acct})
                rep_export_df = pd.DataFrame(rep_export_rows)
                rep_csv = rep_export_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {card_rep} breakdown as CSV",
                    data=rep_csv,
                    file_name=f"rep_breakdown_{card_rep.replace(' ', '_')}_{selected_quarter}.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
