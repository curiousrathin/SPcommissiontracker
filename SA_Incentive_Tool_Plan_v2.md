# Smoke Arsenal — Sales Incentive Allocation Tool
## Claude Code Implementation Plan · v2 (Sample-Validated)
**Python + Streamlit · ~1M row dataset · March 2026 · Confidential**

---

> **v2 Changes — All confirmed against sample.xlsx (62,627 rows, April 2025)**
> - Revenue field is `Untaxed Total` in the raw file — rename to `revenue` at load time
> - Emphasis channel modifier uses `Channel` field (not `Sales Team`)
> - Nancy + Shahvar confirmed as return/credit rows — added to exclusion list
> - Company Pool added to exclusions (unattributed POS/Website)
> - BC Accounts: exclude from rep scoring; its accounts appear as Unassigned
> - Payment Status filter: include Paid / In Payment / Partially Paid; exclude Reversed + Not Paid
> - Null Salesperson rows (1,827): include in company totals, exclude from rep scoring
> - Excel serial date conversion required: `origin='1899-12-30'`
> - Salesperson vs Current Salesperson divergence: 50.8% of rows differ — attribution logic is critical

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Schema](#2-dataset-schema)
3. [Metrics Definition](#3-metrics-definition)
4. [Scoring & Bonus Allocation](#4-scoring--bonus-allocation)
5. [Streamlit UI — Four Tabs](#5-streamlit-ui--four-tabs)
6. [Configuration & File Structure](#6-configuration--file-structure)
7. [Build Phases](#7-build-phases)
8. [Data Caveats](#8-data-caveats)

---

## 1. Project Overview

### 1.1 What We Are Building

A local Python + Streamlit application that ingests a large transaction-level dataset (~1 million rows), computes weighted performance metrics per salesperson, and produces an interactive UI allowing management to:

- Upload or point to the dataset file (CSV or Excel)
- Select which salespeople are eligible to receive a bonus allocation
- Enter the total approved bonus pool (e.g. $2,000)
- Specify which product lines are "key lines" for that period
- Review a company-wide summary dashboard before drilling into individual rep scorecards
- See a transparent, auditable breakdown of how the bonus pool was allocated

### 1.2 Why Local (Not Cloud)

The dataset is ~1 million rows. Running locally means pandas can process the full file in seconds, no data leaves the machine, and there is no dependency on external services.

### 1.3 Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Frontend UI | Streamlit | Fast to build, runs locally, renders dataframes and charts natively |
| Data processing | pandas | ~1M rows fits in memory; well-documented; fastest iteration |
| Visualisation | Plotly (`st.plotly_chart`) | Interactive charts with hover; export-friendly |
| Export | openpyxl / xlsxwriter | One-click Excel download of scorecards and allocation |
| Config persistence | JSON sidecar file | Saves eligible reps, key lines, pool size between sessions |

---

## 2. Dataset Schema

### 2.1 All 28 Source Fields (Confirmed from sample.xlsx)

| Field Name (exact in file) | Plan Alias | Used For | Notes |
|---|---|---|---|
| `Invoice Date` | `invoice_date` | All time series | **Excel serial int — convert with `origin='1899-12-30'`** |
| `Due Date` | `due_date` | Not used in scoring | Load but ignore |
| `Order` | `order_id` | Deduplication reference | Groups multiple lines into one order |
| `MProduct` | `mproduct` | Not used directly | Aggregate via `Product` field instead |
| `Product` | `product` | Key line matching, brand analysis | 3,643 unique values in sample |
| `Product Category` | `product_category` | Category filters | Hierarchical e.g. `Vape > Nicotine Vapes > Disposable E-Cigs` |
| `Qty Ordered` | `qty_ordered` | Volume analysis | `float64` in source (not int) |
| `Sales Team` | `sales_team` | Context only | `'Point of Sale'` / `'Sales'` / `'Website'`. **NOT used for channel emphasis.** |
| `Untaxed Total` | `revenue` | **PRIMARY KPI** | Pre-tax. Negative values = returns. Never clip. Rename at load. |
| `Total in Currency` | `revenue_incl_tax` | Load but do not use | Includes tax; inflates values inconsistently by province |
| `Customer` | `customer` | Primary account field | Apply SA exclusion filter here |
| `Brand` | `brand` | Brand analysis | 4.5% nulls — handle with `na=False` in filters |
| `Province` | `province` | Geographic context | e.g. `'Nova Scotia (CA)'` |
| `Salesperson` | `salesperson` | Attribution % only | 2.9% nulls. **NOT primary rep field.** See Section 2.3. |
| `Current Salesperson` | `current_sp` | **PRIMARY REP GROUPING** | 0.7% nulls. Use for all scorecard aggregation. |
| `Account Type` | `account_type` | Franchise flag | Null = Individual Store. `'Franchise'` = franchise. 72.5% null. |
| `Customer Tier` | `customer_tier` | Bronze emphasis modifier | Bronze/Silver/Gold. 72.5% null = Standard (no modifier). |
| `Franchise` | `franchise` | Franchise grouping | 71.7% null |
| `Channel` | `channel` | **C&G and Hybrid emphasis modifiers** | C&G / Hybrid Vape Store / Vape Store / Cannabis Store / etc. 14% nulls. **This is the correct field for channel modifiers — not Sales Team.** |
| `Payment Status` | `payment_status` | Load-time filter | Filter BEFORE any analysis. See Section 2.2. |
| `Company` | — | Ignore | Always `'Smoke Arsenal Inc.'` or internal entity |
| `Partner` | — | Ignore | Duplicate of Customer with contact name appended |
| `Fiscal Position` | — | Ignore | Tax rules by province |
| `Discount Amount` | — | Ignore | Not used in revenue scoring |
| `Street` / `Zip` / `Delivery City` / `Delivery Address` | — | Ignore | Geographic detail not needed |

---

### 2.2 Load-Time Filters — Apply in This Order Before Any Analysis

#### ⚠️ Step 1 — Payment Status filter

Keep only: `Paid`, `In Payment`, `Partially Paid`  
Drop: `Reversed` (cancelled invoices), `Not Paid`

```python
VALID_STATUS = ['Paid', 'In Payment', 'Partially Paid']
df = df[df['Payment Status'].isin(VALID_STATUS)]
```

#### ⚠️ Step 2 — Smoke Arsenal customer exclusion

Filter out rows where `Customer` contains `'smoke arsenal'` (case-insensitive).  
Confirmed variants in sample: `Smoke Arsenal BC`, `SMOKE ARSENAL BC`, `Bc Smoke Arsenal`, `smoke arsenal Sasha`, `Smoke Arsenal Inc.`, `Smoke Arsenal QC`

```python
df = df[~df['Customer'].str.contains('smoke arsenal', case=False, na=False)]
```

#### ⚠️ Step 3 — Date conversion

`Invoice Date` is stored as an Excel serial integer. Must convert before any date logic.

```python
df['invoice_date'] = pd.to_datetime(df['Invoice Date'], origin='1899-12-30', unit='D')
```

#### ⚠️ Step 4 — Rename key fields

```python
df = df.rename(columns={
    'Untaxed Total':       'revenue',
    'Current Salesperson': 'current_sp',
    'Sales Team':          'sales_team',
    'Customer Tier':       'customer_tier',
    'Account Type':        'account_type',
    'Product Category':    'product_category',
    'Qty Ordered':         'qty_ordered',
})
```

---

### 2.3 Exclusion List — Remove From Rep Scoring

Exclude the following from all rep-level metrics. All confirmed present in sample.

| Name | Appears In | Reason | Rows in Sample |
|---|---|---|---|
| `Shazia` | `current_sp` | Marketing charge-backs distort figures | 176 as Current SP |
| `Nancy` | `salesperson` | Return/credit processor: 902/907 rows negative (–$82.9K) | 907 |
| `Shahvar` | `salesperson` | Predominantly negative revenue (–$21.1K) | 215 |
| `House` | both | Internal house account | 85 SP / 53 Current SP |
| `BC Accounts` | both | Territory bucket. Manages 30 real accounts but is not a rep. Accounts appear as Unassigned. | 85 SP / 3,606 Current SP |
| `Company Pool` | `salesperson` | Unattributed POS + Website sales. Not a rep. | 298 |
| `Dakota (Staff Account)` | both | Staff purchase account | 125 SP / 49 Current SP |
| `Anas` | both | Internal | 18 SP / 18 Current SP |
| `SMOKE ARSENAL BC` | both | Internal (also caught by customer filter) | 10 SP / 16 Current SP |
| `Smoke Arsenal QC` | `salesperson` | Internal | 26 |
| `Bc Smoke Arsenal` | `salesperson` | Internal — 1,827 rows | 1,827 |

```python
EXCLUDE_SP = [
    'Shazia', 'Nancy', 'Shahvar', 'House', 'BC Accounts',
    'Company Pool', 'Dakota (Staff Account)', 'Anas',
    'SMOKE ARSENAL BC', 'Smoke Arsenal QC', 'Bc Smoke Arsenal',
]

# For rep scoring: rows where neither field is excluded AND current_sp is not null
df_reps = df[
    df['current_sp'].notna() &
    ~df['current_sp'].isin(EXCLUDE_SP) &
    (~df['salesperson'].isin(EXCLUDE_SP) | df['salesperson'].isna())
]
```

> **Null Salesperson rows:** 1,827 rows (2.9%) have no salesperson assigned (POS, Website, unattributed Sales). Include in company-wide revenue totals and customer history. Exclude from rep scoring. Do not attempt to infer a salesperson.

---

### 2.4 Feature Engineering

#### A. Spending Tier

Compute from the **full dataset** (not just the analysis period) so quarterly averages are accurate.

```python
# Must run BEFORE period filtering
df['quarter'] = df['invoice_date'].dt.to_period('Q')
cust_q   = df.groupby(['customer', 'quarter'])['revenue'].sum().reset_index()
cust_avg = cust_q.groupby('customer')['revenue'].mean().reset_index()
cust_avg.columns = ['customer', 'avg_quarterly_spend']

def spending_tier(v):
    if v > 20000:   return 'High'
    elif v >= 5000: return 'Medium'
    else:           return 'Low'

cust_avg['spending_tier'] = cust_avg['avg_quarterly_spend'].apply(spending_tier)
df = df.merge(cust_avg[['customer', 'spending_tier']], on='customer', how='left')
```

> **Why quarterly not monthly:** some customers purchase quarterly only. Monthly averages would undercount them significantly.

#### B. Current Salesperson Attribution %

For each customer, what % of their revenue was actually sold by the Current Salesperson (using the `salesperson` field as the source of truth). **This is not an edge case — 50.8% of rows in the sample have a different salesperson vs current_sp.**

```python
df['sp_is_current'] = df['salesperson'] == df['current_sp']

def attribution_pct(grp):
    total = grp['revenue'].sum()
    if total == 0: return 0.0
    return grp.loc[grp['sp_is_current'], 'revenue'].sum() / total

attr = df.groupby('customer').apply(attribution_pct).reset_index()
attr.columns = ['customer', 'current_sp_attr_pct']
df = df.merge(attr, on='customer', how='left')
```

> **Rule:** accounts where `current_sp_attr_pct < attribution_threshold` (default 50%) are excluded from the current rep's growth credit. They still appear in the rep scorecard but are flagged and zero-weighted for Metric 2.

#### C. Key Product Line Matching

User enters key line keywords (e.g. `'Nuud 50K'`, `'BC Pro'`). Match against the `Product` field using case-insensitive substring matching. Handles bundles and multi-SKU lines automatically.

```python
def match_key_line(product_name, key_lines):
    if pd.isna(product_name): return None
    for kl in key_lines:
        if kl.lower() in str(product_name).lower():
            return kl  # return canonical key line label
    return None

df['key_line'] = df['product'].apply(lambda x: match_key_line(x, key_lines))
```

> **UI:** show a live preview of matched `Product` values as the user types the key line name. This lets them verify the match catches all SKUs (including bundles with bracket-prefixed names) before running analysis.

---

## 3. Metrics Definition

### 3.1 Period Definitions

All analysis is relative to a user-selected **Current Quarter (CQ)** vs **Prior Quarter (PQ)**. Derived from `invoice_date` only. No hardcoded dates anywhere in the codebase.

- **CQ:** user-selected quarter (dropdown of all quarters in dataset, default = most recent)
- **PQ:** the quarter immediately before CQ
- **Full history:** all data in the file, used for spending tier computation and reactivation lookback

---

### 3.2 Metric 1 — New Customer Acquisition & Reactivation
**Default weight: 35%**

A customer counts as a qualifying acquisition for a rep's `current_sp` in CQ if either:

- **New:** first-ever invoice date in the dataset falls within CQ
- **Reactivated:** last order before CQ was more than 2 months before CQ start, AND they placed an order in CQ

```python
# First-ever order per customer (use full dataset)
first_order = df.groupby('customer')['invoice_date'].min().reset_index()
first_order.columns = ['customer', 'first_order_date']

# New customers in CQ
new_custs = first_order[
    first_order['first_order_date'].between(cq_start, cq_end)
]['customer'].tolist()

# Reactivated: last pre-CQ order > 2 months before CQ start AND ordered in CQ
pre_cq_last = df[df['invoice_date'] < cq_start].groupby('customer')['invoice_date'].max()
reactivation_cutoff = cq_start - pd.DateOffset(months=2)
cq_active_custs = df[df['invoice_date'].between(cq_start, cq_end)]['customer'].unique()

reactivated = [
    c for c in pre_cq_last[pre_cq_last < reactivation_cutoff].index
    if c in cq_active_custs and c not in new_custs
]

# Score per rep: new_count + (reactivated_count * 0.7)
# Assign to rep via current_sp field on those customers' CQ rows
```

> Reactivation is worth 70% of a new acquisition. New is harder — there is no prior relationship to lean on.

---

### 3.3 Metric 2 — Growth in Existing Accounts with Emphasis Modifiers
**Default weight: 30%**

For accounts present in both PQ and CQ, compute raw QoQ revenue growth ($). Apply multipliers to the raw growth dollar amount before aggregating to rep level.

| Condition | Multiplier | Field | Caveat |
|---|---|---|---|
| Spending Tier = Low | 1.5× | `spending_tier == 'Low'` | Based on full-dataset quarterly avg |
| Channel = C&G | 1.3× | `channel == 'C&G'` | `Channel` field — 14% nulls get no modifier |
| Channel = Hybrid Vape Store | 1.3× | `channel == 'Hybrid Vape Store'` | `Channel` field |
| Customer Tier = Bronze | 1.3× | `customer_tier == 'Bronze'` | 72.5% nulls = Standard, no modifier |
| Multiple conditions | Stack multiplicatively | All that apply | Low + Bronze + C&G = 1.5 × 1.3 × 1.3 = 2.535× |
| Attribution < threshold | Exclude from credit | `current_sp_attr_pct` | Flag in UI; zero-weight for this metric |

```python
# Revenue per customer in CQ and PQ
cq_rev = df[df['invoice_date'].between(cq_start, cq_end)].groupby('customer')['revenue'].sum()
pq_rev = df[df['invoice_date'].between(pq_start, pq_end)].groupby('customer')['revenue'].sum()

# Existing accounts only: present in both quarters
existing = cq_rev.index.intersection(pq_rev.index)
growth_df = pd.DataFrame({'cq_rev': cq_rev[existing], 'pq_rev': pq_rev[existing]})
growth_df['raw_growth'] = growth_df['cq_rev'] - growth_df['pq_rev']

# Merge customer metadata (take most recent values per customer in CQ)
cust_meta = (
    df[df['invoice_date'].between(cq_start, cq_end)]
    .sort_values('invoice_date')
    .groupby('customer')
    .last()[['channel', 'customer_tier', 'spending_tier', 'current_sp', 'current_sp_attr_pct']]
)
growth_df = growth_df.join(cust_meta)

# Apply multipliers
growth_df['mult'] = 1.0
growth_df.loc[growth_df['spending_tier'] == 'Low',              'mult'] *= 1.5
growth_df.loc[growth_df['channel'] == 'C&G',                   'mult'] *= 1.3
growth_df.loc[growth_df['channel'] == 'Hybrid Vape Store',      'mult'] *= 1.3
growth_df.loc[growth_df['customer_tier'] == 'Bronze',           'mult'] *= 1.3

# Zero out low-attribution accounts (flag separately in UI)
growth_df.loc[growth_df['current_sp_attr_pct'] < attr_threshold, 'raw_growth'] = 0

growth_df['adj_growth'] = growth_df['raw_growth'] * growth_df['mult']

# Aggregate to rep level
rep_growth_score = growth_df.groupby('current_sp')['adj_growth'].sum()
```

---

### 3.4 Metric 3 — Overall & Key Product Line Performance vs Company Average
**Default weight: 25%**

#### 3.4a Overall QoQ performance

```python
total_cq = df[df['invoice_date'].between(cq_start, cq_end)]['revenue'].sum()
total_pq = df[df['invoice_date'].between(pq_start, pq_end)]['revenue'].sum()
company_growth = (total_cq - total_pq) / total_pq if total_pq else 0

rep_growth_rate = (rep_cq_rev - rep_pq_rev) / rep_pq_rev if rep_pq_rev else 0
perf_vs_avg = rep_growth_rate - company_growth  # positive = beat company average
```

#### 3.4b Key product line performance

Same calculation filtered to rows where `key_line` is not null. If no key lines are specified, skip this sub-metric and redistribute its weight to 3.4a.

---

### 3.5 Metric 4 — Account Coverage Rate
**Default weight: 10%**

Coverage = accounts that ordered in CQ ÷ eligible assigned accounts.

**Eligibility rule:** the account must have been assigned to this rep (earliest `invoice_date` for this `customer`-`current_sp` pair) more than **1 month before CQ start**.

```python
# Earliest assignment date per customer-rep pair
assign = df.groupby(['customer', 'current_sp'])['invoice_date'].min().reset_index()
assign.columns = ['customer', 'current_sp', 'assignment_start']

# Eligible: assigned >1 month before CQ start
eligible = assign[assign['assignment_start'] < cq_start - pd.DateOffset(months=1)]

# Accounts with any CQ order
cq_ordered = df[df['invoice_date'].between(cq_start, cq_end)]['customer'].unique()

coverage = {}
for rep in eligible_reps:
    rep_elig = eligible[eligible['current_sp'] == rep]['customer']
    ordered  = rep_elig[rep_elig.isin(cq_ordered)]
    coverage[rep] = len(ordered) / len(rep_elig) if len(rep_elig) > 0 else 0
```

> **Why the 1-month rule:** a rep who just inherited a large territory should not be penalised for accounts they have not had time to work yet.

---

## 4. Scoring & Bonus Allocation

### 4.1 Default Metric Weights

| Metric | Default Weight | Adjustable in UI |
|---|---|---|
| New customer acquisition + reactivation | 35% | Yes — slider |
| Growth in existing accounts (emphasis-adjusted) | 30% | Yes — slider |
| Overall + key product line performance vs company avg | 25% | Yes — slider |
| Account coverage rate | 10% | Yes — slider |
| **TOTAL** | **100%** | Sliders must sum to 100% |

### 4.2 Allocation Calculation

1. Compute raw metric values for all eligible reps
2. Normalise each metric to 0–1 range: `norm = (v − min) / (max − min)`. If all reps tie, assign 0.5.
3. `weighted_score = sum(norm_i * weight_i)` for each rep
4. Exclude reps below `min_score_threshold` (default 0.10). Redistribute their share proportionally to other eligible reps.
5. `allocation = (rep_score / sum_eligible_scores) * total_pool`
6. Round to nearest dollar. Assign rounding residual to highest scorer.
7. **Assert:** `sum(allocations) == total_pool` exactly. Do not just check visually.

---

## 5. Streamlit UI — Four Tabs

### Tab 1 — Setup

- File uploader: CSV or Excel. Show row count + date range + quarter list on load.
- Quarter selector: dropdown of all quarters in dataset, default = most recent.
- Key product lines: comma-separated text input. **Live preview:** show matching `Product` values from dataset as user types, so they can verify the match catches all SKUs before running.
- Rep eligibility: multiselect of all valid `current_sp` values (exclusion list pre-filtered). Show all valid reps; user picks who is in the pool.
- Bonus pool: currency-formatted number input.
- Metric weights: four sliders with running sum shown. Must equal 100%. Include a lock button.
- Attribution threshold: slider 0–100%, default 50%.
- Min score threshold: number input, default 0.10.
- **"Run Analysis" button** — use `@st.cache_data` keyed on file hash + config hash so re-runs are fast.

### Tab 2 — Company Overview

- KPI strip: CQ revenue, PQ revenue, QoQ Δ$ and Δ%, active customers, new customers, reactivations
- Revenue bar chart: trailing 4 quarters
- Channel mix: revenue breakdown by `Channel` type (C&G, Hybrid Vape Store, Vape Store, Cannabis Store, etc.)
- Brand / product line performance table: CQ vs PQ revenue, delta, direction indicator
- Key line spotlight (if specified): CQ vs PQ for matched products
- **Company-wide QoQ growth rate — shown prominently as the benchmark reps are compared against**
- Payment Status volume: shows Reversed invoice count as a data quality indicator

### Tab 3 — Rep Scorecards

Show **all** valid `current_sp` values (after exclusion list filter). Eligible reps highlighted with a coloured badge. Non-eligible reps show the same data but scores are greyed out with label: *"Not included in this period's allocation."*

Per-rep expandable panel:
- CQ revenue, PQ revenue, QoQ growth rate, vs company average
- New customers + reactivations this quarter
- Growing accounts (with emphasis flags shown) + declining accounts
- Coverage rate: X of Y eligible accounts ordered this quarter
- Key product line revenue CQ vs PQ (if key lines specified)
- **Attribution %:** what % of their accounts' revenue was actually sold by them
- Accounts excluded from growth credit (attribution < threshold): listed with flag and reason
- Metric score breakdown: raw value, normalised score (0–1), weight applied, contribution to total score

### Tab 4 — Bonus Allocation

- Allocation table: rep name, each metric (raw + normalised + weighted), final score, % of pool, dollar amount
- Total allocated — must equal pool exactly after rounding correction
- Reps below min threshold: listed separately with their score and the threshold they missed
- **Download Excel:** summary sheet + one sheet per rep with full metric detail
- **Download PDF:** clean one-page allocation summary for management sign-off

---

## 6. Configuration & File Structure

### 6.1 config.json

```json
{
  "last_updated": "2026-03-30",
  "revenue_field": "Untaxed Total",
  "key_product_lines": ["Nuud 50K", "BC Pro"],
  "eligible_reps": ["Nikhil Bhat", "Frankie", "Dikansh Goyal"],
  "excluded_reps": [
    "Shazia", "Nancy", "Shahvar", "House", "BC Accounts",
    "Company Pool", "Dakota (Staff Account)", "Anas",
    "SMOKE ARSENAL BC", "Smoke Arsenal QC", "Bc Smoke Arsenal"
  ],
  "bonus_pool": 2000,
  "weights": {
    "new_customers": 0.35,
    "growth": 0.30,
    "performance": 0.25,
    "coverage": 0.10
  },
  "attribution_threshold": 0.50,
  "min_score_threshold": 0.10,
  "valid_payment_status": ["Paid", "In Payment", "Partially Paid"]
}
```

Load on startup and pre-populate all Setup fields. Show notice: *"Loaded saved settings from [date]."* Save back on every "Run Analysis."

### 6.2 Folder Structure

```
smoke_arsenal_incentive/
  app.py                  # Streamlit entry point, tab routing
  config.json             # Persisted settings (auto-created on first run)
  requirements.txt        # pandas streamlit plotly openpyxl xlsxwriter
  /src
    loader.py             # Load, date conversion, payment filter, SA exclusion, rename fields
    engineer.py           # Spending tier, attribution %, key line matching
    metrics.py            # All 4 metric calculations
    scorer.py             # Normalise, weight, allocate, round
    exporter.py           # Excel + PDF export
  /components
    tab_setup.py          # Tab 1 UI
    tab_overview.py       # Tab 2 UI
    tab_scorecards.py     # Tab 3 UI
    tab_allocation.py     # Tab 4 UI
    charts.py             # Shared Plotly chart functions
  /exports                # Output files land here
```

---

## 7. Build Phases

### Phase 1 — Data Foundation
**Build and validate completely before proceeding to Phase 2.**

- `loader.py`: read file, Excel serial date conversion (`origin='1899-12-30'`), payment status filter, SA customer exclusion, rename fields
- `engineer.py`: spending tier, attribution %, key line matching
- **Validation test:** load `sample.xlsx`, check shape after filters (~58K rows expected after Reversed/Not Paid drop and SA customer exclusion), confirm all feature columns created, confirm `revenue` field has no unexpected nulls
- **STOP — do not proceed to Phase 2 until Phase 1 output is verified manually**

### Phase 2 — Metrics Engine
**Build and validate completely before proceeding to Phase 3.**

- `metrics.py`: new customer + reactivation detection
  - Note: needs multi-quarter data to test QoQ fully. The sample is one month only. Document this dependency — the tool will need a full dataset to validate QoQ logic.
- `metrics.py`: QoQ growth with emphasis multipliers + attribution exclusion
- `metrics.py`: overall + key line performance vs company average
- `metrics.py`: coverage rate with 1-month assignment eligibility filter
- **Validation test:** run each metric on sample, print results per rep, manually verify at least 3 reps against raw data
- **STOP — verify before Phase 3**

### Phase 3 — Scoring & Allocation

- `scorer.py`: normalisation, weighting, allocation, rounding correction
- `scorer.py`: minimum threshold exclusion and redistribution
- **Validation:** `assert sum(allocations) == total_pool` — code must assert this, not just display it

### Phase 4 — Streamlit UI

- `app.py`: skeleton with four tabs, config load/save
- `tab_setup.py`: all inputs + live key line preview + Run Analysis button
- `tab_overview.py`: KPI strip + charts
- `tab_scorecards.py`: all-rep table + expandable per-rep detail
- `tab_allocation.py`: final table + download buttons

### Phase 5 — Export

- `exporter.py`: Excel export (summary sheet + one sheet per rep)
- `exporter.py`: PDF allocation summary

---

## 8. Data Caveats

**All caveats below were identified by validating the plan against `sample.xlsx`. Claude Code must handle every one of these.**

| Caveat | Detail | Impact if Ignored |
|---|---|---|
| **Revenue field name** | Raw file calls it `Untaxed Total`. Rename to `revenue` at load. Never use `Total in Currency` (includes tax). | Wrong revenue figures across all metrics |
| **Channel vs Sales Team** | Emphasis modifiers use `Channel` field (`C&G`, `Hybrid Vape Store`). `Sales Team` only has `Point of Sale` / `Sales` / `Website`. | Emphasis multipliers never fire |
| **Excel serial dates** | `Invoice Date` is an integer. Use `pd.to_datetime(origin='1899-12-30')`. | All date logic fails silently |
| **Payment Status filter** | `Reversed` invoices must be excluded before analysis. They are cancellations, not returns. | Revenue inflated; reversed rows corrupt QoQ calculations |
| **Nancy + Shahvar** | Predominantly negative revenue. Return/credit processors, not sales reps. | Drag down company averages; distort normalised scoring |
| **BC Accounts** | Manages real accounts but is not a rep. 3,606 rows as `current_sp`. | Phantom "rep" appears in scorecards |
| **Attribution divergence** | 50.8% of rows have different `salesperson` vs `current_sp`. Not an edge case. | Growth credit assigned to wrong rep at scale |
| **Customer Tier nulls** | 72.5% null = Standard tier. Not missing data — most accounts are untiered. | Bronze multiplier fires incorrectly if null is treated as Bronze |
| **Channel nulls** | 14% null in `Channel`. Null rows receive no channel emphasis modifier. | Crash if `.isin()` is not null-safe |
| **Spending Tier timing** | Must be computed from full dataset before period filtering. | Single-month sample classifies almost everyone as Low spender |
| **Reactivation window** | Needs 2+ months of order history before CQ start to detect reactivations. | Always returns zero reactivations with a single-month dataset |
| **Null Salesperson rows** | 1,827 rows (2.9%). Include in company totals. Exclude from rep scoring. | Attribution calculation divides by zero or assigns revenue to `None` |
| **Negative revenue** | Returns/credits are negative values. Never clip to zero. | Under-counts rep liability for returns; distorts growth metrics |
| **Product vs MProduct** | Use `Product` field (not `MProduct`) for key line matching. `Product` is the clean aggregated name; `MProduct` is the raw SKU variant. | Key line matching misses bundled SKUs |

---

## Business Context Summary

### Why Targets Cannot Be Gap-Filling

Smoke Arsenal's vape portfolio lost Ziip Labs in 2025 and is losing STLTH exclusivity in 2026. STLTH alone contributed $6.92M in 2025. The remaining lines (GH20K, BC10K, ELFBAR, Breeze, Pop Vapor) are all declining simultaneously. At Q1 2026 pace, the portfolio is tracking to ~$38M annualised vs $46.9M in 2025 — a structural –$8.9M decline.

GH20K, BC10K, Ziip, and STLTH were all **management-level brand decisions**. The sales team scaled what they were given. Setting commission targets around closing the portfolio gap asks salespeople to compensate for decisions they had no part in making.

### What the Commission Tool Measures

The tool is built around three things reps can genuinely influence:
1. Whether new or lapsed customers are brought into ordering activity
2. Whether existing accounts grow or are allowed to drift
3. What percentage of assigned accounts are actively ordering

A rep working a declining brand environment can still score well by maintaining high coverage and opening new accounts — because those outcomes reflect their effort, not the portfolio's trajectory.

### Excluded from Rep Performance Analysis

In addition to the exclusion list in Section 2.3, also filter out:
- Any row where `Customer` contains `'smoke arsenal'` (internal transfers)
- Rows where `Payment Status` is `Reversed` or `Not Paid`

---

*Smoke Arsenal · Sales Incentive Allocation Tool · Implementation Plan v2 (Sample-Validated) · March 2026 · Confidential*
