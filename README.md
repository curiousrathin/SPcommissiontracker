# Smoke Arsenal Incentive Tracker

A local Streamlit app to load the `FrankieDS.xlsx` dataset and begin the incentive allocation workflow.

## Run locally

1. Create a virtual environment if needed:

```bash
python3 -m venv .venv
```

2. Activate the environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Notes

- The app loads `Sheet 1` from `FrankieDS.xlsx`.
- Place your data file in the project root — it is excluded from version control via `.gitignore`.
- Applies load-time filters: Smoke Arsenal customer exclusion, Excel serial date conversion, column renaming.
- Computes spending tiers (Low / Medium / High) from full dataset history before any period filtering.
- Bonus pool allocation with tier weights, key-line eligibility gate, and per-rep explanation cards.
