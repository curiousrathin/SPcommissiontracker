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
- It applies the first load-time filters from the plan:
  - Payment Status filtering
  - Smoke Arsenal customer exclusion
  - Invoice Date conversion from Excel serial dates
  - Column renaming for `revenue`, `current_sp`, and related fields

## Next steps

- Add key line matching
- Add rep eligibility and bonus pool allocation
- Add export and report tabs
