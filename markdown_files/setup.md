# Setup — Intuit SMB Underwriting Challenge

_Last updated: 2026-06-05_

This setup is specific to the official repo:

```text
https://github.com/intuit/intuit-techweek-nyc-hackathon-2026
```

The goal is to produce four validated files:

```text
submission_A_decisions.csv
submission_B_trajectory.csv
submission_C_counterfactuals.csv
submission_D_writeup.pdf
```

---

## 0. Time-critical admin

Before modeling, handle the hackathon operations that can block submission:

```text
1. Register the team by Friday 20:00 using the official form from README.md.
2. Confirm team name and 1-4 members; no team changes are allowed afterward.
3. Watch for the private submission link sent after registration.
4. Plan to finish validation before the Saturday 14:00 submission deadline.
```

Do this first. A technically strong solution cannot be uploaded without the private submission link.

---

## 1. Clone the challenge repo

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/intuit/intuit-techweek-nyc-hackathon-2026.git intuit-smb-underwriting
cd intuit-smb-underwriting
```

Check that these files exist:

```bash
ls
ls dataset
ls expected_ids
```

Expected important files:

```text
README.md
dataset/README.md
dataset/dataset-compressed.zip
dataset/data_dictionary.csv
dataset/intervention_queries.csv
dataset/cohort_week_definitions.csv
dataset/submission_B_template.csv
expected_ids/manifest.json
expected_ids/applicant_ids.txt
expected_ids/query_ids.txt
submission_D_writeup_template.md
validate_submission.py
requirements.txt
```

---

## 2. Unzip the data

```bash
cd dataset
unzip dataset-compressed.zip
cd ..
```

After unzip:

```bash
ls dataset/train.csv dataset/validation.csv dataset/test.csv
```

Expected shapes:

```text
train.csv       85,340 x 44
validation.csv   4,489 x 44
test.csv         8,817 x 44
```

---

## 3. Create a Python environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 4. Install dependencies

The repo requirements are intentionally minimal:

```bash
pip install -r requirements.txt
```

Install the modeling stack:

```bash
pip install scikit-learn scipy matplotlib jupyter notebook ipykernel
pip install lightgbm xgboost catboost
pip install joblib pyyaml tqdm optuna shap
```

Optional packages:

```bash
pip install statsmodels mapie
```

Avoid spending time debugging heavy causal packages unless the baseline is already working.

Freeze your environment:

```bash
pip freeze > requirements-dev.txt
```

---

## 5. Add a clean project structure

Create reusable folders inside the cloned challenge repo:

```bash
mkdir -p src scripts notebooks configs runs outputs/reports outputs/submission data/processed
```

Recommended structure:

```text
intuit-smb-underwriting/
  dataset/
    train.csv
    validation.csv
    test.csv
    data_dictionary.csv
    intervention_queries.csv
    cohort_week_definitions.csv
    submission_B_template.csv
  expected_ids/
  src/
    data.py
    features.py
    audit.py
    validation.py
    metrics.py
    models_pd.py
    models_survival.py
    profit.py
    counterfactuals.py
    intervals.py
    submission.py
  scripts/
    00_audit_data.py
    01_train_pd.py
    02_build_policy.py
    03_train_survival.py
    04_generate_submission.py
    05_validate_submission.py
  notebooks/
    01_data_audit.ipynb
    02_pd_modeling.ipynb
    03_profit_policy.ipynb
    04_survival_timing.ipynb
    05_counterfactuals.ipynb
    06_writeup_figures.ipynb
  configs/
    feature_registry.yaml
    baseline.yaml
    model_ensemble.yaml
  runs/
  outputs/
    reports/
    submission/
  validate_submission.py
```

---

## 6. First sanity check script

Create `scripts/00_audit_data.py`:

```python
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"

for name in ["train", "validation", "test"]:
    path = DATA / f"{name}.csv"
    df = pd.read_csv(path)
    print(name, df.shape)
    print(df.dtypes)
    print(df.isna().mean().sort_values(ascending=False).head(15))
    print()

print(pd.read_csv(DATA / "data_dictionary.csv"))
print(pd.read_csv(DATA / "cohort_week_definitions.csv"))
print(pd.read_csv(DATA / "intervention_queries.csv").head())
```

Run:

```bash
python scripts/00_audit_data.py
```

---

## 7. Create the feature registry

Create `configs/feature_registry.yaml` with this starting policy:

```yaml
id_columns:
  - business_id
  - applicant_id

timestamp_columns:
  - application_timestamp

outcome_columns_drop_always:
  - default_flag
  - days_to_default
  - days_to_full_repayment
  - repayment_status
  - final_recovered_amount
  - observation_status

prior_underwriter_predictive_only:
  - prior_underwriter_score
  - prior_decision
  - prior_approved_amount

structural_missingness:
  bank_feed_control: has_linked_bank_feed
  bank_feed_columns:
    - observed_monthly_revenue_avg_3mo
    - observed_revenue_trend_3mo
    - observed_revenue_volatility
    - observed_cash_balance_p10
    - observed_overdraft_count_3mo
    - payroll_regularity_score

profit_columns:
  - requested_amount

cohort_timestamp_column: application_timestamp
```

Update after auditing the actual CSVs.

---

## 8. Required modeling pipeline

Build the code in this order:

```text
1. src/data.py
   - load train/validation/test
   - load data dictionary, cohort weeks, intervention queries
   - verify expected row counts and IDs

2. src/features.py
   - drop outcome columns
   - add missingness indicators
   - parse application_timestamp
   - assign cohort_week
   - build A feature matrix and C causal-safe feature matrix

3. src/models_pd.py
   - logistic baseline
   - CatBoost
   - LightGBM
   - XGBoost
   - model blending
   - calibration

4. src/profit.py
   - compute good-loan gross margin
   - estimate LGD/recovery
   - choose approve/decline by expected profit

5. src/models_survival.py
   - build week-level hazard table
   - fit survival/timing model
   - aggregate approved applicants into B template

6. src/counterfactuals.py
   - load intervention_queries.csv
   - perturb one feature per query_id
   - predict PD with causal-safe model

7. src/intervals.py
   - A/C ensemble + conformal intervals
   - B bootstrap intervals

8. src/submission.py
   - write exactly formatted A/B/C CSVs
   - enforce ranges, interval ordering, B monotonicity
```

---

## 9. Build a dummy valid submission early

Before optimizing models, create a dummy submission that passes format validation. This prevents last-minute schema failures.

```bash
mkdir -p outputs/submission
```

Dummy values are not competitive, but they prove the file contract.

Run after generating files:

```bash
python validate_submission.py outputs/submission
```

Fix every error immediately.

---

## 10. Development loop

Use this loop for every experiment:

```text
1. Create run folder.
2. Save config.
3. Train model.
4. Save validation predictions.
5. Save metrics.
6. Save calibration plot/table.
7. Save A/B/C candidate files.
8. Run validator.
9. Record notes.
```

Example run folder:

```text
runs/2026-06-05_1430_catboost_pd_v3/
  config.yaml
  metrics.json
  validation_predictions.csv
  feature_importance.csv
  calibration_deciles.csv
  profit_curve.csv
  model.cbm
  notes.md
```

---

## 11. Validation commands

Basic validator:

```bash
python validate_submission.py outputs/submission
```

Inspect output files manually:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
sub = Path('outputs/submission')
for f in ['submission_A_decisions.csv','submission_B_trajectory.csv','submission_C_counterfactuals.csv']:
    df = pd.read_csv(sub / f)
    print('\n', f, df.shape)
    print(df.head())
    print(df.describe(include='all'))
PY
```

Check B monotonicity:

```bash
python - <<'PY'
import pandas as pd
b = pd.read_csv('outputs/submission/submission_B_trajectory.csv')
for w, g in b.sort_values(['cohort_week','loan_age_weeks']).groupby('cohort_week'):
    assert (g['cumulative_default_rate'].diff().fillna(0) >= -1e-12).all(), w
print('B monotonicity OK')
PY
```

---

## 12. Export the writeup to PDF

Start from:

```text
submission_D_writeup_template.md
```

Keep the required section names and order. Export as:

```text
outputs/submission/submission_D_writeup.pdf
```

Pandoc option:

```bash
pandoc submission_D_writeup_filled.md \
  -o outputs/submission/submission_D_writeup.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=0.75in \
  -V fontsize=11pt
```

If Pandoc is not available, export from VS Code, Google Docs, or another Markdown editor, but the final filename must be exact.

---

## 13. Final submission checklist

```text
[ ] output folder contains exactly the four required files
[ ] A has 13,306 rows
[ ] B has 169 rows
[ ] C has 900 rows
[ ] all probabilities are in [0, 1]
[ ] all interval lower <= point <= upper
[ ] B point curve is monotone by cohort
[ ] no extra subfolders in final upload folder
[ ] validate_submission.py prints PASS
[ ] writeup is <= 4 pages body, 11pt+, 0.75in+ margins
```
