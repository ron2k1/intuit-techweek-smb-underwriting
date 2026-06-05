# Codex Prompts for Winning the Intuit SMB Underwriting Challenge

_Last updated: 2026-06-05_

Use these prompts inside Codex or Claude Code. They are written for the official repo:

```text
https://github.com/intuit/intuit-techweek-nyc-hackathon-2026
```

The target deliverables are:

```text
submission_A_decisions.csv
submission_B_trajectory.csv
submission_C_counterfactuals.csv
submission_D_writeup.pdf
```

---

## Prompt 0 — Lock the official constraints before coding

```text
Read README.md, dataset/README.md, validate_submission.py, and submission_D_writeup_template.md.

Create docs/official_contract.md with:
- exact four submission filenames
- A/B/C row counts: 13,306 / 169 / 900
- required columns for A/B/C
- B monotonic cumulative-default rule
- interval-order rules for A/B/C
- flat-folder submission rule
- writeup section order, 4-page limit, 11pt minimum font, 0.75in minimum margins
- judging score components: SP&L, Straj, Scal, SC, Swrite
- operational deadline notes: register by Friday 20:00, submit by Saturday 14:00

Then add failing tests or assertions for every machine-checkable rule before building models.
```

---

## Prompt 1 — Initialize challenge-specific project structure

```text
You are working inside the cloned repo for the Intuit SMB Underwriting Challenge.

Create a clean project structure without moving the official files:

src/
  data.py
  audit.py
  features.py
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
configs/
  feature_registry.yaml
  baseline.yaml
runs/
outputs/reports/
outputs/submission/

Do not alter validate_submission.py, expected_ids/, dataset/data_dictionary.csv, dataset/intervention_queries.csv, dataset/cohort_week_definitions.csv, or dataset/submission_B_template.csv.

Add a README section explaining the pipeline:
1. audit data
2. train calibrated PD model
3. choose expected-profit approval policy
4. train survival/timing model
5. generate counterfactual predictions
6. generate intervals
7. validate final submission
```

---

## Prompt 2 — Data loader and schema checks

```text
Implement src/data.py.

Requirements:
- load train.csv, validation.csv, test.csv from dataset/
- load data_dictionary.csv, intervention_queries.csv, cohort_week_definitions.csv, submission_B_template.csv
- parse application_timestamp as datetime
- verify expected shapes if files exist:
  train: 85340 rows
  validation: 4489 rows
  test: 8817 rows
- verify applicant_id uniqueness within each split
- verify business_id does not overlap across train/validation/test, and warn if it does
- create a combined scoring_applicants dataframe = validation + test with split column
- expose functions:
  load_all(root: Path) -> dict[str, pd.DataFrame]
  get_labeled_rows(train, validation) -> tuple[pd.DataFrame, pd.DataFrame]
  assign_cohort_week(df, cohort_defs) -> pd.DataFrame
  validate_ids_against_expected(root, scoring_applicants, intervention_queries) -> dict

Important:
- Outcome columns are default_flag, days_to_default, days_to_full_repayment, repayment_status, final_recovered_amount, observation_status.
- In train, only rows with non-null default_flag and matured outcomes should be treated as supervised labels.
- Do not fill missing default_flag with zero.
```

---

## Prompt 3 — Data audit report

```text
Implement src/audit.py and scripts/00_audit_data.py.

The audit should load all official CSVs and produce outputs/reports/data_audit.md plus outputs/reports/column_profile.csv.

Include:
- shapes
- column dtypes
- missingness rates by split
- unique counts
- duplicate applicant_id checks
- business_id overlap checks
- application_timestamp ranges
- target distribution for rows with known default_flag
- prior_decision distribution
- observation_status and repayment_status distribution
- missing outcome patterns by prior_decision and observation_status
- bank-feed missingness by has_linked_bank_feed
- intervention_queries feature_name frequency table
- cohort_week counts for validation/test after assignment

Add a warning section for:
- outcome leakage columns
- prior_underwriter proxy features
- train rows with missing outcomes
- intervention features not marked intervenable in data_dictionary.csv
- any intervention values outside train+validation feature support
```

---

## Prompt 4 — Feature registry and feature builder

```text
Implement configs/feature_registry.yaml and src/features.py.

Feature registry should include:
- id columns: business_id, applicant_id
- timestamp: application_timestamp
- outcome columns to always drop:
  default_flag, days_to_default, days_to_full_repayment, repayment_status, final_recovered_amount, observation_status
- prior_underwriter predictive-only columns:
  prior_underwriter_score, prior_decision, prior_approved_amount
- bank-feed columns controlled by has_linked_bank_feed
- categorical columns from data_dictionary.csv dtype=categorical or bool
- numeric columns from data_dictionary.csv dtype=float or int

src/features.py should expose:
- build_feature_matrix(df, feature_set='predictive'|'causal_safe', fit_encoder=None)
- add_missingness_indicators(df)
- add_ratio_features(df)
- add_time_features(df)
- get_feature_lists(data_dictionary, feature_set)

Rules:
- predictive feature_set may include prior_underwriter columns.
- causal_safe feature_set must exclude prior_underwriter columns.
- both feature sets must drop outcome columns and ids.
- do not use validation/test outcomes as features.
- preserve exact applicant_id for joining outputs.
```

---

## Prompt 5 — Metrics and calibration utilities

```text
Implement src/metrics.py.

Functions:
- binary_metrics(y_true, p) returning AUROC, average_precision, log_loss, brier_score
- calibration_table(y_true, p, n_bins=10)
- expected_calibration_error(y_true, p, n_bins=10)
- profit_table(y_true, p, requested_amount, recovery_amount=None, thresholds=None)
- evaluate_policy(df, decision_col, pd_col)

The profit approximation should use:
gross_margin_rate = 0.03 + 0.35 * 60 / 365
profit_if_paid = requested_amount * gross_margin_rate

For default losses:
- if final_recovered_amount is available for defaults, use it to compute realized recovery proxy
- otherwise support a configurable lgd_rate fallback

Also implement calibration wrappers:
- fit_isotonic_calibrator(p_val, y_val)
- fit_platt_calibrator(p_val, y_val)
- apply_calibrator(calibrator, p)
```

---

## Prompt 6 — PD model training pipeline

```text
Implement src/models_pd.py and scripts/01_train_pd.py.

Train these models:
1. logistic regression baseline with preprocessing pipeline
2. CatBoostClassifier
3. LightGBMClassifier
4. XGBoostClassifier

Inputs:
- labeled train rows with known default_flag
- validation rows with known default_flag
- feature_set: predictive or causal_safe

Outputs per run:
runs/<timestamp>_<model>/
  config.yaml
  metrics_train_cv.json
  metrics_validation.json
  calibration_deciles.csv
  validation_predictions.csv
  feature_importance.csv
  model artifact

Requirements:
- never use outcome columns as features
- preserve applicant_id
- handle categorical features correctly
- add missingness indicators
- support sample weights for inverse prior-approval propensity, but default to unweighted if not ready
- use early stopping where available
- save raw and calibrated validation predictions
- compare isotonic vs Platt calibration

Choose the best calibrated model/blend based on validation log loss, Brier score, calibration error, and profit curve, not AUROC alone.
```

---

## Prompt 7 — Selection-bias / prior-approval propensity

```text
Implement a selection-bias diagnostic in src/validation.py or src/models_pd.py.

Build a model predicting prior_decision == approved using only application-time features excluding outcome columns.

Report:
- AUROC of prior approval model
- feature importance
- distribution of approval propensity among labeled train rows vs validation rows
- suggested inverse propensity weights: 1 / clip(propensity, 0.05, 1.0)

Add an option to train the PD model with these weights.

Do not pseudo-label prior-declined train rows in the default path. If adding pseudo-labeling, put it behind an explicit --use-pseudo-labels flag and assign low sample weight.
```

---

## Prompt 8 — Expected-profit approval policy

```text
Implement src/profit.py and scripts/02_build_policy.py.

Inputs:
- calibrated PD predictions for validation and test
- requested_amount
- final_recovered_amount / default_flag on labeled rows for recovery estimation

Functions:
- estimate_recovery_rate_table(df, group_cols=['pd_decile','amount_bin'])
- predict_loss_given_default(df, recovery_table, fallback_lgd_rate)
- expected_profit(df, pd_col, requested_amount_col)
- choose_decision(df, expected_profit_col, buffer)
- tune_buffer_on_validation(validation_predictions)

Expected profit formula:
gross_margin_rate = 0.03 + 0.35 * 60 / 365
profit_if_paid = requested_amount * gross_margin_rate
loss_if_default = requested_amount - expected_recovery_if_default
expected_profit = (1 - pd) * profit_if_paid - pd * loss_if_default

Output:
- outputs/reports/profit_policy_report.md
- outputs/reports/profit_threshold_table.csv
- decision column for validation + test applicants

Tune buffer on validation. Include approval rate and realized default rate among approved validation rows.
```

---

## Prompt 9 — Survival / default timing model for Deliverable B

```text
Implement src/models_survival.py and scripts/03_train_survival.py.

Goal: produce cumulative default probabilities by loan age week 1..13 for each applicant.

Use rows with known outcomes. Build a person-week table:
- one row per loan per week 1..13
- event_week = ceil(days_to_default / 7) clipped to 13 for defaulted loans
- hazard_target = 1 if event occurs in this week, else 0
- for paid-in-full rows, hazard_target = 0 for all weeks

Features:
- applicant features from predictive feature set, excluding outcome columns
- loan_age_week
- cohort_week/time features if available

Train a hazard classifier using CatBoost or LightGBM.

Convert hazard predictions to cumulative default curves:
S0 = 1
S_week = previous_S * (1 - hazard_week)
CDR_week = 1 - S_week

Also implement a fallback timing curve:
- compute empirical cumulative timing distribution among defaults by PD decile
- applicant CDR_week = predicted_pd * timing_curve[pd_decile, week]

Blend hazard model with fallback if hazard model validation is unstable.

Generate B:
- use A decisions to select approved validation+test applicants
- assign cohort_week using cohort_week_definitions.csv
- for each cohort_week and loan_age_week, average applicant CDR_week
- shrink sparse cohorts toward global approved curve
- enforce monotonicity with cumulative maximum
- fill dataset/submission_B_template.csv prediction columns

Output:
outputs/submission/submission_B_trajectory.csv
outputs/reports/b_timing_report.md
```

---

## Prompt 10 — Counterfactual predictions for Deliverable C

```text
Implement src/counterfactuals.py and integrate it into scripts/04_generate_submission.py.

Inputs:
- intervention_queries.csv
- scoring_applicants = validation + test
- trained causal_safe calibrated PD model
- preprocessing pipeline for causal_safe feature set

For every query_id:
1. locate applicant_id in scoring_applicants
2. copy applicant feature row
3. set feature_name to intervention_value
4. preserve all other features
5. rebuild derived features only if the derived feature is deterministic and directly depends on changed feature; otherwise keep everything fixed per challenge wording
6. predict calibrated PD
7. construct 90% interval using interval module
8. write exactly one row per query_id

Requirements:
- do not filter to intervenable=True; answer all 900 queries
- preserve query order if possible
- handle categorical intervention values
- handle duplicate applicant/query combinations independently
- create outputs/reports/c_counterfactual_diagnostics.csv with feature-level delta summaries and support checks

Output columns:
query_id,predicted_pd_cf,pd_cf_lower_90,pd_cf_upper_90
```

---

## Prompt 11 — Intervals for A, B, and C

```text
Implement src/intervals.py.

A/C intervals:
- accept an array/dataframe of calibrated predictions from multiple models/seeds/folds
- point = mean or selected calibrated blend
- base lower/upper = 5th/95th percentiles across model predictions
- fit conformal widening on validation residuals within risk bins
- apply widening to validation/test predictions
- widen C intervals further for interventions outside feature p01/p99 support
- clip to [0, 1]
- enforce lower <= point <= upper

B intervals:
- bootstrap approved applicants within each cohort
- recompute cumulative default curves for each bootstrap sample
- use p05/p95 as lower/upper
- if cohort has few approved loans, combine with beta-binomial standard error
- clip to [0, 1]
- enforce lower <= point <= upper
- ensure point curve is non-decreasing by cohort

Add unit tests or assertions for all interval ordering/range rules.
```

---

## Prompt 12 — Generate final A/B/C submission files

```text
Implement src/submission.py and scripts/04_generate_submission.py.

The script should:
1. load trained final models/artifacts
2. load validation + test applicants
3. generate calibrated PDs for all 13,306 applicants
4. generate pd intervals
5. compute expected_profit and decision
6. write submission_A_decisions.csv
7. generate B using approved applicants and survival curves
8. write submission_B_trajectory.csv from template grid
9. generate C from intervention queries
10. write submission_C_counterfactuals.csv
11. run internal assertions matching validate_submission.py rules

Exact columns:
A: applicant_id,decision,predicted_pd,pd_lower_90,pd_upper_90
B: cohort_week,loan_age_weeks,cumulative_default_rate,cdr_lower_90,cdr_upper_90
C: query_id,predicted_pd_cf,pd_cf_lower_90,pd_cf_upper_90

All probability columns must be numeric and in [0, 1].
Decision must be integer 0/1.
B must be a full 13x13 grid and monotone by cohort.
```

---

## Prompt 13 — Validation wrapper

```text
Implement scripts/05_validate_submission.py.

It should:
- call the official validate_submission.py on outputs/submission
- print PASS/FAIL clearly
- if failure, print diagnostics and common fixes
- also run extra checks:
  A row count = 13306
  B row count = 169
  C row count = 900
  no NaNs
  no duplicate IDs
  all intervals valid
  B monotonicity for point/lower/upper if desired
  final PDF exists

Do not replace the official validator. This wrapper should call it.
```

---

## Prompt 14 — Writeup draft generator

```text
Create submission_D_writeup_filled.md from submission_D_writeup_template.md.

Keep these exact section headers and order:
1. Problem framing & assumptions violated
2. Methodology
3. Causal reasoning & counterfactual methodology
4. Calibration & uncertainty quantification
5. Limitations & what we'd do differently

Draft concise technical content using actual results from outputs/reports:
- validation metrics
- calibration table summary
- approval rate and expected profit
- survival/timing validation result
- counterfactual methodology and support checks
- interval methodology and coverage

Required claims to include:
- training labels are selectively observed due to prior underwriting approval/maturity
- missing outcomes were not treated as non-defaults
- approval decision optimizes expected profit, not classification accuracy
- default timing is modeled with survival/hazard or timing curves
- counterfactuals are generated by do(feature=value) one-feature perturbation while holding other features fixed, as specified
- observational counterfactuals require assumptions and are not proof of unrestricted causal effects
- prior underwriter variables are predictive but not used as causal drivers

Keep the body under 4 pages with 11pt+ font and 0.75in+ margins.
```

---

## Prompt 15 — Final end-to-end runner

```text
Create scripts/run_all.py that executes the full pipeline:

python scripts/00_audit_data.py
python scripts/01_train_pd.py --models catboost lightgbm xgboost logistic --feature-set predictive
python scripts/01_train_pd.py --models catboost lightgbm --feature-set causal_safe
python scripts/02_build_policy.py
python scripts/03_train_survival.py
python scripts/04_generate_submission.py
python scripts/05_validate_submission.py

The runner should stop on any error.
It should write a final run summary to outputs/reports/final_run_summary.md containing:
- chosen model artifacts
- validation metrics
- approval rate
- expected profit estimate
- interval method
- validator result
- paths to final four files
```

---

## Prompt 16 — Fast fallback if time is running out

```text
Implement a reliable fallback submission in less than one hour.

A:
- Train CatBoostClassifier on labeled train rows using predictive features.
- Calibrate on validation using isotonic regression.
- Predict validation+test PD.
- Decision = approve if expected_profit > 0 using conservative LGD fallback.
- Intervals = calibrated p +/- validation conformal q90 by risk bin, clipped.

B:
- Learn empirical cumulative default timing curve among known defaulted loans.
- For each approved applicant, CDR_week = predicted_pd * timing_curve_week.
- Aggregate by cohort_week.
- Shrink sparse cohorts to global approved curve.
- Enforce monotonicity.

C:
- Train causal-safe CatBoost without prior_underwriter features.
- For each query, set one feature to intervention_value and predict PD.
- Use same interval logic as A, widened for OOD interventions.

D:
- Fill the template honestly with the above methodology.

Then run validate_submission.py until PASS.
```
