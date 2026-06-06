# Intuit SMB Underwriting Challenge: Repo Analysis and Action Plan

_Last updated: 2026-06-05_

## Executive decision

Treat this as a **selection-biased SMB credit underwriting + survival forecasting + causal counterfactual** challenge, not a generic classification contest.

The highest expected-value solution is:

```text
1. Build a calibrated probability-of-default model.
2. Convert PD into approve/decline decisions using expected profit, not a fixed 0.5 threshold.
3. Build a week-by-week default timing model for approved loans.
4. Answer intervention queries with a separate causal-safe counterfactual model.
5. Produce defensible 90% intervals and a regulator-ready writeup.
6. Validate every output with validate_submission.py before upload.
```

The main technical traps are **selective labels**, **censoring**, **profit-thresholding**, **counterfactual vs observational prediction**, and **calibrated uncertainty**.

---

## Official brief and judging alignment

The brief is for the Intuit NY Tech Week AI/ML Hackathon in New York City, June 5-6, 2026. The challenge defines the team as a small-business lender deciding which SMB applications to fund, maximizing portfolio profit, and defending what would happen under interventions that were never observed.

Official score components:

```text
SP&L    realized portfolio value of the loans funded in A
Straj   accuracy of cohort default-trajectory forecasts in B
Scal    90% interval calibration for PD and trajectory forecasts without needless width
SC      accuracy of interventional PD predictions in C against true causal effects
Swrite  quality of the methodological defense in D
```

Implication for execution:

```text
1. A is a profit policy, not only a PD prediction file.
2. B must forecast timing for the team's own approved loans, not all applicants.
3. Intervals should be empirically calibrated; wide intervals are not a free win.
4. C must be written and modeled as intervention, not naive feature association.
5. D should explicitly defend selective labels, censoring, causal assumptions, and regulatory explainability.
```

Operational constraints from the brief:

```text
Friday 20:00      team registration should be complete; no team changes after
Saturday 14:00    submission deadline
Saturday 16:00    awards / closing
Before upload     run the official README flow and validate_submission.py until PASS
```

---

## What the repo actually asks for

You must submit exactly four files, flat in one folder:

```text
submission_A_decisions.csv
submission_B_trajectory.csv
submission_C_counterfactuals.csv
submission_D_writeup.pdf
```

The files must satisfy the validator before upload.

### Deliverable A — lending policy

For every applicant in `validation.csv` plus `test.csv`, output:

```text
applicant_id
decision                  # 1 approve, 0 decline
predicted_pd              # calibrated PD for everyone, including declines
pd_lower_90
pd_upper_90
```

Rows required: **13,306**.

### Deliverable B — default timing trajectory

For each `cohort_week` 1-13 and `loan_age_weeks` 1-13, predict the cumulative default fraction among **your approved loans** from that cohort.

Rows required: **169**.

Rules:

```text
0 <= cumulative_default_rate <= 1
cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90
cumulative_default_rate must be non-decreasing with loan_age_weeks within each cohort
```

### Deliverable C — counterfactual intervention predictions

For each intervention query:

```text
query_id
applicant_id
feature_name
intervention_value
```

predict:

```text
predicted_pd_cf
pd_cf_lower_90
pd_cf_upper_90
```

Rows required: **900**.

The repo says the intervention is `do(feature = value)` while holding everything else fixed. That makes model perturbation acceptable as a baseline, but the writeup must explain the causal limitations.

### Deliverable D — technical writeup

A max-4-page PDF with these sections, in this order:

1. Problem framing & assumptions violated
2. Methodology
3. Causal reasoning & counterfactual methodology
4. Calibration & uncertainty quantification
5. Limitations & what we'd do differently

Section 3 is the highest-leverage human-reviewed section.

---

## Dataset contract

### Files

```text
dataset/dataset-compressed.zip
  train.csv        85,340 x 44
  validation.csv    4,489 x 44
  test.csv          8,817 x 44

dataset/data_dictionary.csv
dataset/intervention_queries.csv
dataset/cohort_week_definitions.csv
dataset/submission_B_template.csv
expected_ids/
validate_submission.py
requirements.txt
submission_D_writeup_template.md
```

### Column groups

The 44 columns fall into:

```text
business_identity
self_reported
bank_feed
bureau_credit
platform_engagement
application_context
prior_underwriter
outcome
```

Outcome columns are always blank in test. In train, outcomes exist only for prior-approved matured loans; prior-declined or immature loans have blank outcomes. Validation is labeled and should be used for tuning/calibration.

### Product terms

Every funded loan uses fixed terms:

```text
amount = requested_amount
term = 60 days
daily ACH repayment
APR = 35% annualized
origination fee = 3%, collected up front
```

Approximate gross margin on a fully repaid loan:

```text
origination fee + 60-day simple APR interest
= 0.03 + 0.35 * 60 / 365
= 8.753% of requested_amount
```

If default LGD were 100%, the rough break-even PD threshold would be:

```text
8.753% / (100% + 8.753%) = 8.05%
```

But do **not** hard-code 8.05%. Estimate loss/recovery from `final_recovered_amount`, `days_to_default`, and validation outcomes.

### Default definition

A loan defaults if any of these happen:

```text
1. 3 consecutive missed ACH draws
2. 6 total missed ACH draws over the life of the loan
3. positive outstanding balance at day 90
```

`days_to_default` is the first day in `[1, 90]` when a default condition is met. Week 13 of Deliverable B effectively covers the full 90-day default window.

---

## Recommended winning architecture

## 1. Build a source-of-truth feature registry

Create `configs/feature_registry.yaml` with each column classified as:

```text
id
feature_known_at_application
feature_with_structural_missingness
prior_underwriter_proxy
outcome_or_post_outcome
counterfactual_query_feature
causal_safe_feature
profit_feature
survival_feature
```

Immediate rules:

```text
Drop from modeling: default_flag, days_to_default, days_to_full_repayment,
repayment_status, final_recovered_amount, observation_status.

Use with caution for A: prior_underwriter_score, prior_decision, prior_approved_amount.

Exclude or downweight for C: prior_underwriter_score, prior_decision, prior_approved_amount,
because they are decision-system outputs/proxies, not direct causal drivers.
```

Keep missingness indicators, especially for bank-feed fields. `has_linked_bank_feed = False` makes bank-feed nulls meaningful, not random.

## 2. Fix the labeled-data problem before modeling

The challenge has selective labels:

```text
train labels are mostly from prior-approved matured loans
train prior-declines are often unlabeled
validation is labeled
```

Use three datasets internally:

```text
labeled_train = train rows with known default_flag and matured outcomes
labeled_validation = validation rows with known default_flag
unlabeled_train = train rows with missing outcome fields
```

Do:

```text
- Train baseline PD models on labeled_train.
- Calibrate/tune on labeled_validation.
- Estimate prior-approval / label-observed propensity to understand selection bias.
- Consider inverse-propensity weighting for labeled_train.
- Use validation to stress-test performance on applicants not restricted to prior approvals.
```

Do not naively fill missing `default_flag = 0`. That would teach the model that prior-declined or censored loans did not default.

## 3. Train two PD models, not one

### Model A: predictive underwriting model

Purpose: maximize A profitability and ranking.

Candidate features:

```text
all application-time features
business identity features
self-reported features
bank-feed features + missingness indicators
bureau-credit features
platform-engagement features
application-context features
prior_underwriter features
```

Models:

```text
CatBoostClassifier
LightGBMClassifier
XGBoostClassifier
regularized logistic regression baseline
```

Blend calibrated probabilities from the best 2-4 models.

### Model C: causal-safe counterfactual model

Purpose: stable and defensible intervention predictions.

Candidate features:

```text
same as Model A, except remove prior_underwriter outputs/proxies
and consider monotonic constraints/sign checks for risk variables.
```

Use this for `submission_C_counterfactuals.csv` unless validation strongly proves otherwise.

## 4. Calibrate PDs aggressively

Primary targets:

```text
log loss
Brier score
ECE / calibration-by-decile
reliability plot
profit by PD decile
```

Recommended approach:

```text
1. Train model ensemble with grouped/time-aware CV where possible.
2. Generate out-of-fold predictions for labeled training rows.
3. Calibrate on validation using isotonic regression or Platt scaling.
4. Compare calibration before/after.
5. Store final calibrator and apply to validation + test.
```

Use validation outcomes for calibration; do not directly leak validation labels into validation-row predictions in a way you cannot defend. For the submitted validation rows, prefer OOF/cross-fit predictions or a single locked model pipeline applied uniformly.

## 5. Choose approvals by expected profit

Create expected profit per applicant:

```text
A = requested_amount
p = calibrated PD
fee = 0.03 * A
interest_if_paid = 0.35 * (60 / 365) * A
profit_if_paid = fee + interest_if_paid
loss_if_default = model-predicted LGD dollars, or conservative estimate
expected_profit = (1 - p) * profit_if_paid - p * loss_if_default
```

Approve when:

```text
expected_profit > buffer
```

Use a buffer to account for model error and interval uncertainty:

```text
buffer = 0                  # aggressive
buffer = 0.005 * A          # safer
buffer = uncertainty_penalty # best if calibrated
```

Tune approval policy on validation:

```text
- expected portfolio profit
- approval rate
- profit by PD decile
- sensitivity to LGD assumptions
- performance when excluding prior_underwriter features
```

The approval threshold should vary by requested amount/LGD risk, not just PD.

## 6. Model default timing with discrete-time survival

For B, a single PD is not enough. Build a week-level survival dataset from rows with known outcomes.

For each funded/labeled loan and week `a = 1..13`:

```text
target_event_by_week = 1 if days_to_default <= 7*a else 0
hazard_at_week = 1 if default first happens during week a else 0
```

Train either:

```text
Option 1: direct cumulative default classifiers for each week
Option 2: discrete-time hazard model and convert hazards to cumulative CDR
Option 3: parametric curve using predicted PD + learned timing distribution
```

Recommended hybrid:

```text
- Fit a discrete-time hazard model using applicant features + loan_age_week.
- Also fit a timing distribution by risk decile.
- Blend hazard output with decile timing curves for stability.
- Aggregate only over applicants approved by your A policy.
- Shrink sparse cohort curves toward global approved-portfolio curve.
- Apply cumulative maximum to enforce monotonicity.
```

For each approved applicant:

```text
p_i(a) = P(default by week a | applicant i, approved)
```

For each cohort week:

```text
cumulative_default_rate(w, a) = mean_i[p_i(a)] among approved applicants in cohort w
```

If a cohort has very few approved applicants, use empirical-Bayes shrinkage:

```text
cdr_shrunk = weight * cohort_curve + (1 - weight) * global_curve
weight = n_approved / (n_approved + k)
```

## 7. Counterfactual logic for C

For each query:

```text
row = applicant features for applicant_id
row_cf = row.copy()
row_cf[feature_name] = intervention_value
predicted_pd_cf = calibrated_causal_safe_model(row_cf)
```

Then apply causal guardrails:

```text
- Clip to [0, 1].
- Use same preprocessing as base model.
- Recompute engineered features only if logically required and documented.
- Do not change other features unless the repo explicitly says to propagate effects.
- For obviously monotone risk features, flag/sign-check counterfactual deltas.
```

Important: `intervention_queries.csv` may include features not marked `intervenable=True` in the dictionary, so do not filter queries. Answer every query_id.

## 8. Build calibrated 90% intervals

### A and C interval recipe

Use an ensemble + conformal widening:

```text
1. Train N models across seeds/folds/model families.
2. For each applicant, get calibrated PD samples.
3. Base interval = 5th/95th percentile of ensemble predictions.
4. On validation, compute residuals |y - p| or bin-wise calibration error.
5. Widen intervals by a conformal quantile within risk bins.
6. Clip to [0, 1] and enforce lower <= point <= upper.
```

For C, widen intervals for out-of-distribution interventions:

```text
if intervention_value is outside the training range for that feature:
    add extra interval width
```

### B interval recipe

For each cohort/week:

```text
- Bootstrap approved applicants within the cohort.
- Recalculate cumulative curves.
- Combine bootstrap uncertainty with model ensemble uncertainty.
- Use 5th/95th percentiles.
- Apply monotonicity to point, lower, and upper curves.
```

## 9. Writeup strategy

The writeup should be technical and direct. Do not write marketing copy.

Must say:

```text
- The data violates iid/fully-observed-label assumptions because training outcomes
  are selectively observed for prior-approved matured loans.
- The underwriting decision is optimized for expected profit, not accuracy.
- Default timing is modeled as survival/hazard, not a second classifier.
- Counterfactuals are do-interventions with all else fixed, but observational data
  cannot prove causal effects without assumptions.
- Prior underwriter features are useful predictive proxies but not causal drivers.
- Intervals are calibrated on validation and widened for uncertainty/shift.
```

Include one small causal DAG in text form:

```text
business quality / cash flow / credit risk -> requested amount, prior underwriter score, default
platform engagement -> bank-feed availability, prior lending history, default
application channel / timing -> applicant mix, default
prior underwriter decision -> label observed in train
```

Regulatory defense:

```text
- Separate predictive features from causal claims.
- Explain that protected-class proxies were not intentionally used.
- Use monotonic sanity checks for credit-risk features.
- Provide feature importance and partial dependence only as model diagnostics,
  not causal proof.
- Prefer causal-safe model for intervention claims.
```

---

## Concrete project plan

## Day 0 / first 60 minutes

```bash
git clone https://github.com/intuit/intuit-techweek-nyc-hackathon-2026.git challenge
cd challenge
unzip dataset/dataset-compressed.zip -d dataset
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn lightgbm xgboost catboost scipy matplotlib joblib pyyaml tqdm optuna shap
```

Create output folders:

```text
outputs/submission/
outputs/reports/
runs/
configs/
src/
notebooks/
```

## Day 1 morning — audit and baselines

```text
1. Load train/validation/test.
2. Verify shapes and IDs.
3. Build feature registry from data_dictionary.csv.
4. Identify labeled_train and labeled_validation.
5. Train logistic + CatBoost + LightGBM PD baselines.
6. Evaluate log loss, Brier, AUROC, calibration deciles, and validation profit.
7. Save first valid A/C/B dummy submission and run validator.
```

## Day 1 afternoon — profit, survival, intervals

```text
1. Build expected-profit decision policy.
2. Fit LGD/recovery model or conservative default-loss table.
3. Fit survival/hazard model for days_to_default.
4. Build B output from approved validation+test applicants.
5. Add interval generation for A/B/C.
6. Run validator after every file write.
```

## Day 2 morning — counterfactual and robustness

```text
1. Build causal-safe PD model.
2. Generate C by perturbing one feature per query.
3. Add monotonic/sign sanity checks for major risk features.
4. Stress-test decisions without prior_underwriter features.
5. Compare validation calibration and profit across model variants.
```

## Day 2 final — writeup and validation

```text
1. Fill submission_D_writeup_template.md.
2. Export to PDF named submission_D_writeup.pdf.
3. Put exactly four files flat in outputs/submission/.
4. Run python validate_submission.py outputs/submission.
5. Inspect row counts and interval bounds one final time.
6. Upload only after PASS.
```

---

## Minimum viable submission path

If time is tight, do this:

```text
A:
  CatBoost PD model -> isotonic calibration -> expected-profit threshold.

B:
  Predicted PD * learned empirical cumulative timing curve by risk decile,
  aggregated over approved applicants by cohort; cummax for monotonicity.

C:
  Same calibrated model with single-feature perturbation, but preferably exclude
  prior_underwriter features for causal defensibility.

Intervals:
  Ensemble percentiles + validation conformal widening.

D:
  Be honest and precise about selective labels, censoring, causal assumptions,
  calibration, and limitations.
```

## Highest-upside improvements

Prioritize in this order:

```text
1. Calibration and expected-profit thresholding.
2. Selection-bias handling / validation-based recalibration.
3. Survival/timing model for B.
4. Counterfactual-specific causal-safe model and monotonic checks.
5. Bootstrap/conformal intervals.
6. Model ensembling and Optuna tuning.
```

## Things not to waste time on

```text
- Deep neural nets for tabular data.
- Beautiful dashboards.
- Slide decks.
- Overly complex causal packages before a strong baseline works.
- Hand-tuned thresholds without validation profit curves.
- Treating missing train outcomes as non-defaults.
```
