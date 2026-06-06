# Metrics, Libraries, and Techniques — Intuit SMB Underwriting Challenge

_Last updated: 2026-06-05_

## Core package stack

Install the repo requirements first:

```bash
pip install -r requirements.txt
```

Then install the modeling stack:

```bash
pip install scikit-learn scipy matplotlib jupyter notebook ipykernel
pip install lightgbm xgboost catboost
pip install joblib pyyaml tqdm optuna shap
```

Optional:

```bash
pip install statsmodels mapie
```

Recommended minimum for a strong submission:

```text
pandas
numpy
scikit-learn
CatBoost
LightGBM
XGBoost
scipy
matplotlib
joblib
pyyaml
```

---

## Official scoring map

The brief names five scoring components. Use local metrics as proxies, but keep the official target in view:

```text
SP&L    portfolio profit/value from A decisions
Straj   cohort default-trajectory accuracy for B
Scal    calibration and width tradeoff for 90% intervals in A and B
SC      counterfactual/interventional PD accuracy for C
Swrite  human review of the 4-page technical defense
```

Local proxy metrics:

```text
SP&L  -> validation profit curve, approval rate, loss-given-default sensitivity
Straj -> cohort-age cumulative default error, survival calibration, monotonicity checks
Scal  -> empirical 90% coverage, mean interval width, coverage by PD/cohort bins
SC    -> intervention plausibility checks, support distance, directional sanity checks
Swrite-> checklist coverage of assumptions, causal reasoning, calibration, limitations
```

Do not over-optimize a metric that is not in the judging surface. AUROC is useful for debugging, but portfolio value, calibration, trajectory timing, and causal defensibility decide the challenge.

---

## Metrics by deliverable

## Deliverable A: PD + lending decisions

### Predictive metrics

Use on labeled validation data:

```text
AUROC                ranking quality
average precision    rare-default ranking quality
log loss             probability quality
Brier score          squared probability error
ECE                  calibration quality
calibration deciles  human-readable calibration
```

Python:

```python
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
```

### Business metrics

A is scored partly on portfolio profitability, so evaluate decisions directly.

Compute:

```text
approval_rate
mean_predicted_pd_approved
realized_default_rate_approved
expected_profit_total
expected_profit_per_approved_loan
realized_profit_total on validation if outcomes/recoveries allow
profit by PD decile
profit by requested_amount bin
```

Expected profit approximation:

```python
gross_margin_rate = 0.03 + 0.35 * 60 / 365
profit_if_paid = requested_amount * gross_margin_rate
expected_profit = (1 - pd) * profit_if_paid - pd * expected_loss_given_default
```

Tune decisions using expected profit, not F1.

### Threshold diagnostics

Create a threshold table:

```text
threshold_or_buffer
approval_rate
approved_count
mean_pd_approved
realized_default_rate_approved
expected_profit
realized_profit_proxy
```

Pick the policy that maximizes validation profit subject to acceptable uncertainty.

---

## Deliverable B: default-timing trajectory

### Core metrics

Local validation metrics:

```text
MAE by cohort/week
RMSE by cohort/week
weighted MAE by approved cohort size
integrated absolute error across weeks
monotonicity violations
coverage of 90% intervals
average interval width
```

Because the official formula is not published, optimize for stable calibrated cumulative curves rather than just one metric.

### Survival metrics

For hazard/timing model validation:

```text
week-level log loss
week-level Brier score
cumulative default curve MAE
calibration by predicted PD/risk decile
```

### Required post-processing

Always apply:

```python
curve = np.maximum.accumulate(curve)
curve = np.clip(curve, 0, 1)
lower = np.minimum(lower, curve)
upper = np.maximum(upper, curve)
lower = np.clip(lower, 0, 1)
upper = np.clip(upper, 0, 1)
```

Then re-check monotonicity.

---

## Deliverable C: counterfactual PD

### Metrics available locally

The true counterfactuals are hidden, so local evaluation is indirect.

Use:

```text
base validation PD calibration
counterfactual delta sanity checks
monotonic sign checks for risk features
out-of-distribution intervention rate
interval width by feature
duplicate-query consistency
```

### Counterfactual diagnostics

For each `feature_name` in intervention queries, report:

```text
count
mean intervention value
training min/max/p1/p99 for that feature
mean base PD
mean counterfactual PD
mean delta
share of deltas with intuitive sign, where sign is known
mean interval width
```

Flag:

```text
- interventions outside training support
- categorical values not seen in train/validation/test
- very large PD jumps
- duplicated query_ids or missing applicants
```

---

## Calibration metrics

## Brier score

```python
from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y_true, p)
```

Lower is better.

## Log loss

```python
from sklearn.metrics import log_loss
ll = log_loss(y_true, p.clip(1e-6, 1-1e-6))
```

Lower is better and strongly penalizes overconfident wrong probabilities.

## Calibration deciles

```python
import pandas as pd

def calibration_table(y, p, n_bins=10):
    df = pd.DataFrame({'y': y, 'p': p})
    df['bin'] = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    return df.groupby('bin').agg(
        n=('y', 'size'),
        mean_pred=('p', 'mean'),
        observed_rate=('y', 'mean'),
        abs_error=('y', lambda s: 0.0),
    ).assign(abs_error=lambda t: (t['mean_pred'] - t['observed_rate']).abs())
```

## ECE

```python
def expected_calibration_error(y, p, n_bins=10):
    tab = calibration_table(y, p, n_bins)
    return (tab['n'] / tab['n'].sum() * tab['abs_error']).sum()
```

---

## Interval techniques

## A/C: ensemble + conformal widening

Recommended:

```text
1. Train multiple models/folds/seeds.
2. Calibrate each model or calibrate blended output.
3. Compute per-row ensemble mean, p05, p95.
4. On validation, compute residual = abs(y - p_mean).
5. Within risk bins, compute q90 residual.
6. lower = min(p05, p_mean - q90_bin_adjusted)
7. upper = max(p95, p_mean + q90_bin_adjusted)
8. clip to [0, 1]
```

For C, add OOD widening:

```python
if intervention_value < train_p01[feature] or intervention_value > train_p99[feature]:
    width += extra_ood_width
```

## B: bootstrap + beta-binomial adjustment

For each cohort/week:

```text
1. Bootstrap approved applicants.
2. Recompute mean cumulative default curve.
3. Take p05/p95 across bootstraps.
4. Widen for model ensemble uncertainty.
5. Apply monotonic post-processing.
```

If using predicted Bernoulli probabilities, approximate finite-sample variance:

```python
var = np.mean(p_i * (1 - p_i)) / n_approved
se = np.sqrt(var)
lower = point - 1.645 * se
upper = point + 1.645 * se
```

Clip and monotonicize.

---

## Model choices

## CatBoost

Best first model for this challenge because it handles categorical features and missing values well.

Use for:

```text
PD model
causal-safe PD model
hazard model baseline
LGD/recovery model
```

Notes:

```text
- pass categorical feature names/indices
- use class weights if default is imbalanced
- tune depth, learning_rate, l2_leaf_reg, iterations
- use early stopping on validation
```

## LightGBM

Fast and strong for tabular data.

Use for:

```text
PD ensemble
hazard model
feature importance
monotonic constraints if feature direction is confirmed
```

Notes:

```text
- use categorical_feature where possible
- handle missing values natively
- tune num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction
```

## XGBoost

Useful ensemble member.

Use with:

```text
hist tree method
one-hot/ordinal encoding
scale_pos_weight for imbalance
```

## Logistic regression

Use as transparent baseline and calibration sanity check.

```text
numeric imputation
one-hot categoricals
standard scaling
L2 regularization
```

---

## Feature engineering techniques

## Missingness indicators

Add:

```text
is_missing_<feature>
```

Especially for bank-feed fields.

## Ratios

Existing ratio:

```text
requested_amount_to_observed_revenue
```

Additional safe ratios:

```text
requested_amount / stated_annual_revenue
existing_debt_obligations / stated_annual_revenue
existing_debt_obligations / observed_monthly_revenue_avg_3mo
requested_amount / observed_cash_balance_p10_abs_adjusted
prior_loans_default_count / max(prior_loans_count, 1)
```

Be careful with denominator zero/null. Add missing/invalid flags.

## Time features

From `application_timestamp`:

```text
cohort_week
day_of_week
week_of_year
month
is_weekend
```

Use cohort/week effects for B. Avoid future information.

## Risk deciles

Create PD deciles for:

```text
calibration tables
profit curves
B timing curves
LGD tables
interval widths
```

---

## Selection-bias techniques

## Prior approval propensity

Train a model:

```text
target = prior_decision == approved
features = application-time features before prior decision
```

Use it to compute:

```text
approval_propensity = P(prior approved | features)
```

Then use weights for labeled train rows:

```text
weight = 1 / clip(approval_propensity, 0.05, 1.0)
```

This reduces over-reliance on the prior-approved population.

## Validation recalibration

Because validation is labeled across applicants, use it to correct overall PD level:

```text
p_calibrated = calibrator.fit_transform(p_model, y_validation)
```

Keep a holdout fold if you want an unbiased estimate of calibration quality.

---

## Causal/counterfactual techniques

## Baseline perturbation

```text
For each query, copy applicant row, set feature to intervention_value,
predict with causal-safe calibrated model.
```

This matches the repo's “holding everything else fixed” requirement.

## Causal-safe model

Exclude:

```text
prior_underwriter_score
prior_decision
prior_approved_amount
outcome columns
```

Consider excluding features that are clearly post-intervention for a given query, but do not overcomplicate unless time allows.

## Support checks

For each query:

```text
feature p01, p99, min, max in train+validation
is intervention within support?
is categorical value observed?
```

Use support checks to widen intervals.

---

## Submission-file checks

## A

```text
columns exactly required by scorer:
applicant_id, decision, predicted_pd, pd_lower_90, pd_upper_90

row count: 13,306
unique applicant_id
predicted_pd and intervals in [0, 1]
decision in {0, 1}
```

## B

```text
columns:
cohort_week, loan_age_weeks, cumulative_default_rate, cdr_lower_90, cdr_upper_90

row count: 169
full 13 x 13 grid
cumulative_default_rate monotone by cohort
values and intervals in [0, 1]
```

## C

```text
columns:
query_id, predicted_pd_cf, pd_cf_lower_90, pd_cf_upper_90

row count: 900
unique query_id
values and intervals in [0, 1]
```

---

## Recommended experiment sequence

```text
exp_001_dummy_valid_submission
exp_002_logistic_pd_baseline
exp_003_catboost_pd
exp_004_lightgbm_pd
exp_005_calibrated_blend
exp_006_profit_threshold_policy
exp_007_survival_curve_baseline
exp_008_hazard_model
exp_009_counterfactual_perturbation
exp_010_intervals_v1
exp_011_final_ensemble
```

Stop adding complexity once validation profit/calibration stops improving.
