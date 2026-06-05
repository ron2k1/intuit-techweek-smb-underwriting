# Intuit ML Hackathon Game Plan — Actual Repo-Specific Strategy

_Last updated: 2026-06-05_

## Winning thesis

This is an SMB underwriting challenge with four simultaneous objectives:

```text
A. decide whom to fund and estimate PD
B. forecast when defaults accumulate by origination cohort
C. answer do-intervention counterfactual PD queries
D. defend the methodology in a concise technical PDF
```

The right strategy is **not** “train the best classifier and threshold at 0.5.” The right strategy is:

```text
calibrated PD
+ expected-profit approval policy
+ default-timing survival model
+ causal-safe counterfactual model
+ calibrated 90% intervals
+ direct writeup about selection bias and causal assumptions
```

## Judge-facing scorecard

Design every experiment around the official scoring surface:

```text
SP&L    maximize realized portfolio value from A approval decisions
Straj   accurately forecast cumulative default timing for approved cohorts in B
Scal    make 90% intervals cover truth without being needlessly wide
SC      predict true interventional PDs for C, not just observational correlations
Swrite  defend the method clearly in the 4-page writeup
```

Practical meaning:

```text
Do not optimize AUROC alone.
Do not approve by a fixed PD threshold unless it is justified by expected profit.
Do not build B on all applicants; build it on the approved set created by A.
Do not let C reuse every leakage/proxy feature without a causal-support check.
Do not bury the causal reasoning section; it is the highest-leverage writeup section.
```

---

## Non-negotiable repo contract

Submit exactly these four files:

```text
submission_A_decisions.csv
submission_B_trajectory.csv
submission_C_counterfactuals.csv
submission_D_writeup.pdf
```

Put them flat in one folder. Run:

```bash
python validate_submission.py outputs/submission
```

A valid submission must print `PASS`.

---

## Core facts to build around

## Dataset

```text
train.csv:       85,340 rows x 44 columns
validation.csv:   4,489 rows x 44 columns
test.csv:         8,817 rows x 44 columns
```

A requires decisions for:

```text
validation + test = 13,306 applicants
```

C requires:

```text
900 intervention queries
```

B requires:

```text
13 cohort weeks x 13 loan ages = 169 rows
```

## Loan terms

```text
funded amount = requested_amount
term = 60 days
APR = 35% annualized
origination fee = 3% upfront
default window = up to day 90
```

Approximate full-repayment gross margin:

```text
3% fee + 35% * 60/365 = 8.753% of requested_amount
```

This means approval decisions should be based on **expected profit**, not just low PD.

## Outcome / default definition

A funded loan defaults when any trigger is met:

```text
3 consecutive missed daily ACH draws
6 total missed draws
outstanding balance > 0 at day 90
```

`days_to_default` gives the first day the trigger is met. Use it for B.

---

## Main challenge risks

## 1. Selective labels

Training outcomes are only filled for prior-approved matured loans. Prior-declined or immature training rows often have blank outcomes.

Do not do this:

```text
missing default_flag -> 0
```

Do this instead:

```text
- train supervised PD models only on rows with known outcomes
- use validation, which is labeled, to tune and calibrate
- estimate selection bias from prior_decision / observation_status
- consider inverse-propensity weights for train labeled rows
```

## 2. Censoring

Some historical loans are still open/censored. Do not treat open loans as paid or defaulted unless the repo says they matured.

## 3. Prior underwriter leakage/proxy risk

`prior_underwriter_score`, `prior_decision`, and `prior_approved_amount` are known before your decision, so they are legitimate predictive features for A. But they are also selection/proxy variables and should not be treated as causal drivers.

Use two feature sets:

```text
predictive_feature_set_A: includes prior_underwriter signals
causal_feature_set_C: excludes prior_underwriter outputs/proxies
```

## 4. Counterfactual mismatch

For C, the repo asks for `do(feature = value)` while holding everything else fixed. A single-row feature perturbation is the baseline, but the writeup must admit that observational data cannot identify arbitrary causal effects without assumptions.

## 5. Interval scoring

The validator only checks interval ordering and ranges, but scoring includes calibration. Use validation/calibration, not arbitrary +/- 0.05 intervals.

---

## Feature policy

## Always drop from predictive features

```text
default_flag
days_to_default
days_to_full_repayment
repayment_status
final_recovered_amount
observation_status
```

## Usually keep for A

```text
business_identity features
self_reported features
bank_feed features + missingness flags
bureau_credit features
platform_engagement features
application_context features
prior_underwriter_score
prior_decision
prior_approved_amount
```

## Usually exclude for C causal-safe model

```text
prior_underwriter_score
prior_decision
prior_approved_amount
```

## Treat as structural missingness

```text
bank_feed columns when has_linked_bank_feed = False
external-decline / inquiry recency columns when there is no such event
```

Add missingness indicators before model training.

---

## Deliverable A plan — decisions and PD

## Modeling

Train several models and blend calibrated predictions:

```text
1. Logistic regression baseline
2. CatBoostClassifier
3. LightGBMClassifier
4. XGBoostClassifier
```

Recommended target:

```text
default_flag == True
```

Training rows:

```text
labeled train rows with known default_flag
+ validation rows for calibration/final refit after model selection
```

Validation discipline:

```text
- Use validation.csv as the main calibration/tuning set.
- Inside labeled train, use group/time CV for model development.
- Do not trust random CV alone because labels are selected and time-dependent.
```

## Calibration

Evaluate:

```text
AUROC
log loss
Brier score
ECE / decile calibration
profit curve by PD threshold
approval rate
```

Calibrate using:

```text
isotonic regression if enough validation data
Platt/logistic scaling if isotonic overfits
```

## Decision policy

Estimate expected profit:

```text
amount = requested_amount
p = calibrated_pd
profit_if_paid = amount * (0.03 + 0.35 * 60 / 365)
loss_if_default = expected_loss_given_default(amount, applicant_features)
expected_profit = (1 - p) * profit_if_paid - p * loss_if_default
```

Approve when:

```text
expected_profit > safety_buffer
```

Tune the safety buffer on validation.

Start with a conservative fallback:

```text
loss_if_default = requested_amount - expected_recovery_if_default
```

Estimate `expected_recovery_if_default` from `final_recovered_amount` among known defaults. If time is short, use recovery-rate by requested_amount bin and risk decile.

## Output

```text
applicant_id,decision,predicted_pd,pd_lower_90,pd_upper_90
```

Include every validation and test applicant exactly once.

---

## Deliverable B plan — cumulative default trajectory

## Objective

For each cohort week and loan age week:

```text
P(default by day 7*a among applicants you approved in cohort w)
```

## Cohort assignment

Map `application_timestamp` into cohort weeks using:

```text
1:  2025-06-30 to 2025-07-06
2:  2025-07-07 to 2025-07-13
3:  2025-07-14 to 2025-07-20
4:  2025-07-21 to 2025-07-27
5:  2025-07-28 to 2025-08-03
6:  2025-08-04 to 2025-08-10
7:  2025-08-11 to 2025-08-17
8:  2025-08-18 to 2025-08-24
9:  2025-08-25 to 2025-08-31
10: 2025-09-01 to 2025-09-07
11: 2025-09-08 to 2025-09-14
12: 2025-09-15 to 2025-09-21
13: 2025-09-22 to 2025-09-28
```

## Recommended model

Build a discrete-time survival/hazard table:

```text
one row per loan per week
features = applicant features + loan_age_week + cohort/week effects
target = default first occurs in this week
```

Convert hazard to cumulative default probability:

```text
S(0) = 1
S(a) = S(a-1) * (1 - hazard_a)
CDR(a) = 1 - S(a)
```

Alternative fast baseline:

```text
predicted_pd_i * empirical_timing_curve_by_risk_decile(a)
```

Then aggregate over your approved applicants in each cohort:

```text
cohort_cdr[w, a] = mean_i CDR_i(a)
```

Apply:

```text
- shrinkage toward global approved curve for sparse cohorts
- cumulative max within each cohort to guarantee monotonicity
- clipping to [0, 1]
```

---

## Deliverable C plan — intervention counterfactuals

## Baseline algorithm

For each query:

```python
row = applicant_features.loc[applicant_id].copy()
row_cf = row.copy()
row_cf[feature_name] = intervention_value
predicted_pd_cf = causal_safe_calibrated_model.predict_proba(row_cf)
```

Use the causal-safe model for this. It should be trained without prior underwriter outputs.

## Guardrails

```text
- Answer every query_id, even if feature_name is not marked intervenable.
- Do not drop duplicate-looking queries; output one row per query_id.
- Keep all non-intervened features fixed.
- Preserve categorical encodings exactly.
- Clip predictions and intervals to [0, 1].
- Widen intervals for out-of-training-range intervention values.
```

## Sanity checks

For risk-increasing features, the counterfactual effect should usually have the intuitive direction:

```text
higher aggregate_credit_utilization -> higher PD
higher recent_inquiries_count_6mo -> higher PD
higher existing_debt_obligations -> higher PD
higher observed_overdraft_count_3mo -> higher PD
higher invoice_payment_delinquency_rate -> higher PD
higher observed_cash_balance_p10 -> lower PD
higher payroll_regularity_score -> lower PD
higher stated_annual_revenue -> lower or context-dependent PD
```

Do not force signs blindly, but investigate major violations.

---

## Uncertainty and intervals

## A and C

Use:

```text
model ensemble dispersion
+ validation conformal residual widening
+ risk-bin calibration
```

Output point estimates as calibrated ensemble means. Output lower/upper as 5th/95th ensemble quantiles widened enough to satisfy validation coverage goals.

## B

Use:

```text
bootstrap approved applicants within cohorts
+ model ensemble variation
+ beta-binomial/Wilson-style finite-sample adjustment
```

Ensure:

```text
cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90
all three series are within [0, 1]
point cumulative_default_rate is non-decreasing by loan_age_weeks
```

---

## Writeup narrative

The writeup should make the judges trust the methodology.

Use this structure:

## 1. Problem framing & assumptions violated

Say the problem violates standard iid/fully observed label assumptions because training outcomes are selectively observed. Decisions affect labels. Some outcomes are censored. Profit is asymmetric.

## 2. Methodology

Describe the PD ensemble, calibration, expected-profit approval rule, survival model for timing, and separate causal-safe model for C.

## 3. Causal reasoning & counterfactual methodology

This is the most important section. State:

```text
Our A model predicts default risk under the observed data-generating process.
Our C model answers do(feature=value) queries by changing one feature at a time
and holding other features fixed, as specified by the challenge. Because the data
is observational, these effects rely on adjustment/functional-form assumptions
and should not be interpreted as proven causal effects outside the queried support.
```

Explain why prior underwriter signals are excluded from causal claims.

## 4. Calibration & uncertainty quantification

Explain isotonic/Platt calibration, validation reliability curves, conformal widening, ensemble dispersion, and B bootstrapping.

## 5. Limitations

Be honest:

```text
- unobserved confounding remains
- reject inference is imperfect
- validation size limits interval calibration
- counterfactual support may be weak for extreme interventions
- final profitability depends on unobserved true recovery and repayment dynamics
```

---

## Execution checklist

```text
[ ] Register team by Friday 20:00 to receive the private submission link.
[ ] Confirm team membership is final after registration.
[ ] Clone repo.
[ ] Unzip dataset.
[ ] Build feature registry.
[ ] Audit missingness and labels.
[ ] Train PD baselines.
[ ] Calibrate PDs.
[ ] Build expected-profit approval policy.
[ ] Train survival/timing model.
[ ] Generate A.
[ ] Generate B.
[ ] Generate C.
[ ] Write D.
[ ] Run validator until PASS.
[ ] Submit exactly four files flat in one folder before Saturday 14:00.
```

---

## Recommended priority order

```text
1. Valid files that pass the validator.
2. Calibrated PD model.
3. Expected-profit approval rule.
4. Survival/timing model for B.
5. Counterfactual model and C output.
6. Interval calibration.
7. Polished, concise writeup.
8. Extra tuning/ensembling.
```
