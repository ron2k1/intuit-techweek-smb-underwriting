# Technical Concepts for the Intuit SMB Underwriting Challenge

_Last updated: 2026-06-05_

## Mental model

This challenge combines:

```text
credit risk modeling
+ profit-aware lending policy
+ survival/default-timing forecasting
+ causal counterfactual prediction
+ calibrated uncertainty
+ methodological writeup
```

The strongest solution will separate these tasks instead of forcing one model to do everything.

---

## What the judges are testing

The official brief scores five things:

```text
portfolio value
default-timing accuracy
90% interval calibration without excessive width
counterfactual PD accuracy under interventions
methodological defense
```

That means the correct technical posture is:

```text
1. Treat A as a lending policy optimized for value.
2. Treat B as a survival/timing problem over the approved loans from A.
3. Treat C as a causal what-if task, even if the final implementation uses a pragmatic perturbation baseline.
4. Treat D as part of the product, not as afterthought documentation.
```

---

## 1. Probability of default (PD)

PD is:

```text
P(loan defaults | applicant features, funded loan terms)
```

Deliverable A requires `predicted_pd` for every validation and test applicant, even declined applicants.

PD is not the same as decision. A high-PD applicant may still be approved if requested amount is small, expected recovery is high, or profit margin compensates for risk. A low-PD applicant may be declined if amount/LGD risk makes expected profit negative.

---

## 2. Expected profit / expected value

The product terms are fixed:

```text
APR = 35% annualized
term = 60 days
origination fee = 3% upfront
amount = requested_amount
```

Approximate full-repayment profit:

```text
profit_if_paid = requested_amount * (0.03 + 0.35 * 60 / 365)
```

A decision should maximize expected value:

```text
expected_profit = (1 - PD) * profit_if_paid - PD * loss_if_default
```

`loss_if_default` should account for expected recovery, not just principal.

Practical LGD approaches:

```text
fast: average recovery rate among known defaults
better: recovery rate by amount bin + PD decile
best: separate model for final_recovered_amount / requested_amount among defaults
```

---

## 3. Selective labels / reject inference

The hardest issue: training outcomes are selectively observed.

In train:

```text
outcomes are filled only for prior-approved matured loans
prior-declined and immature loans have blank outcomes
```

That means the labeled training sample is not representative of all applicants. It is a sample filtered by a previous underwriting policy.

Why this matters:

```text
P(default | labeled train) may differ from P(default | all applicants)
```

Do not fill missing outcomes as non-default. They are unknown, not negative examples.

Methods:

```text
1. Train on labeled mature loans only.
2. Calibrate on validation, which has outcomes filled.
3. Model probability of label observation / prior approval.
4. Use inverse-propensity weights to reduce selection bias.
5. Use low-weight pseudo-labels for prior declines only as a late optional experiment.
```

Writeup language:

```text
The historical labels are selected by the prior lender's approval policy. We handle
this by separating observed-outcome rows from unlabeled/censored rows and using the
fully labeled validation set for calibration and robustness checks.
```

---

## 4. Censoring

A censored loan has not yet fully matured by the observation cutoff. It is not necessarily paid or defaulted.

Relevant columns:

```text
observation_status
repayment_status
days_to_default
days_to_full_repayment
```

For PD modeling:

```text
use rows with known mature outcomes
avoid treating open/censored as paid
```

For timing/survival modeling:

```text
censored loans can contribute information up to their observed age if observation windows are available
```

If the provided CSV does not include enough censoring age information, use only mature outcomes for the first version.

---

## 5. Default definition

A funded loan defaults if any trigger occurs:

```text
3 consecutive missed daily ACH draws
6 total missed draws
outstanding balance > 0 at day 90
```

`days_to_default` is the first day the loan meets one of these rules. It supports default-timing modeling.

For Deliverable B, loan age weeks mean:

```text
week 1  -> default by day 7
week 2  -> default by day 14
...
week 13 -> default by day 91, effectively the full day-90 default window
```

---

## 6. Survival analysis / hazard modeling

Deliverable B asks for cumulative default rate over time, not a single PD.

Key quantities:

```text
hazard(a) = P(default during week a | survived to start of week a)
survival(a) = P(no default through week a)
cumulative_default(a) = 1 - survival(a)
```

Discrete-time conversion:

```text
S(0) = 1
S(a) = S(a-1) * (1 - h(a))
CDR(a) = 1 - S(a)
```

Practical model:

```text
Create one row per loan per week.
Features = applicant features + loan_age_week + cohort/time indicators.
Target = 1 if default first occurs in that week.
Train a classifier for hazard.
```

Then aggregate only over applicants you approve.

---

## 7. Calibration

Calibration means predicted probabilities match observed frequencies.

Example:

```text
Among applicants predicted at 10% PD, roughly 10% should default.
```

Why it matters:

```text
- A requires calibrated PD values.
- Profit thresholding depends on true probabilities.
- C counterfactuals are PD values, not rankings.
- Intervals are scored for calibration.
```

Useful diagnostics:

```text
Brier score
log loss
calibration-by-decile table
expected calibration error (ECE)
reliability plot
```

Useful methods:

```text
isotonic regression
Platt/logistic calibration
temperature scaling for ensembles
risk-bin recalibration
```

---

## 8. Uncertainty intervals

The challenge requires 90% intervals for A, B, and C.

Intervals should reflect:

```text
model uncertainty
calibration uncertainty
sampling uncertainty
out-of-distribution interventions
cohort sample size for B
```

Practical methods:

```text
ensemble percentiles
bootstrap
conformal widening on validation
beta-binomial / Wilson intervals for cohort rates
```

Do not create fake narrow intervals. Wide but calibrated intervals are safer than overconfident intervals.

---

## 9. Prediction vs intervention

Prediction asks:

```text
Given what we observe, what is the default risk?
```

Intervention asks:

```text
If we set feature X to value v, what would default risk be?
```

The challenge's C file uses `do(feature = value)` queries.

Important distinction:

```text
Observational model perturbation estimates how the model score changes when a feature changes.
It is not automatically a true causal effect unless the feature is manipulable,
confounding is controlled, and the intervention stays within support.
```

Because the repo says to hold everything else fixed, the practical algorithm is one-feature perturbation. The writeup should still disclose assumptions.

---

## 10. Causal DAG thinking for this challenge

A useful mental DAG:

```text
business quality -> revenue, cash balance, credit utilization, default
owner creditworthiness -> credit band, debt, inquiries, default
platform engagement -> bank feed availability, prior loans, invoice delinquency, default
application context -> channel, multi-lender inquiries, applicant mix, default
prior underwriter model -> prior_decision, label observed in train
prior_decision -> whether outcome is observed in train
requested_amount -> repayment burden, default, profit/loss
```

Implications:

```text
- Prior underwriter features can improve prediction.
- Prior underwriter features are not causal drivers of default.
- Label observation is downstream of prior_decision.
- Bank-feed missingness is informative.
- Requested amount can affect default and profit.
```

---

## 11. Confounding

A confounder affects both a feature/intervention and the outcome.

Example:

```text
business quality -> requested_amount
business quality -> default
requested_amount -> default
```

If stronger businesses request larger loans, a naive model may incorrectly infer that larger loans reduce default. You need controls and sanity checks.

Useful controls:

```text
revenue
cash balance
credit utilization
prior debt
platform history
business age
sector/geography
application timing/channel
```

---

## 12. Monotonicity sanity checks

Some relationships should usually be directionally stable:

Risk-increasing candidates:

```text
aggregate_credit_utilization
recent_inquiries_count_6mo
existing_debt_obligations
observed_overdraft_count_3mo
invoice_payment_delinquency_rate
multi_lender_inquiry_count_30d
prior_loans_default_count
requested_amount_to_observed_revenue
```

Risk-decreasing candidates:

```text
observed_cash_balance_p10
payroll_regularity_score
observed_monthly_revenue_avg_3mo
stated_annual_revenue
stated_time_in_business
vintage_years
```

Potentially ambiguous:

```text
requested_amount
application_channel
owner_personal_credit_band  # depends on encoding direction
employee_count_bucket       # depends on encoding direction
sector/geography            # anonymized categories
```

Use monotonic constraints only when encoding direction is confirmed.

---

## 13. Feature leakage

Drop outcome and post-outcome columns:

```text
default_flag
days_to_default
days_to_full_repayment
repayment_status
final_recovered_amount
observation_status
```

Review proxy-risk features:

```text
prior_underwriter_score
prior_decision
prior_approved_amount
```

They are allowed for prediction if available at application time, but they can dominate the model and weaken causal claims. Maintain separate A and C models.

---

## 14. Group and time validation

Use `application_timestamp` for temporal validation and `business_id` for group checks.

Even though businesses do not span train/validation/test, a business may have multiple applications inside a split. Avoid leakage in internal CV by grouping on `business_id` when possible.

Validation styles:

```text
random stratified CV        # sanity baseline
group CV by business_id     # avoids same-business leakage
time split                  # checks temporal drift
time + group split          # strongest internal stress test
validation.csv calibration  # final tuning/calibration
```

---

## 15. What each model is for

```text
PD model with prior_underwriter features:
  Deliverable A ranking and profit policy.

Causal-safe PD model without prior_underwriter features:
  Deliverable C counterfactuals and regulator-facing causal discussion.

Survival/hazard model:
  Deliverable B cumulative default timing.

LGD/recovery model:
  Approval decisions and expected profit.

Calibration/interval model:
  Required uncertainty bounds.
```

---

## 16. Regulator-ready explanation

Good language:

```text
We distinguish predictive associations from causal claims. Prior underwriting
outputs are used only as predictive benchmarks/proxies in the approval model and
are excluded from causal counterfactual claims. For counterfactual predictions, we
apply the challenge-specified intervention by changing one feature while holding
others fixed, and we report calibrated uncertainty that widens under weak support.
```

Avoid saying:

```text
SHAP proves this feature causes default.
```

Better:

```text
SHAP and partial dependence identify model sensitivity. Causal interpretation
requires the stated adjustment and support assumptions.
```
