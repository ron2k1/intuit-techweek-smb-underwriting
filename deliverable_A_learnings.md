# Deliverable A Learnings

## Current State

Deliverable A is a calibrated underwriting policy, not a generic classifier.

The best observed-label AUROC so far is:

```text
0.757504
```

Best ranking configuration:

```text
model: LogisticRegression
feature_set: baseline_no_prior_score
C: 0.06
class_weight: None
score type: raw probability score
```

The earlier calibrated ensemble baseline was:

```text
AUROC: 0.754979
```

So the ranking lift from additional model work was small:

```text
+0.002525 AUROC
```

## What This Means

The main bottleneck is probably not model family. CatBoost and LightGBM did not materially improve ranking on labeled validation rows.

Do not spend the remaining effort trying to push observed-label AUROC from ~0.76 to ~0.77 unless the lift also improves calibrated NPV. The official A/B surface is not "best classifier":

```text
A decision column controls realized portfolio value.
A predicted_pd + interval column controls calibration/uncertainty.
B is scored on the default trajectory of the loans we approved in A.
```

The stronger interpretation is:

```text
observed labels are selected by the prior underwriter
```

The supervised model mostly learns:

```text
P(default | prior lender approved, applicant features)
```

not the full target:

```text
P(default | applicant features, our approval policy)
```

That selection structure is the key statistical trap.

Local dataset check:

```text
train observed outcome rate      = 0.6061
train prior approval rate        = 0.6061
validation observed outcome rate = 0.5683
validation prior approval rate   = 0.5683
test observed outcome rate       = 0.0000
```

So the label mechanism is not incidental; it is engineered into the data.

## Important Findings

### Prior Underwriter Score

`prior_underwriter_score` is useful for support and selection diagnostics, but it can hurt ranking when used directly in the PD model.

Removing it produced the best AUROC so far.

Additional check:

```text
prior_underwriter_score perfectly separates prior approvals from prior declines.
But inside the observed approved population, 1 - prior_underwriter_score has only weak default AUROC:
train:      0.5868
validation: 0.5469
```

Interpretation: it is mostly a support/selection signal, not a sufficient risk signal.

Interpretation:

```text
prior_underwriter_score may cause the model to imitate the old underwriting gate
instead of ranking true default risk better.
```

Use it for:

```text
- prior-approval support diagnostics
- uncertainty widening
- policy risk review
- writeup explanation
```

Do not blindly rely on it as a core PD predictor.

### Confounder Features

The dataset has strong latent confounder clusters:

```text
cash stress
credit stress
business maturity
revenue scale
repayment burden
platform engagement
prior-underwriter selection support
```

Hand-built index features did not improve AUROC enough to keep as default production features. They remain useful for diagnostics and writeup framing.

### AUROC Ceiling

Observed-label AUROC appears to be near a practical ceiling around:

```text
0.755 - 0.758
```

Further score gains are more likely to come from:

```text
- approval policy economics
- loss/recovery estimation
- selection-aware uncertainty
- Deliverable B timing accuracy
- Deliverable C causal-support handling
```

not from small classifier tweaks.

### Economics Beats Fixed PD Thresholding

The brief's product terms imply a paid-loan margin of:

```text
3% origination fee + 35% APR * 60 / 365 = 8.7534% of requested_amount
```

But default is not automatically a full principal loss. Defaults may occur after successful ACH draws, and `final_recovered_amount` recovers part of the principal. On observed validation rows:

```text
mean paid-loan NPV       ~=  2.4k
mean default-loan NPV    ~= -4.9k
rough break-even PD      ~=  0.33
```

This is why an apparently high PD cutoff can still be economically valid. The decision should be dollar-weighted expected value, not a fixed PD threshold and not a 0.5 classifier threshold.

## Actions To Carry Forward

## 1. Prior Approval / Selection

Do not treat historical declines as non-defaults.

Keep a prior-approval propensity model:

```text
P(prior_decision = 1 | applicant features)
```

Use it for:

```text
- support diagnostics
- interval widening for low-support applicants
- approval policy guardrails
- B cohort aggregation risk review
- C out-of-support warnings
```

For B, this matters because the approved set may include applicants outside the historical approved-label support. Their timing predictions should have wider uncertainty.

## 2. Approval Policy

The decision rule should remain:

```text
approve if E[NPV | approve] > buffer
```

not:

```text
approve if PD < fixed threshold
```

The A policy should feed directly into B. B must forecast default trajectories for the team's own approved set, not all validation/test applicants.

Implementation decision:

```text
Use E[NPV] from the brief's cash-flow equation.
Tune/record a per-dollar buffer for robustness.
Default to a positive buffer when validation diagnostics show a better realized NPV/approved-book tradeoff than zero buffer.
Report approval rate, mean approved PD, expected NPV, and realized labeled-validation NPV.
```

The current practical rule is:

```text
approve if E[NPV | approve] / requested_amount > buffer_rate
```

where `buffer_rate` is small and positive by default. This avoids approving marginal loans whose modeled positive NPV is within calibration/timing/recovery noise.

Current selected buffer:

```text
buffer_rate = 0.005
labeled-validation realized NPV = 3,789,087.66
labeled-validation approve-all NPV = 2,139,702.49
labeled-validation approved count = 2,015 / 2,551 observed rows
validation+test approval rate = 0.6730
```

## 3. Default Timing For B

B should use the A approved applicants and estimate:

```text
P(default by week a | applicant i, approved)
```

Recommended first B implementation:

```text
PD_i * empirical timing curve by risk decile
```

Then aggregate:

```text
cumulative_default_rate(cohort_week, age_week)
  = mean_i P(default by age_week | applicant i)
```

Only include applicants approved by the A decision policy.

If time allows, upgrade to a discrete-time survival model:

```text
one row per approved/labeled loan per week
target = default first occurs during this week
features = applicant features + loan_age_week
```

But a decile timing curve is the right fast baseline.

Current implementation uses the stronger version:

```text
discrete-time weekly hazard model
one row per observed approved loan per week
features = applicant features + loan_age_week
aggregate Pr(default by week a | x_i) over A-approved validation+test applicants
```

B output must be named:

```text
submission_B_trajectory.csv
```

Keep any count-bearing diagnostic files separate from the official submission CSV.

## 4. Middle-Trick / Cohort Drift

The middle-trick analysis did not find a giant week-7 feature cliff, but it did flag possible cohort/time drift.

For B, compute and monitor:

```text
- approval rate by cohort_week
- mean PD by cohort_week
- mean prior_underwriter_score by cohort_week
- mean prior-approval propensity by cohort_week
- approved count by cohort_week
```

Sparse or low-support cohorts should shrink toward the global approved timing curve.

## 5. Uncertainty

Uncertainty should be baked into B, not added as arbitrary bands.

For A:

```text
ensemble dispersion
+ calibration-bin error
+ prior-approval support penalty
```

For B:

```text
approved-applicant bootstrap
+ finite-sample beta-binomial/Wilson uncertainty
+ support widening for low-support cohorts
+ monotonic post-processing
```

B intervals must satisfy:

```text
cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90
values in [0, 1]
cumulative_default_rate non-decreasing by loan_age_weeks
```

## 6. Counterfactual C

C should not reuse the A model blindly.

Use a causal-safe model that excludes prior-underwriter outputs/proxies:

```text
prior_underwriter_score
prior_decision
prior_approved_amount
```

For interventions:

```text
set feature_name = intervention_value
hold other features fixed
recompute deterministic engineered features when needed
```

Do not modify or drop columns from:

```text
data/intervention_queries.csv
```

That file is the source of truth for C. Any counterfactual working file should be a separate derived artifact that preserves `query_id`, `applicant_id`, `feature_name`, and `intervention_value`.

Especially recompute:

```text
requested_amount_to_observed_revenue
requested / revenue ratios
debt / revenue ratios
cash stress / repayment burden features
```

Widen C intervals when intervention values are outside train/validation support.

## Recommended Next Step

Build Deliverable B now.

The B baseline should be:

```text
1. Load final A decisions and PDs.
2. Assign validation/test applicants to cohort_week.
3. Learn empirical default timing curves from train labeled defaults.
4. Stratify timing by PD/risk decile.
5. For each approved applicant, convert PD into cumulative-by-week default probabilities.
6. Aggregate by cohort_week and loan_age_weeks.
7. Shrink sparse cohorts toward the global approved curve.
8. Bootstrap/Wilson intervals.
9. Enforce monotonicity and bounds.
10. Write submission_B_trajectory.csv.
```

This is more likely to improve total challenge score than continuing to chase small AUROC gains.
