# Causal Trap And Middle-Trick Analysis

## Bottom Line

The main causal trap is **selection by the previous underwriter**.

`default_flag` is populated only for applications the previous lender approved and that matured. Prior-declined applicants have blank outcomes. That means a naive model trained only on non-null labels estimates:

```text
P(default | prior lender approved)
```

not:

```text
P(default | we approve this applicant)
```

This matters because Deliverable A asks for decisions across validation + test applicants, including applicants the prior lender declined.

## What Is Confounded

The previous underwriter decision acts like a selection gate:

```text
Applicant/business features -> prior_underwriter_score -> prior_decision -> observed outcome
Applicant/business features -> true default risk -> observed default, if funded
```

So variables used by the old underwriter can be both:

- predictors of whether an outcome is observed
- predictors of true repayment/default risk

Those variables are confounders/proxies for causal interpretation and counterfactual claims.

## Highest-Priority Confounders

From the local scan, the strongest confounder/proxy candidates are:

- `prior_underwriter_score`
- `invoice_payment_delinquency_rate`
- `aggregate_credit_utilization`
- `observed_cash_balance_p10`
- `requested_amount`
- `owner_personal_credit_band`
- `payroll_regularity_score`
- `observed_revenue_volatility`
- `requested_amount_to_observed_revenue`
- `bookkeeping_recency_days`
- `observed_overdraft_count_3mo`
- `vintage_years`

These variables both separate prior-approved from prior-declined applicants and correlate with observed default among prior-approved loans.

## Leakage Risks

Do not use outcome columns as features:

- `default_flag`
- `days_to_default`
- `days_to_full_repayment`
- `repayment_status`
- `final_recovered_amount`
- `observation_status`

Treat `prior_decision` and `prior_approved_amount` carefully. They are not outcome leakage, but they are previous-policy outputs. If they dominate the model, your system may imitate the prior lender rather than learn a better lending policy.

## The Likely "Trick In The Middle"

I did not find a giant obvious week-7 numeric cliff after removing `application_timestamp`, which is trivially different by cohort.

The likely middle trick is subtler:

```text
cohort/time drift
+ prior-underwriter selection behavior
+ calibration differences around the middle of the validation/test window
```

The most useful checks are:

- calibration by `cohort_week`, especially weeks 6-8
- approval rate by `cohort_week`
- mean `prior_underwriter_score` by `cohort_week`
- model error by `cohort_week` on validation rows with observed outcomes
- ablations with and without `application_timestamp`, `cohort_week`, `prior_underwriter_score`, `prior_decision`, and `prior_approved_amount`

## Week-7 Leads From The Scan

The largest non-time week-7 signals were modest, not huge:

- `observed_overdraft_count_3mo`
- `observed_monthly_revenue_avg_3mo`
- `payroll_regularity_score`
- `prior_loans_default_count`
- `observed_cash_balance_p10`
- `days_since_last_external_decline`
- `days_since_last_inquiry_elsewhere`
- `account_age_days`
- `recent_inquiries_count_6mo`
- `requested_amount_to_observed_revenue`

These should be treated as leads, not proof.

## Counterfactual Query Traps

Some intervention queries stress extrapolation:

- `recent_inquiries_count_6mo`
- `observed_overdraft_count_3mo`

Those interventions often set values near or above the high end of the validation/test distribution. Your intervals should widen for these.

Some queried features are not clean real-world levers even if they are mechanically settable:

- `prior_loans_count`
- `platform_active_months`

In the writeup, distinguish:

```text
mechanical do(feature = value)
```

from:

```text
realistic business intervention
```

## Modeling Implications

1. Build a prior-approval propensity model:

```text
P(prior_decision = 1 | X)
```

Use it for diagnostics, reweighting, or calibration strata.

2. Train PD on observed outcomes, but calibrate by:

- `prior_decision`
- prior-score bins
- `cohort_week`
- bank-feed presence

3. For prior-declined applicants, be explicit that risk is extrapolated.

4. Keep `prior_underwriter_score` as a benchmark/proxy feature, but run ablations without it.

The model-family NPV bakeoff now supports this ablation. The best challenger was `LightGBM + no_prior_score`, which removed `prior_underwriter_score` and derived score proxies:

```text
LightGBM + no_prior_score labeled-validation NPV: $3.868M
active direct-NPV blend labeled-validation NPV:   $3.835M
```

This does not prove LightGBM will win on test, but it shows that reducing dependence on prior-underwriter score can improve the NPV policy while making the reject-region story cleaner. The LightGBM/no-prior-score policy has now been promoted to active A/B because it lifts labeled-validation NPV from `$3.835M` to `$3.868M`; the writeup should still disclose that the funded prior-declined region is not directly labeled.

5. Add missing indicators for bank-feed fields because missingness is structural through `has_linked_bank_feed`.

6. For counterfactual C, set the queried feature and recompute deterministic downstream features where appropriate. The main example is:

```text
requested_amount_to_observed_revenue
```

when `requested_amount` or observed revenue changes.

7. For Deliverable B, model default timing separately from PD. A large day-90 default mass suggests a distinct terminal default mechanism from the positive-balance-at-day-90 rule.
