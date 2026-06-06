# Deliverable D - Technical Writeup

**Team:** Global Intuit Hackers

## 1. Problem framing & assumptions violated

We treated the challenge as underwriting policy plus survival forecasting plus causal counterfactual estimation, not as a generic binary classifier. The target business decision is whether funding an applicant at the requested amount creates positive risk-adjusted value. The dataset violates the usual supervised-learning assumption that labels are representative: default outcomes are observed primarily for historically approved and matured loans, while prior declines and immature loans are unlabeled. Thus a naive model estimates `P(default | prior-approved, observed features)`, not `P(default | funded by our policy, observed features)`.

Our DAG separates latent business health, credit demand, application context, bank-feed observability, bureau signals, prior-lender policy, our funding decision, repayment behavior, and outcome observation. We therefore used prior-underwriter variables as selection/support diagnostics, but did not treat them as causal drivers. We also preserved bank-feed missingness because the measurement process is informative: applicants without linked feeds are not random missing rows.

## 2. Methodology

For Deliverable A, we trained calibrated PD models on labeled training loans and converted PDs into funding decisions using the challenge cash-flow formula rather than a flat PD cutoff. For a repaid loan, value is `F + R*r*T/365`; for a default, value is `F + D*(t*-1) + rec - R`, where `R` is requested amount, `F=0.03R`, `r=0.35`, `T=60`, `D=R*(1+r*T/365)/T`, `t*` is expected default day, and `rec` is expected recovery. The active policy approves only when the NPV margin clears a threshold and applies an added guardrail for prior-declined applicants, whose labels are unobserved.

The final A policy uses a compact LightGBM/no-prior-policy feature set with raw application, bureau, bank-feed, and valid prior-loan history fields. Prior-underwriter score, prior decision, and prior-approved amount are excluded from the PD model. Current validation diagnostics show labeled-validation realized NPV of about `$3.91M`, 9,033 approved validation+test applicants, and 2,591 prior-declined approvals under explicit support guardrails.

For Deliverable B, we made the trajectory forecast coherent with A: B is estimated only over applicants approved by our A policy. We use applicant-level cumulative hazard/default curves, normalize timing to PD, aggregate by origination cohort and loan age, shrink sparse cohorts toward the global approved curve, and enforce monotonic cumulative default rates. We then apply conservative empirical-Bayes tail calibration from labeled validation cohort-age cells, with partial shrinkage rather than cohort memorization. Validation timing diagnostics show mean absolute CDR error about `0.0117`, week-13 mean predicted CDR `0.1470` vs actual `0.1465`, and interval coverage about `0.970`.

## 3. Causal reasoning & counterfactual methodology

Deliverable C asks for `p_q^cf = Pr(y_iq=1 | do(f_q=v_q), X_iq,-f_q=x_iq,-f_q)`. We therefore separated predictive associations from causal claims. The C model uses the full engineered risk/segment feature layer, including repayment burden, credit stress, cash stress, maturity, revenue scale, platform engagement, bank-feed support, and interaction features, but excludes prior-underwriter artifacts and selection-support proxies.

For each intervention query, we copy the applicant row, set the queried feature to the intervention value, recompute deterministic engineered quantities, and predict with the causal-safe calibrated model. We then apply feature-specific causal guardrails. Direct business-state interventions, such as credit utilization, overdrafts, invoice delinquency, revenue, cash balance, or requested amount, retain most or all of the model-implied delta. Self-reported, historical, application-context, platform-history, and measurement-process fields are shrunk toward the baseline prediction and receive wider intervals. We also check intervention support: 74 queries were in tail support, none were outside training min/max, and no categorical intervention used an unseen level.

For clearly monotone features, if the observational model produced a material opposite-sign effect, we neutralized the point-effect delta to zero and widened the interval rather than claiming an implausible causal effect. The causal audit found 82 raw material sign violations, all guarded, leaving 0 final material sign violations. This keeps the submission consistent with the distinction `Pr(y | do(X=x)) != Pr(y | X=x)` under confounding. We also cached duplicate applicant-feature-value interventions, so identical queries return identical counterfactual PDs.

## 4. Calibration & uncertainty quantification

We calibrated PDs with holdout/isotonic-style calibration and checked log loss, Brier score, ECE, and bin-level coverage. A intervals combine calibration-bin uncertainty, model dispersion, and conservative floors. The active A model has validation AUROC about `0.746`, log loss `0.435`, Brier `0.137`, and bin-level interval coverage at or above the 90% target.

B intervals combine applicant-level hazard uncertainty, finite-sample uncertainty, sparse-cohort widening, and tail-calibration uncertainty, then enforce interval ordering and monotonicity. C intervals use the same calibrated PD interval machinery, with additional widening for low support, historical/proxy interventions, measurement-process interventions, and monotonic neutralization. The final C file covers all 900 queries, has mean counterfactual PD about `0.295`, mean interval width about `0.114`, and no range or interval-order violations.

## 5. Limitations & what we'd do differently

The largest remaining risk is reject-region extrapolation. Prior-declined applicants have no observed repayment outcomes, so their expected NPV is modeled, not directly verified. We handled this with prior-declined margin guardrails, support diagnostics, and sensitivity reports, but cannot prove hidden-region performance from local data. The second risk is cohort/timing drift: B is strong on average, but some late-cohort ages have larger residual errors. We tail-calibrated and widened intervals for these cells, but a longer observation window would improve survival calibration.

For C, the true intervention labels are hidden and the dataset is observational. Our approach is a pragmatic hackathon compromise: use predictive models as a baseline, but separate predictive from causal features, shrink non-clean interventions, check support, neutralize implausible monotone effects, and disclose assumptions. With another day, we would add doubly robust or structural sensitivity checks for the most common C features, tune B intervals by cohort-specific coverage, and produce a regulator-facing model card with fairness and adverse-action diagnostics.
