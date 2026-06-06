# Advanced Statistical Edges for the Intuit Techweek NYC Hackathon 2026

Based on the challenge repository, this is not a vanilla default-classification hackathon. It is a joint **underwriting policy + survival forecasting + causal inference + calibrated uncertainty** challenge: Deliverable A is approve/decline plus PD, Deliverable B is cohort default timing, Deliverable C is `do(feature=value)` counterfactual PD, and Deliverable D is a technical defense. Scoring includes profitability, timing accuracy, interval calibration, counterfactual accuracy, and writeup quality, with exact weights unpublished.

## Top 10 Advanced Statistical Edges

### 1. Selection bias / reject inference / censoring correction

The training labels are not missing at random: historical outcomes are filled only for loans the prior lender approved and that matured; declined or immature loans have blank outcomes.

A strong team would explicitly model the prior approval process, likely using `prior_underwriter_score`, `prior_decision`, and approval propensity weighting, then correct or sensitivity-test PD estimates for the rejected population.

Novice teams will train only on observed approved loans and assume the resulting PD generalizes to everyone.

### 2. Expected-profit underwriting, not PD classification

Winning Deliverable A is a **decision problem**, not just a probability problem. The product has requested amount as exposure, a 60-day term, 35% APR, and a 3% upfront origination fee.

A sophisticated team would estimate expected profit per applicant using PD, loss-given-default, recovery, amount, and timing. A rough zero-recovery sanity check puts break-even PD near single digits, not 50%, so “approve if PD < 0.5” is catastrophic.

### 3. Discrete-time survival / hazard modeling for default timing

Deliverable B asks for cumulative default fractions by cohort week and loan age. The default process is time-based: missed daily ACH draws, cure behavior, cumulative missed draws, and day-90 unpaid balance all matter.

Strong teams will model weekly or daily hazards, cumulative incidence, censoring, and repayment/default competing risks.

Novices will predict a single default flag and smear it over 13 weeks with an average curve.

### 4. Policy-coupled trajectory aggregation

Deliverable B is not “the market’s default curve.” It is the cumulative default rate of **your approved loans** in each cohort.

A winning implementation computes individual survival curves for every applicant, applies the team’s own Deliverable A approval policy, then aggregates `F_i(7a)` over approved loans within each cohort week.

Novices will produce B independently of A, creating incoherence between their policy and trajectory forecast.

### 5. Calibration and honest 90% uncertainty intervals

Calibration is explicitly scored for A and B, and the submissions require lower/point/upper interval estimates.

Strong teams will use out-of-fold calibration, isotonic or beta calibration for PD, conformalized intervals, bootstrap model uncertainty, and beta-binomial or hierarchical intervals for cohort curves.

Novices will submit arbitrary `±0.05` or `±0.10` bands that pass validation but miss coverage.

### 6. Causal inference for C, not blind feature perturbation

Deliverable C asks for PD under an intervention: set one feature to a value, holding other features fixed, i.e. `do(feature=value)`. The writeup also says causal reasoning is the most heavily weighted section.

Strong teams will separate observational prediction from intervention using a causal graph, support checks, doubly robust / DML treatment-effect estimates where possible, causal forests or local effect models, and sensitivity analysis for non-manipulable proxies.

Novices will simply run `model.predict(x with one column changed)` and call it causal.

### 7. Informative missingness and measurement-process modeling

Bank-feed fields are null when the applicant did not link a feed, and other fields are legitimately blank for structural reasons.

Strong teams will model missingness as signal: missingness indicators, separate submodels for linked vs. unlinked applicants, imputation conditional on link propensity, and interactions between “has feed” and bank-feed values.

Novices will median-impute everything and erase a major behavioral signal.

### 8. Temporal / cohort shift handling

The challenge uses application timestamps and 13 origination cohort weeks for Deliverable B.

Strong teams will use time-aware validation, recency weighting, cohort-specific calibration, and drift diagnostics between train, validation, and test.

Novices will random-split rows, overestimate performance, and miss cohort effects.

### 9. Hierarchical shrinkage and shape constraints

There are many sparse segments: sector, geography, employee bucket, cohort week, channel, prior-loan history, and bank-feed availability.

Strong teams will partially pool risk estimates across related groups, shrink noisy cohort/segment effects, enforce monotonic cumulative default curves, and possibly use monotone constraints for credit-risk variables.

Novices will trust raw cell rates or let a tree ensemble overfit small pockets.

### 10. Robust validation, ensembling, and regulator-grade diagnostics

The best teams will build out-of-fold pipelines: calibrated GBMs/CatBoost, penalized GLMs or GAMs, survival models, recovery models, and causal nuisance models, then stack them without leakage.

They will also defend drivers using stability checks, monotonicity, subgroup calibration, partial-dependence sanity checks, and unobserved-confounding sensitivity—not just SHAP charts.

This matters because the writeup is human-reviewed for substance, and it explicitly asks teams to distinguish observational from interventional reasoning and defend default drivers to a regulator.

## Execution Trap

The validator enforces exact filenames, ID coverage, valid probability ranges, interval ordering, and non-decreasing B trajectories. A failing submission cannot be scored.

## Sources

- Challenge repository: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026
- Participant instructions / README: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026/blob/master/README.md
- Dataset guide: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026/blob/master/dataset/README.md
- Data dictionary: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026/blob/master/dataset/data_dictionary.csv
- Writeup template: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026/blob/master/submission_D_writeup_template.md
- Validator: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026/blob/master/validate_submission.py
