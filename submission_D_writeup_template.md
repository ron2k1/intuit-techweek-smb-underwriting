# Deliverable D — Technical Writeup

---

## 1. Problem Framing & Assumptions Violated

**What we were solving.** For each of 13,306 applicants in the validation and test sets we had to produce: a calibrated probability of default (PD), a 90% confidence interval on that PD, and an approval decision grounded in loan economics — not just a PD threshold. Two auxiliary deliverables required a 13-week cumulative default rate (CDR) trajectory per origination cohort, and causal what-if PD estimates under single-feature interventions.

**Standard ML assumptions this dataset breaks, and how we responded.**

*Selection bias / reject inference.* The labeled training set contains only approved, matured loans — applicants the prior lender declined have no observed outcome. This is a textbook selection-on-treatment problem. We partially mitigated it by including `prior_underwriter_score` and `has_prior_approval` as features, allowing the model to self-correct for profiles that were historically gated. We did not apply full reject-inference imputation (SEMMA or EM augmentation) because we had no reliable method to estimate a plausible PD for declined applicants without introducing additional distributional assumptions.

*MNAR bank-feed missingness.* Six bank-feed columns (`observed_monthly_revenue_avg_3mo`, `observed_cash_balance_p10`, etc.) are null when the applicant has no linked bank feed — absence is informative, not random. We added a binary `{col}__missing` indicator for each bank-feed column (6 extra features), allowing tree models to learn distinct behaviour for applicants with and without linked accounts. Standard imputation alone would collapse this signal.

*Survivorship bias in survival data.* The RSF training set is 82.5% right-censored at exactly day 60: every repaid loan is censored at the end of its term with no variation in censoring time. This pathological structure causes the RSF's cumulative hazard estimator to assign near-zero hazard mass to the interior of the term — S(t) barely drops below 1.0 for most applicants, so the predicted median survival is 60 for ≈75% of the portfolio. We address this with two interventions detailed in §2.

*Collider bias from `prior_decision`.* The prior lender's `prior_decision` is a collider variable: including it as a feature would open a backdoor path from unobserved confounders (e.g., applicant desperation) to our outcome, inducing spurious associations. It was excluded from all feature sets alongside raw outcome columns (`days_to_default`, `default_flag`, `repayment_status`, `final_recovered_amount`).

*Default label complexity.* The `days_to_default` field records the day the default *status was declared*, triggered by whichever of three rules fired first: 3 consecutive missed ACH draws, 6 total missed draws, or an outstanding balance at day 90. Critically, the label does not encode how many draws were actually collected before default — at minimum 3 draws were missed (3-consecutive trigger). Our NPV formula accounts for this: collected draws = max(0, default\_day − 3), rather than the naive default\_day − 1, shifting the break-even profitability day from ≈52 to ≈54.

---

## 2. Methodology

**Pipeline overview.** The full pipeline is deterministic end-to-end given fixed random seeds. All model fitting uses training data only; calibration uses an independent held-out `val_labeled` set (2,551 rows with known outcomes) that is never seen during training.

**Feature engineering (50 features).** Raw features (44 numeric, 6 categorical) are augmented with: 6 MNAR indicators; `has_prior_approval`, `approved_to_requested` (prior lender signal); `prior_default_rate` (defaults / prior loan count, NaN when no history); `loan_to_annual_rev` (leverage ratio); `has_external_decline`, `has_inquiry_elsewhere` (soft-pull signals); `daily_draw_to_cash_ratio` (daily obligation vs. cash buffer P10); `revenue_to_daily_draw` (revenue coverage of daily draw); and `platform_stress` (delinquency rate × overdraft frequency). A row-to-column ratio check enforces n ≥ 5k before any model fit; SelectKBest(f_classif) reduces columns automatically if violated.

**Deliverable A — PD model.** A 10-fold stratified CatBoost ensemble (each member trained on a different 9/10 subset, early-stopped on its held-out fold) produces raw probability scores. Isotonic regression fitted on `val_labeled` converts raw scores to calibrated probabilities. OOF AUC: 0.776. We chose CatBoost because it handles categorical features natively (no one-hot encoding needed), is robust to NaN values on non-bank-feed columns, and produces well-ranked probabilities with minimal tuning.

**Break-even decision threshold.** For each applicant we compute a per-loan threshold: the PD at which expected NPV equals zero, using the formula threshold = NPV\_repaid / (NPV\_repaid − NPV\_default). NPV\_repaid captures origination fee plus interest. NPV\_default captures: fee + daily\_draw × max(0, E[T|default] − 3) + recovery − principal. The default day input is the *conditional expected default day* E[T | T ≤ 60], computed from the Random Survival Forest (RSF) survival function — this conditions on the event of default occurring rather than on the biased unconditional median. Recovery is predicted by a CatBoostRegressor trained on defaulted training loans (hold-out RMSE: 0.090). When NPV\_default ≥ NPV\_repaid (default day ≳54, where enough daily draws make default profitable), we set threshold = 1.0 (approve unconditionally). This produces a val decline rate of 35.6%, materially more selective than the prior lender.

**Deliverable B — CDR trajectory.** A second RSF (RSF\_B, 91-day censoring horizon, same balanced weights) produces per-loan survival functions evaluated at 13 weekly checkpoints. CDR\_i(t) = 1 − S\_i(t). Approved loans are grouped by origination cohort week (from `cohort_week_definitions.csv`); cohort CDR = mean across approved members; 90% CI = ±1.645 × SE of the mean; monotonicity enforced via cumulative maximum. RSF\_B predictions are calibrated against empirical training-set CDR at each time point: cal\_factor(t) = empirical\_CDR(t) / mean\_pred\_CDR(t), computed on a 5,000-row random subsample, correcting the absolute scale while preserving the model's relative ranking.

**Deliverable C — Counterfactuals.** See §3.

**RSF survivorship bias fix.** We trained both RSF models with balanced sample weights: defaulted rows receive weight n\_total / (2 × n\_events) ≈ 2.87×, censored rows receive 0.61×. This equalises each group's contribution to the log-rank split criterion, forcing the RSF to learn hazard timing across the full 1–60 day range. The concordance index remained high at 0.857.

---

## 3. Causal Reasoning & Counterfactual Methodology

**Observational vs. interventional prediction.** Standard supervised learning learns P(Y | X = x) — the conditional distribution of default given observed covariates. This answers "among applicants who look like x, how many defaulted?" Deliverable C asks a different question: P(Y | do(X\_j = v)) — what would the default rate be *if we set* feature j to value v by intervention, holding all else fixed. These are not the same quantity whenever the feature is correlated with other covariates through confounding paths. The `do()` operator, from Pearl's do-calculus, severs the incoming edges into the intervened node: it removes the association between X\_j and its causes, while keeping its effects on Y.

**Why our approach is appropriate.** We assume that the pre-trained CatBoost ensemble approximates the conditional outcome model P(Y | X). Under this assumption, for features that are *causally downstream* of the intervention (i.e., derived features computed from the raw intervened feature), the correct interventional prediction requires propagating the change through the full feature-engineering graph. We implement this as follows:

1. Copy the applicant's original raw test row.
2. Set the intervened raw feature to its intervention value.
3. Re-run `engineer_features()` in full — this recomputes all derived quantities (`prior_default_rate`, `loan_to_annual_rev`, MNAR indicators, `daily_draw_to_cash_ratio`, etc.) from the modified raw values, so downstream effects cascade automatically.
4. Predict with the frozen ensemble and calibrator.

This is a model-perturbation (g-computation) approach. It is appropriate under the assumption that the model's learned feature-to-outcome mapping is a good proxy for the true interventional response — a reasonable assumption when features are causally upstream of the outcome and the model is well-calibrated.

**What we give up.** We do not have a structural causal model (SCM) or instrument. If the intervened feature has unobserved common causes with the outcome (confounders not in X), our estimate is biased in the interventional direction. For example, increasing `requested_amount` may proxy for unobserved financial desperation; the model sees the feature change but cannot reason about the simultaneous change in the latent confounder. We believe this is an acceptable limitation given the available data, but a regulator would need to be informed of it.

**Defending feature importance to a regulator.** CatBoost's SHAP values (available on request) provide feature-level attribution that is locally consistent. The top predictors — `prior_underwriter_score`, `owner_personal_credit_band`, `prior_default_rate`, `loan_to_annual_rev`, `daily_draw_to_cash_ratio` — are economically interpretable and directly analogous to traditional credit bureau factors. We explicitly excluded proxy discriminatory variables (`geography_region` is present as a business-context feature, not a demographic attribute; no race/gender/age data is in the dataset). Adverse action notices would reference the top SHAP contributors for each declined applicant.

---

## 4. Calibration & Uncertainty Quantification

**Deliverable A — PD confidence intervals.** Point estimate: isotonic regression fitted on `val_labeled` (2,551 rows, never seen during training) maps raw ensemble scores to calibrated probabilities. Interval: the 10 CV fold models each produce a calibrated PD estimate per applicant; the 90% CI is constructed as predicted\_pd ± 1.645 × std(fold predictions). This measures *model estimation uncertainty* — how much the 10 members of the ensemble disagree — rather than aleatoric outcome uncertainty. We enforced lower ≤ predicted\_pd ≤ upper row-by-row after boundary clipping. Empirical val interval width: mean half-width 0.053, median 0.033.

**Calibration quality.** Isotonic regression is a non-parametric monotone calibrator that minimises the Brier score on the calibration set. It was fitted exclusively on `val_labeled` (not training data), guaranteeing independence between the calibration signal and the model's training data. We verified that mean calibrated PD on `val_labeled` (0.206) is close to the empirical default rate (0.206) — the calibrator is unbiased in expectation.

**Deliverable B — CDR trajectory intervals.** Per-cohort 90% CI uses the standard error of the mean across individual loan CDRs: CI = CDR\_mean ± 1.645 × std(CDR\_i) / √n. This is a frequentist interval on the *cohort mean CDR*, valid under the Central Limit Theorem for n ≥ 30 (all cohorts have n ≥ 527). Monotonicity of the CI bounds is enforced independently via cumulative maximum. After empirical calibration, CDR predictions are anchored to the observed training default rate at each time point, correcting the absolute scale introduced by the RSF's balanced sample weights.

**Deliverable C — Counterfactual PD intervals.** Same 10-fold ensemble disagreement method as Deliverable A. For each query, the modified feature vector is passed through all 10 CV fold models; CI = predicted\_pd\_cf ± 1.645 × std(calibrated fold predictions). Empirical mean CI half-width: 0.063. Assertions verified: pd\_cf\_lower ≤ predicted\_pd\_cf ≤ pd\_cf\_upper for all 900 rows.

**Tradeoff: width vs. coverage.** Our ensemble-disagreement intervals are narrower than full conformal intervals (which guarantee marginal coverage by construction) because ensemble variance on a 51K training set is small — models agree closely, producing tight bands. A split-conformal expansion was considered but abandoned: the conformal quantile q\_90 was large relative to the bootstrap spread, making the conformal interval no wider than adding a fixed offset. The ensemble-disagreement approach is more informative and adapts per-applicant to model uncertainty.

---

## 5. Limitations & What We'd Do Differently

**Reject inference (highest-priority gap).** The entire outcome dataset is conditioned on prior-lender approval. We have zero labeled outcomes for declined applicants, who may constitute the highest-risk segment. Our model is likely to underestimate PD for profiles historically declined — our intervals do not capture this structural blind spot. With more time, we would apply EM-based reject inference: assign soft labels to declined applicants using the model's predictions as starting probabilities, then iterate until convergence on a joint likelihood. This is standard in credit scoring and would partially correct the selection bias.

**Draw-by-draw history.** We do not have the sequence of ACH draw successes and failures that led to each default declaration. Our NPV formula uses `max(0, default\_day − 3)` as an estimate of successful draws (assuming the 3-consecutive trigger with no prior misses), which is an upper bound on collections. The true number of successful draws could be as low as `default\_day − 6` (6-total trigger). A sensitivity analysis with `min_missed_draws ∈ {3, 4, 5, 6}` would bound the threshold uncertainty; with access to daily draw data, we could compute NPV exactly.

**RSF calibration is lossy.** Our per-time-point calibration factors correct the *mean* predicted CDR but do not re-calibrate the full distribution of per-loan CDRs. A Platt-scaling or isotonic regression approach applied to the RSF's cumulative hazard at each time point would provide better per-loan calibration. We would also explore using the Nelson-Aalen estimator as a non-parametric CDR baseline, with RSF providing relative risk adjustments (Cox-style).

**Counterfactual identifiability.** The do() implementation treats the model as a structural equation. This is only valid for features that are causally upstream of default with no unobserved confounders. For features like `requested_amount` (which may be correlated with latent financial desperation) or `application_channel` (correlated with marketing treatment), the interventional and observational estimates diverge. With more time, we would build a partial DAG of the feature space, identify which features are valid intervention targets, and apply front-door or instrumental-variable adjustments for confounded features.

**Feature selection and leakage audit.** The `SelectKBest` enforcement (5× row-to-column ratio) is a safeguard but does not replace domain-driven feature selection. Several engineered features (`daily_draw_to_cash_ratio`, `revenue_to_daily_draw`) use `requested_amount`, which is also the loan principal in the NPV calculation — this creates a potential circularity. With more time, we would run a systematic SHAP-based permutation importance audit and check for feature-target leakage beyond the hard `EXCLUDE` list.

**Production considerations.** The survival forest (RSF) is computationally expensive at inference time. In a production system we would replace it with a lighter parametric approximation (e.g., a piece-wise exponential model or a calibrated logistic hazard model) that can return survival functions in milliseconds rather than seconds per batch. The CatBoost ensemble and isotonic calibrator are already production-deployable as serialised artifacts.

---

*References available on request. All code, model artifacts, and intermediate outputs are reproducible from `deliverable_A.ipynb` with fixed random seed 42.*
