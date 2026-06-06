---
title: "Deliverable D — Technical Writeup"
subtitle: "SMB Loan Underwriting Challenge"
author: "Abhimanyu"
date: "June 2026"
geometry: "top=0.9in, bottom=0.9in, left=0.85in, right=0.85in"
fontsize: 11pt
papersize: letter
mainfont: "Palatino"
monofont: "Courier"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{xcolor}
  - \usepackage{float}
  - \let\origfigure\figure
  - \let\endorigfigure\endfigure
  - \renewenvironment{figure}[1][H]{\origfigure[H]}{\endorigfigure}
  - \setlength{\parindent}{0pt}
  - \setlength{\parskip}{5pt}
  - \setlength{\topsep}{2pt}
  - \setlength{\itemsep}{1pt}
---

## 1. Problem Framing & Assumptions Violated

**What we were solving.** For 13,306 applicants across validation and test sets we produced: a calibrated probability of default (PD), a 90% confidence interval on that PD, and an approval decision grounded in loan economics. Two auxiliary deliverables required a 13-week cumulative default rate (CDR) trajectory per origination cohort, and causal what-if PD estimates under single-feature interventions.

**Standard ML assumptions this dataset breaks, and how we responded.**

*Selection bias / reject inference.* The labeled training set contains only approved, matured loans — applicants the prior lender declined have no observed outcome. This is a textbook selection-on-treatment problem. We partially mitigated it by including `prior_underwriter_score` and `has_prior_approval` as features, allowing the model to self-correct for profiles historically gated. Full EM-based reject-inference imputation was not applied due to the absence of a reliable prior PD for declined applicants.

*MNAR bank-feed missingness.* Six bank-feed columns are null when no linked bank feed exists — absence is informative, not random. We added a binary `{col}__missing` indicator for each, allowing tree models to learn distinct behaviour for applicants with and without linked accounts. Standard imputation alone would collapse this signal.

*Survivorship bias in survival data.* The RSF training set is 82.5% right-censored at exactly day 60: every repaid loan shares the same censoring time. This pathological structure causes the RSF's cumulative hazard estimator to assign near-zero hazard to the interior of the term — $S(t)$ barely drops below 1.0 for most applicants, with median survival $= 60$ for $\approx$75% of the portfolio. Two targeted interventions address this (see §2).

*Collider bias from `prior_decision`.* The prior lender's approval decision is a collider: including it opens a backdoor path from unobserved confounders (e.g., applicant desperation) to our outcome. It was excluded alongside all raw outcome columns.

*Default label complexity.* `days_to_default` records the day default *status was declared* — triggered by 3 consecutive missed ACH draws, 6 total missed draws, or outstanding balance at day 90. The label does not encode how many draws were actually collected. At minimum 3 draws were missed (3-consecutive trigger). Our NPV formula corrects for this: successful draws $= \max(0,\ \text{default\_day} - 3)$, shifting the break-even profitability day from $\approx$52 to $\approx$54.

---

## 2. Methodology

**Pipeline overview.** The pipeline is fully deterministic given fixed random seed 42. All model fitting uses training data only; calibration uses an independent held-out set (`val_labeled`, 2,551 rows with known outcomes) never seen during training.

![Underwriting pipeline: raw data flows through feature engineering into the CatBoost ensemble and isotonic calibrator; the RSF branches off to produce CDR trajectories; the NPV model converts calibrated PD into approval decisions.](figures/fig_pipeline.png){ width=100% }

**Feature engineering (50 features).** Raw features (44 numeric, 6 categorical) are augmented with: 6 MNAR indicators; `has_prior_approval`, `approved_to_requested` (prior lender signal); `prior_default_rate` (defaults / prior loan count); `loan_to_annual_rev` (leverage ratio); `has_external_decline`, `has_inquiry_elsewhere` (soft-pull signals); `daily_draw_to_cash_ratio` (daily obligation vs. P10 cash buffer); `revenue_to_daily_draw` (revenue coverage); and `platform_stress` (delinquency rate $\times$ overdraft frequency). A row-to-column ratio check enforces $n \geq 5k$ before any model fit, with `SelectKBest(f_classif)` reducing features automatically if violated.

**Deliverable A — PD model.** A 10-fold stratified CatBoost ensemble (each member trained on a different 9/10 subset, early-stopped on its held-out fold) produces raw probability scores. Isotonic regression fitted on `val_labeled` converts them to calibrated probabilities. OOF AUC: **0.776**. CatBoost was chosen for native categorical support, NaN tolerance, and well-ranked probabilities with minimal tuning.

**Break-even decision threshold.** For each applicant we compute the PD at which expected NPV equals zero:
$$\text{threshold} = \frac{\text{NPV}_\text{repaid}}{\text{NPV}_\text{repaid} - \text{NPV}_\text{default}}$$
$\text{NPV}_\text{repaid}$ captures origination fee plus interest. $\text{NPV}_\text{default}$ uses: fee $+$ daily\_draw $\times \max(0,\ E[T|\text{default}] - 3)$ $+$ recovery $-$ principal. The default day is the *conditional expected default day* $E[T \mid T \leq 60]$ from the RSF survival function, which conditions on default occurring rather than using the biased unconditional median. Recovery is predicted by a CatBoostRegressor (hold-out RMSE: 0.090). When $\text{NPV}_\text{default} \geq \text{NPV}_\text{repaid}$ (default day $\geq$54, where late defaults are profitable), threshold $= 1.0$. This yields a val decline rate of **35.6%**.

![NPV as a function of default day for a representative \$25,000 loan with zero recovery. The corrected formula (v2, day$-$3) shifts break-even from day 52 to day 54; loans defaulting after day 54 are net profitable even in the worst case.](figures/fig_npv_breakeven.png){ width=88% }

**Deliverable B — CDR trajectory.** A second RSF (`rsf_b`, 91-day horizon) produces per-loan survival functions at 13 weekly checkpoints. $\text{CDR}_i(t) = 1 - S_i(t)$. Approved loans are grouped by origination cohort; cohort CDR $=$ mean across members; 90% CI $= \pm 1.645 \times \sigma/\sqrt{n}$; monotonicity enforced via cumulative maximum. Predictions are calibrated against empirical training CDR: $\text{cal\_factor}(t) = \text{empirical\_CDR}(t) / \text{mean\_pred\_CDR}(t)$, correcting the absolute scale while preserving relative loan rankings.

![CDR trajectories for all origination cohorts (colour = cohort week, shaded band = 90% CI). Later cohorts trend higher, consistent with macroeconomic seasoning effects. Monotonicity is enforced via cumulative maximum.](figures/fig_cdr_trajectories.png){ width=88% }

**RSF survivorship bias fix.** Both RSF models use balanced sample weights: defaulted rows receive weight $\approx$2.87$\times$, censored rows 0.61$\times$. This equalises each group's log-rank split contribution, spreading learned hazard across the full 1–60 day range. Concordance index: **0.857**.

---

## 3. Causal Reasoning & Counterfactual Methodology

**Observational vs. interventional prediction.** Standard supervised learning estimates $P(Y \mid X = x)$ — the conditional distribution of default given observed covariates. Deliverable C asks $P(Y \mid do(X_j = v))$ — what would happen if we *set* feature $j$ to value $v$ by intervention, holding all else fixed. These differ whenever $X_j$ has confounders outside $X$. The $do(\cdot)$ operator from Pearl's do-calculus severs the incoming edges into the intervened node, removing its association with its causes while preserving its effects on $Y$.

**Implementation.** For each of the 900 intervention queries:

1. Copy the applicant's original raw test row.
2. Set the intervened feature to its intervention value, preserving correct dtype.
3. Re-run `engineer_features()` in full — this recomputes all derived quantities (`prior_default_rate`, `loan_to_annual_rev`, MNAR indicators, stress features, etc.) from the modified raw values, so downstream effects cascade automatically through the feature graph.
4. Predict with the frozen 10-fold ensemble and isotonic calibrator.

This is a g-computation (model-perturbation) approach. It is appropriate under the assumption that the ensemble approximates the true conditional outcome model and that the intervened feature is causally upstream of the outcome.

**What we give up.** We do not have a structural causal model or instrument. If the intervened feature has unobserved common causes with the outcome, our estimate is biased interventionally. For example, `requested_amount` may proxy for latent financial desperation; the model sees the feature change but cannot account for the simultaneous shift in the latent confounder. This is an accepted limitation of the model-perturbation approach with observational data.

**Regulatory defensibility.** SHAP values from the CatBoost ensemble provide locally consistent, feature-level attribution. The top predictors — `prior_underwriter_score`, `owner_personal_credit_band`, `prior_default_rate`, `loan_to_annual_rev`, `daily_draw_to_cash_ratio` — are economically interpretable and analogous to traditional credit bureau factors. No demographic proxies are present. Adverse action notices would cite the top SHAP contributors per declined applicant.

---

## 4. Calibration & Uncertainty Quantification

**Deliverable A — PD confidence intervals.** Point estimate: isotonic regression on `val_labeled` (independent of training) maps raw ensemble scores to calibrated probabilities. Interval: the 10 CV fold models each produce a calibrated PD; the 90% CI is $\hat{p} \pm 1.645 \times \text{std(fold predictions)}$. This measures *model estimation uncertainty* — ensemble disagreement — rather than aleatoric outcome uncertainty. We enforced lower $\leq \hat{p} \leq$ upper row-by-row after clipping. Mean CI half-width: **0.053** (val), **0.054** (test).

**Calibration quality.** Isotonic regression minimises the Brier score on the calibration set, fitted exclusively on `val_labeled`. Mean calibrated PD on `val_labeled` (0.206) matches the empirical default rate (0.206) — the calibrator is unbiased in expectation.

![Left: PD distribution for approved vs declined applicants — the decision boundary cleanly separates high-PD declined loans from lower-PD approvals. Right: CI width vs predicted PD — uncertainty peaks near PD $\approx$ 0.3–0.5 where ensemble disagreement is highest.](figures/fig_pd_distribution.png){ width=100% }

**Deliverable B — CDR trajectory intervals.** Per-cohort 90% CI: $\bar{\text{CDR}} \pm 1.645 \times \sigma / \sqrt{n}$ — a frequentist interval on the cohort mean CDR, valid under the CLT (all cohorts have $n \geq 527$). Monotonicity of CI bounds enforced independently via cumulative maximum. After empirical calibration the population-mean CDR is anchored to the observed training default rate at each time point.

**Deliverable C — Counterfactual PD intervals.** Same 10-fold ensemble disagreement method as Deliverable A. Each modified feature vector passes through all 10 fold models; CI $= \hat{p}_\text{cf} \pm 1.645 \times \text{std(calibrated fold predictions)}$. Mean half-width: **0.063**. Assertion verified: lower $\leq \hat{p}_\text{cf} \leq$ upper for all 900 rows.

**Width vs. coverage tradeoff.** Ensemble-disagreement intervals are narrower than full conformal intervals because variance on a 51K training set is small. A split-conformal expansion was evaluated but the conformal quantile $q_{90}$ dwarfed the bootstrap spread, making conformal no better than a fixed offset. The ensemble-disagreement approach is more informative and adapts per-applicant to model uncertainty.

---

## 5. Limitations & What We'd Do Differently

**Reject inference (highest-priority gap).** The entire outcome dataset is conditioned on prior-lender approval. Zero labeled outcomes exist for declined applicants, who may constitute the highest-risk segment. Our model likely underestimates PD for historically-declined profiles, and our intervals do not capture this structural blind spot. With more time, we would apply EM-based reject inference: assign soft labels to declined applicants using model predictions as starting probabilities, then iterate on the joint likelihood until convergence — standard practice in production credit scoring.

**Draw-by-draw history.** We lack the ACH draw sequence leading to each default declaration. Our NPV formula uses $\max(0,\ \text{default\_day} - 3)$ as an upper bound on successful draws (3-consecutive trigger, no prior misses). The true count could be as low as $\text{default\_day} - 6$ (6-total trigger). A sensitivity analysis over `min_missed_draws` $\in \{3, 4, 5, 6\}$ would bound threshold uncertainty; daily draw data would allow exact NPV computation.

**RSF calibration is lossy.** Our per-time-point calibration factors correct the *mean* predicted CDR but do not re-calibrate the full per-loan distribution. Isotonic regression applied to the RSF's cumulative hazard at each time point would provide better per-loan calibration. We would also explore the Nelson-Aalen estimator as a non-parametric CDR baseline, with the RSF providing relative risk adjustments (Cox-style).

**Counterfactual identifiability.** The $do(\cdot)$ implementation assumes causally upstream features with no unobserved confounders. Features like `requested_amount` or `application_channel` have plausible confounders. With more time, we would build a partial DAG of the feature space and apply front-door or instrumental-variable adjustments for confounded features.

**Production considerations.** The RSF is computationally expensive at inference time ($\approx$200 trees, survival function evaluation per applicant). In production, we would replace it with a parametric approximation (piece-wise exponential model) returning survival functions in milliseconds. The CatBoost ensemble and isotonic calibrator are already production-deployable as serialised artifacts.

---

*All code, model artifacts, and intermediate outputs are fully reproducible from* `deliverable_A.ipynb` *with random seed 42.*
