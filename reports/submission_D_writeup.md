# Deliverable D - Technical Writeup

**Team:** Global Intuit Hackers

## 1. Problem framing & assumptions violated

We act as a small-business lender choosing whom to fund to **maximize realized portfolio value**, while also forecasting cohort default trajectories (B), answering interventional what-ifs (C), and quantifying uncertainty. The dataset deliberately breaks several standard ML assumptions, and each broke a different part of our pipeline:

- **No positivity / deterministic selection (the dominant trap).** Repayment outcomes exist *only* for prior-approved, matured loans, and prior approval is **deterministic** in `prior_underwriter_score`: declined applicants occupy scores [0, 0.273], approved occupy [0.273, 1.0], with **zero overlap**. The IID/representative-sample assumption fails, and—critically—inverse-propensity reject inference is *mathematically void*: `P(approved|X)=1` on every labeled row and `1/0` in the declined region. We confirmed the cutoff empirically (≈0.273, 99.6% threshold accuracy, 0 overlap rows).
- **Outcome leakage.** `default_flag, days_to_default, days_to_full_repayment, repayment_status, final_recovered_amount, observation_status` are post-outcome and blank in test; we quarantine them as features and use them only to define targets.
- **MNAR missingness.** Bank-feed columns are null exactly when no feed is linked, and that is informative (observed default 19.1% no-feed vs 16.7% with-feed). We add explicit missingness indicators rather than imputing.
- **Self-report inflation.** `stated_*` fields are optimistically biased vs the observed bank-feed counterpart; the stated/observed *gap* is predictive (default 8%→28% across its quintiles) but the stated value itself has ~0 causal effect on repayment—central to Deliverable C.
- **Right-censoring + drift.** The labeled default rate drifts **15.2%→21.1%** across the 18-month history; late cohorts are under-observed. B requires survival methods, and calibration must track the recent regime.
- **Planted economics trap.** Taking `final_recovered_amount` as the only recovery implies LGD≈0.91 and a ~9% break-even, which rejects a wide band of profitable loans. But loans **amortize** via daily ACH draws, so a default at the mean day (≈43 of 60) has already repaid most principal—true realized LGD≈0.25.

## 2. Methodology

**Shared pipeline.** One feature frame for all deliverables: drop leakage/ids/raw timestamp, add MNAR missingness indicators, keep NaN native (HistGradientBoosting handles it), integer-coded categoricals declared as categorical.

**Deliverable A — PD + decision.** The PD model is a **calibrated blend, 0.4·(25-model bootstrap HistGBT) + 0.6·(L2 logistic)**: the signal is largely additive, so the logistic out-ranks the tree, and the blend lifts honest 5-fold OOF AUC 0.774→0.777 (positive in all 5 folds). We keep `prior_underwriter_score` (a strong predictor; it is only a collider for *causal* questions, §3). Isotonic calibration on validation gives OOF ECE 0.0018. The decision uses the **brief's exact amortizing NPV**: repaid `= F + R·r·T/365`; default at day t* `= F + D·(t*−1) + rec − R`. The measured effective LGD is **0.25**, so break-even PD `= 0.0875/(0.0875+0.25) ≈ 0.259`—cross-validated (train OOF 0.265, validation 0.250 bracket it; *not* tuned to validation profit, which we showed swings 0.19–0.32 and overfits). We fund the **observed-support region only** (prior-approved, PD<0.259): the positivity violation makes declined-region funding unverifiable, and under the empirical default floor "gated/trust" expansions fall *below* the conservative book while risking $59–87M of principal. Conservative expected NPV (~$6.3M) is invariant to the unobservable.

**Deliverable B — cohort trajectory (Markov chain).** The default definition *is* an absorbing Markov chain over delinquency states (3 consecutive / 6 cumulative missed daily draws / positive balance at day 90); `days_to_default` is the observed **absorption day**. We model its discrete-time absorption-time CDF `G(a)` and factor `CDR(cohort c, age a) = (mean PD of A-approved loans in c) × G(a)`—justified because the default-day *shape* is ~proportional across risk strata while the *level* (PD) differs. This reproduces the empirical trajectory at **MAE 0.0056** with the correct, non-trivial shape (near-zero weeks 1–2 from the 3-miss floor; a plateau weeks 9–12 from the days 61–89 dead zone; the day-90 balance-check jump), beating a naive "scale-the-final-rate" baseline (MAE 0.0122).

**Deliverable C — counterfactuals.** A backdoor-restricted model-perturbation g-formula (full detail in §3).

**How they fit together.** One calibrated PD model (A) sets each loan's total default mass; the absorbing-chain timing (B) shapes *when* that mass lands; the backdoor-restricted model (C) answers interventions. A's approved set defines B's cohorts, giving a coherent A→B by construction.

## 3. Causal reasoning & counterfactual methodology

Observational prediction conditions on everything correlated with default; an **interventional** estimate `P(default | do(F=v))` must block backdoor paths. Our C model is a **backdoor-restricted, monotone-constrained model-perturbation g-formula**, built on the same modeling spine as A but on a deliberately different feature frame.

**Backdoor / collider removal.** We drop the three selection nodes—`prior_underwriter_score` (the deterministic selection score), `prior_decision` (the collider), `prior_approved_amount` (a prior-decision output)—because conditioning on any of them opens the path `score → [decision] ← X` and biases every interventional estimate. They are valuable *predictors* (kept in A) but invalid for *do()*. The causal frame keeps 37 features at val AUC 0.756 vs A's 0.758—a negligible cost for backdoor validity.

**Self-report trap, enforced structurally.** `stated_annual_revenue`, `stated_time_in_business`, and `intended_use_of_funds` are application statements, not manipulable business state; their true causal effect is ~0 (the bank-*observed* counterpart drives repayment). We **exclude** them, so `do(stated_*)` returns *exactly* the baseline PD (verified delta 0.000000). This is stricter and more defensible than a shrink factor.

**The g-formula for deterministic descendants.** For `do(F=v)` we set the raw parent and **recompute** the engineered leverage `= requested_amount/(observed_monthly_revenue×12)` (we replace the dataset's source-mixed ratio with a clean observed-only one we control), then re-predict. `requested_amount` is a genuine lever (size→leverage→default), not a zero-effect self-report.

**Why a naive multivariate counterfactual is wrong, and our fix.** `existing_debt_obligations` has a strongly *positive* raw marginal (default 10.5%→25.9% across quintiles) but a *negative* multivariate partial, because it is collinear with `aggregate_credit_utilization` (corr 0.63). A naive "change the input, re-predict" model therefore returns the **wrong interventional sign** for `do(debt↑)`. We inject the unambiguous economic signs **structurally**—HistGBT `monotonic_cst` plus a sign-bounded logistic arm—on the 15 monotone drivers, fixing the sign for −0.002 AUC. All eight adversarial sign checks pass (`do(utilization/debt/delinquency/overdraft ↑)→PD↑`; `do(cash_balance↑)→PD↓`; `do(stated_*)→0`).

**What we give up (honestly).** This is model-perturbation, not a fitted structural causal model: it is causal only under our backdoor-sufficiency and descendant-recompute assumptions. We guarantee the *sign* of each effect, not its *magnitude*—a constrained lever can under-respond, and rather than fabricate magnitude with hand-tuned shrink factors, we accept conservative (sometimes ~0) effects on weak/collinear drivers. 200 of the 300 queried applicants sit in the never-labeled declined region or have no bank feed, so those counterfactuals are extrapolations—we widen their intervals 2.5×. **Regulator defense:** our stated drivers are observed, monotone, and intuitive (utilization, debt burden, invoice delinquency, cash buffer), never self-reported or selection-inherited.

## 4. Calibration & uncertainty quantification

- **A (PD intervals).** Isotonic-calibrated on held-out validation; the 90% interval is the bootstrap-ensemble 5th/95th percentile of the PD, validated at **10/10 decile-coverage** with mean width ≈0.12. We *rejected* a tighter conformal band: it held 90% coverage in-sample only—honest 5-fold OOF coverage collapsed to 38–50%—so we chose honest width over false tightness.
- **B (trajectory intervals).** Predictive bands combine (1) per-loan PD ensemble uncertainty on the cohort mean, (2) bootstrap of the absorption-time curve, and (3) **binomial sampling of each cohort's approved count**—the correct object for a realized *fraction*, avoiding the 1/√n CI collapse. Mean width ≈0.039.
- **C (counterfactual intervals).** Bootstrap ensemble of the causal model, widened additively in the never-labeled declined region (positivity-limited, flagged as uncalibrated there).
- **Tradeoff.** Throughout we prioritize *honest* coverage over narrow width, validated out-of-fold (A on held-out validation, calibration cross-fit), because an overconfident band fails `S_cal` worse on the drifting test cohort than an honestly wide one.

## 5. Limitations & what we'd do differently

- **Declined-region positivity wall.** We cannot verify any funding decision in the never-labeled region, so we abstain. If the scorer credits model-extrapolated declined-region NPV, our conservative book leaves expected upside on the table—a deliberate, risk-adjusted choice, not an oversight (the same expansions go *negative* under the empirical default floor).
- **Capped vs uncapped NPV.** The brief's literal default formula credits late defaults (t*>60, 22.5% of defaults) *above* full repayment; we cap at the repaid margin (conservative). The uncapped reading would raise break-even to ≈0.28 and fund more—we would confirm the scorer's convention with the organizer.
- **AUC ceiling.** Honest OOF AUC plateaus at ≈0.775; the signal is largely linear and added model complexity, feature engineering, and alternative GBTs gave no generalizing gain. We are confident this is near the data's information limit, not a modeling shortfall.
- **C is model-perturbation, not full identification.** With another day we would write the explicit DAG, add a no-prior-score A variant for defensibility, and attach sensitivity bounds to the counterfactuals rather than point estimates.

<!-- References (optional; do not count toward the 4-page body limit). -->
