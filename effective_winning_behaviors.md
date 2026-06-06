# Effective Winning Behaviors for the Intuit ML Hackathon

## Core Thesis

Winning is not mainly about running the most experiments. It is about building a solution that is hard to penalize across the full scoring surface:

```text
Profitable underwriting policy
+ coherent default timing forecast
+ cautious causal counterfactuals
+ calibrated uncertainty
+ validator-clean engineering
+ defensible writeup
```

This challenge should be treated as a combined **underwriting policy + survival forecasting + causal inference + uncertainty calibration** problem, not a vanilla classification contest.

---

## 1. Optimize the Actual Business Objective

Deliverable A is not just a probability prediction task. It asks who should be funded and what their probability of default is.

A strong team converts PD into a funding decision using expected value:

```text
expected profit = fees + interest - expected credit loss
```

Key idea:

- Do not approve just because `PD < 0.5`.
- The break-even PD is likely much lower than 50%.
- Funding decisions should account for requested amount, expected loss, default timing, recovery assumptions, and uncertainty.

Winning behavior:

```text
Approve only when expected profit is positive after risk controls.
```

---

## 2. Use Risk-Adjusted Approval, Not Pure Expected Value

Because underwriting involves downside risk, a pure average-profit strategy can be fragile.

Use a risk-adjusted rule:

```text
Approve if:
  expected profit > threshold
  AND PD_upper_90 is acceptable
  AND segment support is sufficient
  AND exposure concentration is controlled
  AND timing/default curve is not fragile
```

Practical controls:

- Use point PD for expected value.
- Use upper-bound PD as a safety brake.
- Penalize large requested amounts when uncertainty is high.
- Cap approval concentration in sparse segments.
- Avoid over-approving rows where the model is extrapolating.

---

## 3. Anticipate Black-Swan-Like Failure Modes

The hidden risk is probably not a literal macroeconomic black swan. It is more likely a scoring or data-tail issue.

Possible black-swan-like failures:

### Sparse segment blowup

A small niche segment looks safe but defaults heavily because the training data is thin or biased.

### Hidden scoring-weight fragility

The exact scoring weights are unpublished. A team that over-optimizes A may get punished by weak B, C, calibration, or writeup quality.

### Cohort shift

A model validated on random splits may fail when evaluated on later cohorts.

### Counterfactual extrapolation

Deliverable C asks for `do(feature=value)` causal counterfactuals. Naively changing one feature in a predictive model can produce unsupported, overconfident results.

### Tail-risk applicants

Some applicants may look profitable on average but have high downside exposure because of large requested amounts, sparse segment history, or volatile repayment signals.

Winning behavior:

```text
Maximize robust expected value, not raw expected value.
```

---

## 4. Make Deliverables A and B Coherent

Deliverable B should forecast cumulative default timing for the loans **your policy approves**.

Weak approach:

```text
Train A independently.
Train B independently.
Submit both.
```

Strong approach:

```text
Estimate individual PD / hazard curves.
Apply the Deliverable A approval policy.
Aggregate approved loans into cohort-level cumulative default curves.
Enforce non-decreasing trajectories.
```

This matters because B is not the market default curve. It is the default curve of the portfolio you create.

---

## 5. Treat Calibration as a Scored Product Feature

This challenge requires point estimates plus lower/upper uncertainty intervals.

A strong submission should not submit arbitrary bands like `PD ± 0.05`.

Better approaches:

- out-of-fold calibration
- isotonic calibration
- Platt / logistic calibration
- beta calibration if available
- bootstrap uncertainty
- conformal-style interval checks
- segment-level calibration diagnostics

Winning behavior:

```text
Calibrate probabilities and intervals on validation data before final submission.
```

---

## 6. Do Not Confuse Prediction with Causality

Deliverable C asks for counterfactual PD under an intervention:

```text
do(feature = value)
```

This is not the same as:

```text
model.predict(row with one feature changed)
```

Strong behavior:

- Separate observational prediction from causal interpretation.
- Identify which features are plausibly intervenable.
- Treat historical/proxy variables cautiously.
- Check support before trusting counterfactuals.
- Widen intervals when the intervention is outside observed support.
- Explain causal assumptions directly in the writeup.

Practical hackathon compromise:

```text
Use the predictive model as a baseline, but apply causal guardrails, support checks, shrinkage, and uncertainty widening.
```

---

## 7. Respect Selection Bias and Censoring

Historical outcomes are not automatically representative of all applicants. They may be observed primarily for loans that were previously approved and matured.

Risk:

```text
The model learns default risk among historically approved borrowers,
not default risk among all applicants we might fund.
```

Winning behavior:

- Model prior approval / prior decision effects.
- Check whether labels are missing systematically.
- Use missingness indicators.
- Avoid assuming rejected applicants are missing at random.
- Discuss reject inference and censoring limitations in the writeup.

---

## 8. Use Time-Aware Validation

Random splits can make the model look stronger than it is.

This challenge includes cohort timing and future default trajectories, so temporal leakage and cohort shift matter.

Better validation:

```text
Train on earlier cohorts.
Validate on later cohorts.
Check calibration by cohort week.
Stress-test default timing curves.
```

Winning behavior:

```text
Prefer validation that resembles the hidden evaluation setup over validation that gives the prettiest metric.
```

---

## 9. Preserve Informative Missingness

Missing values may be signal, especially bank-feed fields or fields only present under certain applicant behaviors.

Weak approach:

```text
Median-impute everything and move on.
```

Strong approach:

- Add missingness indicators.
- Model linked-bank-feed vs unlinked applicants separately if useful.
- Let CatBoost / LightGBM handle missingness where appropriate.
- Test whether missingness itself predicts default.

Winning behavior:

```text
Treat the measurement process as part of the signal.
```

---

## 10. Build Validator-Clean Submission Engineering

A great model that fails the validator gets no value.

Operational checklist:

- Correct filenames.
- Complete ID coverage.
- Valid probability ranges.
- Lower ≤ point ≤ upper.
- Non-decreasing cumulative default curves.
- No missing required rows.
- Deterministic generation pipeline.
- Final `validate_submission.py` pass.

Winning behavior:

```text
Automate the final submission build and validation.
```

---

## 11. Write Like a Model-Risk Reviewer Is Reading

The writeup is not just a formality. It is where teams can distinguish themselves on causal reasoning, assumptions, uncertainty, and limitations.

Strong writeup traits:

- States the business objective clearly.
- Explains why the approval policy maximizes risk-adjusted profit.
- Distinguishes prediction from causality.
- Explains uncertainty calibration.
- Discusses selection bias, censoring, and cohort shift.
- Shows limitations honestly.
- Avoids overclaiming causal certainty.

Winning behavior:

```text
Make the writeup a technical defense of the modeling system, not a generic summary.
```

---

## Why Exact Weights Being Unpublished Matters

Unpublished weights prevent teams from overfitting the scoring formula.

Practical implication:

```text
Do not optimize narrowly for one deliverable.
Build a balanced solution that is strong enough across A, B, C, calibration, and D.
```

Bad strategy:

```text
Maximize one metric and ignore the rest.
```

Better strategy:

```text
Avoid catastrophic weakness anywhere.
```

---

## Final Winning Pattern

The best teams usually do this:

```text
1. Understand the scoring surface.
2. Model the data-generating process.
3. Build a calibrated PD model.
4. Convert PD into risk-adjusted funding decisions.
5. Model default timing for approved loans.
6. Generate cautious causal counterfactuals.
7. Validate by time/cohort.
8. Stress-test sparse segments and downside risk.
9. Pass the validator cleanly.
10. Defend assumptions in the writeup.
```

The goal is not the fanciest model.

The goal is a submission that is:

```text
profitable,
calibrated,
coherent,
causally cautious,
robust to hidden scoring weights,
and impossible to disqualify on formatting.
```
