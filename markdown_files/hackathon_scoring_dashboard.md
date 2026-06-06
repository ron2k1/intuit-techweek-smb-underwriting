# Hackathon Scoring Dashboard and Comparison Table

_Last updated: 2026-06-05_

This table replaces the older screenshot-style comparison as the main operating dashboard. The screenshot factors are still useful for detecting the LGD/recovery trap and prior-lender selection trap, but the judges score the submission on five weighted components:

```text
S = 0.30 S_P&L + 0.25 S_traj + 0.20 S_cal + 0.10 S_C + 0.15 S_write
```

The practical implication is that a high-AUROC or high-headline-NPV model can still lose if C is missing, B is not rebuilt after A changes, intervals are uncalibrated, or the writeup does not defend the causal assumptions.

## Executive Scorecard

| Score component | Weight | What judges care about | Current proxy metric | Current status | Biggest risk | Next action |
|---|---:|---|---:|---|---|---|
| `S_P&L` | 30% | Realized portfolio value from A decisions | Active labeled-val NPV: `$3.868M`; 9,005 approved; 3,104 prior-declined approvals | Strong but not final | Prior-declined region has no labels, so headline NPV can be optimistic | Keep LightGBM/no-prior-score active unless sensitivity audit argues against it |
| `S_traj` | 25% | Accuracy of B cumulative default trajectories on our approved set | Active B CDR MAE: `0.0150`; week-13 mean pred `0.1383` vs actual `0.1327` | Good, rebuilt for current A | Cohort 13 tail is overpredicted | Reduce cohort 13 overprediction or widen interval if time |
| `S_cal` | 20% | 90% intervals on A PD and B trajectories contain truth without being too wide | Active A AUROC `0.744`, log loss `0.436`, Brier `0.137`; PD interval bin coverage `0.900` | Partially ready | AUROC is lower than the old blend; calibration/interval defense matters more | Report coverage by PD bin, prior-decision stratum, cohort week, and approved set |
| `S_C` | 10% | Counterfactual PDs match true intervention effects, not naive re-prediction | `submission_C_counterfactuals.csv` missing | Blocking | Validator fails; cannot score complete package | Build C next using DAG-safe intervention handling and support checks |
| `S_write` | 15% | Clear methodological defense | `submission_D_writeup.pdf` missing | Blocking for final score | Good modeling choices are invisible without the defense | Write 4-page D around selection, economics, B timing, C causality, and intervals |

## Submission Readiness

| Deliverable | Required artifact | Current artifact | Completeness | Readiness metric | Current value | Score impact |
|---|---|---|---:|---|---:|---|
| A | `submission_A_decisions.csv` | Present | 100% file-ready | Approved applicants | 9,005 | Drives `S_P&L`, B denominator, and A interval calibration |
| A | PD and 90% PI columns | Present | Needs final stratified interval audit | AUROC / log loss / Brier | `0.744 / 0.436 / 0.137` | Drives `S_cal`; not directly enough for `S_P&L` |
| B | `submission_B_trajectory.csv` | Present | 100% file-ready for current A | Cohort-age rows | 169 | Drives `S_traj` and B interval calibration |
| B | CDR forecasts and intervals | Present | Needs final coverage table | CDR MAE | `0.0150` | Strong enough to keep unless A changes |
| C | `submission_C_counterfactuals.csv` | Missing | 0% file-ready | Query coverage | 0 / 900 | Hard blocker; current validator fails |
| D | `submission_D_writeup.pdf` | Missing | 0% file-ready | 4-page writeup | Missing | 15% direct score plus defense of all modeling choices |
| Package | Flat submission folder | A/B only | Fails validator | `validate_submission.py` | 1 error, 1 warning | Cannot be final-submitted as-is |

## Model Performance Metrics to Track

| Area | Primary metric | Secondary metrics | Why it matters | Do not over-index on |
|---|---|---|---|---|
| A portfolio value | Labeled-validation realized NPV under the exact brief cash-flow formula | Headline expected NPV, approval rate, approved default rate, NPV by cohort, NPV by prior-decision stratum | This is the closest local proxy for `S_P&L` | Flat PD threshold, F1, accuracy |
| A PD quality | Log loss and Brier score | AUROC, average precision, calibration slope/intercept, ECE, calibration bins | PD feeds intervals, expected NPV, and credibility | AUROC alone |
| A interval quality | Empirical 90% coverage and mean interval width | Coverage by PD decile, prior decision, prior score, bank-feed missingness, cohort | Direct proxy for `S_cal` | Wide intervals that trivially cover everything |
| Prior-declined extrapolation | Sensitivity-adjusted NPV | Prior-declined approval rate, support distance, PD stress test, abstain policy comparison | This is the main selection-bias trap | Treating unlabelled reject-region profit as proven |
| B trajectory | Weighted CDR MAE/RMSE by cohort-age | Week-13 error, monotonicity violations, interval coverage/width, worst cohort errors | Direct proxy for `S_traj` | Only final-week default rate |
| C counterfactuals | Query coverage and causal-plausibility checks | Duplicate-query consistency, do-feature support, OOD rate, directional sanity, interval width | Direct proxy for `S_C` | Naive re-prediction without causal caveats |
| D writeup | Scoring-criteria coverage | Explicit trap discussion, formulas, audits, limitations, sensitivity tables | Direct proxy for `S_write`; also protects A/C choices | Long generic ML explanation |

## Hackathon Performance Comparison

This table is the better comparison to use during the hackathon because it ties each policy to score-facing performance, not just the screenshot factors.

| Candidate / policy | `S_P&L` proxy | `S_traj` proxy | `S_cal` proxy | `S_C` readiness | `S_write` readiness | Hackathon risk | Recommendation |
|---|---:|---:|---:|---|---|---|---|
| Active Steven policy: LightGBM no-prior-score + exact NPV threshold | Labeled-val NPV `$3.868M`; 9,005 approved; 3,104 prior-declined approvals | B MAE `0.0150` on current approved set | AUROC `0.744`; log loss `0.436`; Brier `0.137`; PD bin interval coverage `0.900` | Missing | Missing | Strong A/B, but incomplete package cannot score | Keep active; finish C/D before more A tinkering |
| Previous direct-NPV blend | Labeled-val NPV `$3.835M`; headline expected NPV `$14.459M`; 9,199 approved | Old B MAE `0.0145`; stale after A switch | AUROC `0.755`; log loss `0.431`; Brier `0.135` | Missing | Missing | Better probability metrics but lower NPV and stale current B | Keep as fallback only |
| HGB no-prior-score challenger | Labeled-val NPV `$3.835M`; 9,036 approved; 3,115 prior-declined approvals | Not rebuilt yet | AUROC `0.753`; log loss `0.431`; Brier `0.136` | Missing | Missing | Near-tie with active, no clear score lift | Useful robustness check, not a clear replacement |
| Ayush branch, current clone | Slide-formula labeled-val NPV `$3.839M`; headline `$10.387M`; avoids LGD trap | Unknown locally | Uses prior score; branch model is HGB bootstrap ensemble | Unknown | Unknown | Conservative on prior-declined region; may leave `S_P&L` on table | Good sanity benchmark for NPV formula and LGD |
| Ronil branch, current clone | Slide-formula labeled-val NPV `$3.080M`; headline `$10.298M`; avoids LGD trap | Has B/C files in branch, not scored here | Drops prior score; HGB + logistic blend | Has branch C | Unknown | Lower verifiable A value in current clone | Mine C/writeup ideas, not A policy |
| Abhi branch, generated A | Slide-formula labeled-val NPV `$1.362M`; headline `$4.528M` | Unknown | CatBoost + isotonic but LGD trap present | Unknown | Unknown | Falls into `LGD ~= 0.91` recovery trap and underfunds | Do not use economics; use only as warning case |

## Trap-Factor View

Keep this thinner table for diagnosis, not for final strategy ranking.

| Person / policy | Model family | LGD/economics assumption | Break-even decision rule | Keeps prior-underwriter score | Funds prior-declined region | Recovery trap? | Main lesson |
|---|---|---|---|---|---|---|---|
| Steven active | LightGBM no-prior-score | Exact brief cash-flow formula via active timing/recovery curves | Tuned NPV margin threshold `0.00938` | No | Yes | Avoided | Best current A/B package, but C/D missing |
| Previous Steven blend | Calibrated HGB/logistic + direct NPV blend | Exact brief cash-flow formula with timing/recovery | NPV margin sign with buffer, not flat PD | Mostly avoids direct prior-score dependence in active comparison | Yes | Avoided | Useful fallback with better AUROC/log loss |
| Ayush | HGB bootstrap ensemble | `LGD = 0.30` amortization-aware shorthand | Flat break-even PD around `0.226` | Yes | No | Avoided | Very good verifiable NPV; conservative reject-region stance |
| Ronil | HGB + logistic blend | `LGD = 0.30` amortization-aware shorthand | Flat break-even PD around `0.226` | No | Yes | Avoided | Aggressive reject-region funding needs sensitivity proof |
| Abhi | CatBoost + isotonic | `LGD ~= 0.91` empirical recovery shortcut | Flat break-even PD around `0.088` | Yes | Yes | Fell in | Underfunds because it ignores pre-default daily draws |

## Progress Metrics

Use this as the working burndown table.

| Workstream | Progress metric | Current value | Target before submit | Owner action |
|---|---:|---:|---:|---|
| A file validity | Expected applicant rows | 13,306 applicants | 13,306, validator clean | Already file-ready |
| A economics | Labeled-val realized NPV | `$3.868M` active, `$3.835M` previous blend | Freeze unless sensitivity audit rejects it | Run reject-region sensitivity if time |
| A approval policy | Test approval rate | `67.6%` active, `69.0%` previous blend | Defensible by NPV sign and support audit | Avoid arbitrary flat PD threshold |
| B file validity | Expected cohort-age rows | 169 | 169, monotone, validator clean | Already file-ready for active A |
| B accuracy | CDR MAE | `0.0150` | Keep near current level after any A rebuild | Rebuild B if A changes |
| B tail/cohort risk | Worst absolute CDR error | `0.0677`, cohort 13 | Reduce if time | Recalibrate cohort 13 or widen interval |
| C file validity | Intervention query rows | 0 / 900 | 900 / 900 | Build immediately |
| C methodology | Causal-safe intervention logic | Not implemented | Feature class rules plus support checks | Use DAG memo as design contract |
| D writeup | 4-page PDF | Missing | Present, concise, formula-backed | Draft after C skeleton, using these tables |
| Final package | Validator result | FAIL | PASS | Add C and D, re-run validator |

## Decision Rule

The next highest-value work is not another AUROC bakeoff. The active A/B package is competitive, but the submission currently fails validation because C is missing, and the writeup is absent. Since C and D are worth 25% combined, and calibration is another 20%, the highest expected score gain is:

```text
1. Build submission_C_counterfactuals.csv.
2. Draft submission_D_writeup.pdf around the actual traps and formulas.
3. Run final interval coverage/width tables for A and B.
4. Only then decide whether to keep LightGBM/no-prior-score or fall back to the prior direct-NPV blend.
5. If A changes, rebuild B and re-run the validator.
```

For model selection, use labeled-validation NPV as the first screen, then reject-region sensitivity as the tiebreaker. For final hackathon performance, optimize the weighted score: profitable loan book, accurate B trajectories, calibrated intervals, credible intervention effects, and a concise defense.
