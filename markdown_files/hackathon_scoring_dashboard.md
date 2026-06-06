# Hackathon Scoring Dashboard and Comparison Table

_Last updated: 2026-06-06_

This table replaces the older screenshot-style comparison as the main operating dashboard. The screenshot factors are still useful for detecting the LGD/recovery trap and prior-lender selection trap, but final strategy should use the full scoring surface. The exact official weights are not public in the repo README, so the weights below are an internal working proxy, not a claimed scoring formula:

```text
working proxy = 0.30 S_P&L + 0.25 S_traj + 0.20 S_cal + 0.10 S_C + 0.15 S_write
```

The practical implication is that a high-AUROC or high-headline-NPV model can still lose if C is missing, B is not rebuilt after A changes, intervals are uncalibrated, or the writeup does not defend the causal assumptions.

## Executive Scorecard

| Score component | Weight | What judges care about | Current proxy metric | Current status | Biggest risk | Next action |
|---|---:|---|---:|---|---|---|
| `S_P&L` | 30% proxy | Realized portfolio value from A decisions | Active capped labeled-val NPV: `$2.712M`; 7,571 approved; 1,937 prior-declined approvals; capped headline `$14.045M` | Stronger verified value | Prior-declined region has no labels, so headline NPV can be optimistic | Keep feature-regime value layer and prior-declined margin floor |
| `S_traj` | 25% proxy | Accuracy of B cumulative default trajectories on our approved set | Active B CDR MAE: `0.0062`; week-13 mean pred `0.1253` vs actual `0.1255`; interval coverage `1.000` | Stronger | Cohort 13 early/mid ages remain largest residuals but are covered | Keep Markov-switching B; disclose sparse-cohort/tail limitation in D |
| `S_cal` | 20% proxy | 90% intervals on A PD and B trajectories contain truth without being too wide | Active A AUROC `0.740`, log loss `0.439`, Brier `0.138`; A bin coverage `1.000`; B coverage `1.000` | Stronger and sharper | A intervals are conservative; B mean width now `0.0556` | Defend coverage/width tradeoff and avoid further widening |
| `S_C` | 10% proxy | Counterfactual PDs match true intervention effects, not naive re-prediction | 900 / 900 C queries; mean CF PD `0.295`; 74 tail-support queries; 93 monotone guards | Ready | True intervention labels are hidden; causal assumptions drive accuracy | Defend treatment plan, support checks, shrinkage, and monotone neutralization |
| `S_write` | 15% proxy | Clear methodological defense | `submission_D_writeup.pdf` present; 4 pages; validator clean; team `Global Intuit Hackers` | Ready | PDF must be regenerated after any markdown edit | Keep concise D and re-run validator |

## Submission Readiness

| Deliverable | Required artifact | Current artifact | Completeness | Readiness metric | Current value | Score impact |
|---|---|---|---:|---|---:|---|
| A | `submission_A_decisions.csv` | Present | 100% file-ready | Approved applicants | 7,571 | Drives `S_P&L`, B denominator, and A interval calibration |
| A | PD and 90% PI columns | Present | Final report-card audit complete | AUROC / log loss / Brier | `0.740 / 0.439 / 0.138` | Drives `S_cal`; not directly enough for `S_P&L` |
| B | `submission_B_trajectory.csv` | Present | 100% file-ready for current A | Cohort-age rows | 169 | Drives `S_traj` and B interval calibration |
| B | CDR forecasts and intervals | Present | Markov-switching calibrated for current A | CDR MAE / coverage | `0.0062 / 1.000` | Strong enough to keep unless A changes |
| C | `submission_C_counterfactuals.csv` | Present | 100% file-ready | Query coverage | 900 / 900 | Drives `S_C`; methodology must be defended |
| D | `submission_D_writeup.pdf` | Present | 100% file-ready | 4-page limit | 4 pages | 15% direct score plus defense of all modeling choices |
| Package | Flat submission folder | A/B/C/D present | Passes validator cleanly | `validate_submission.py` | 0 errors, 0 warnings | Ready to upload structurally |

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
| Steven final | Capped labeled-val NPV `$2.712M`; capped headline `$14.045M`; 7,571 approved; 1,937 prior-declined approvals | B MAE `0.0062`, coverage `1.000`, mean width `0.0556` on current approved set | AUROC `0.740`; log loss `0.439`; Brier `0.138`; A bin interval coverage `1.000` | 900 / 900 queries; 30 feature treatments; 93 raw sign violations guarded to 0 final | Present with team `Global Intuit Hackers` | Best current capped complete package; lower headline but better verified value | Freeze unless hidden-risk preference shifts sharply conservative |
| Steven previous | Capped labeled-val NPV `$2.652M`; capped headline `$14.706M`; 7,973 approved; 2,145 prior-declined approvals | B MAE `0.0066`, coverage `1.000`, mean width `0.0556` after Markov B | AUROC `0.740`; log loss `0.439`; Brier `0.138`; A bin interval coverage `1.000` | 900 / 900 queries; same C file | Older D text | More approvals and higher headline, but lower verified capped NPV | Replaced by feature-regime value layer |
| Ayush branch, current clone | Capped labeled-val NPV `$2.628M`; capped headline `$10.263M`; 5,883 approved; 0 prior-declined approvals | Unknown locally | AUROC `0.758`; uses prior score; branch model is HGB bootstrap ensemble | Unknown | Unknown | Safest reject-region floor but leaves headline upside on table | Main conservative benchmark |
| Ronil branch, current clone | Capped labeled-val NPV `$2.200M`; capped headline `$10.181M`; 5,327 approved; 943 prior-declined approvals | Has B/C files in branch, not scored here | AUROC `0.757`; drops prior score; HGB + logistic blend | Has branch C | Unknown | Lower verified and lower headline than Steven/Ayush | Mine C/writeup ideas, not A policy |
| Abhi branch, generated A | Capped labeled-val NPV `$1.115M`; capped headline `$4.403M`; 2,229 approved; 389 prior-declined approvals | Unknown | AUROC `0.764`; CatBoost + isotonic | Unknown | Unknown | Falls into empirical-recovery/LGD trap and underfunds | Do not use economics; use only as warning case |
| Previous direct-NPV blend | Capped labeled-val NPV `$2.596M`; capped headline `$14.326M`; 9,199 approved | Old B stale after A switch | AUROC `0.755`; log loss `0.431`; Brier `0.135` | Not tied to current package | Draft now exists but not based on this policy | Better probability metrics but worse capped P&L package | Keep as fallback only |

## Trap-Factor View

Keep this thinner table for diagnosis, not for final strategy ranking.

| Person / policy | Model family | LGD/economics assumption | Break-even decision rule | Keeps prior-underwriter score | Funds prior-declined region | Recovery trap? | Main lesson |
|---|---|---|---|---|---|---|---|
| Steven | Compact risk-factor LightGBM | 60-day draw-capped cash-flow via active timing/recovery curves | NPV margin threshold plus 3x prior-declined default-odds stress | No | Yes, with guardrail | Avoided | Best current complete A/B/C/D package |
| Previous Steven blend | Calibrated HGB/logistic + direct NPV blend | Exact brief cash-flow formula with timing/recovery | NPV margin sign with buffer, not flat PD | Mostly avoids direct prior-score dependence in active comparison | Yes | Avoided | Useful fallback with better AUROC/log loss |
| Ayush | HGB bootstrap ensemble | `LGD = 0.30` amortization-aware shorthand | Flat break-even PD around `0.226` | Yes | No | Avoided | Very good verifiable NPV; conservative reject-region stance |
| Ronil | HGB + logistic blend | `LGD = 0.30` amortization-aware shorthand | Flat break-even PD around `0.226` | No | Yes | Avoided | Aggressive reject-region funding needs sensitivity proof |
| Abhi | CatBoost + isotonic | `LGD ~= 0.91` empirical recovery shortcut | Flat break-even PD around `0.088` | Yes | Yes | Fell in | Underfunds because it ignores pre-default daily draws |

## Progress Metrics

Use this as the working burndown table.

| Workstream | Progress metric | Current value | Target before submit | Owner action |
|---|---:|---:|---:|---|
| A file validity | Expected applicant rows | 13,306 applicants | 13,306, validator clean | Already file-ready |
| A economics | Capped labeled-val realized NPV | `$2.712M` active, `$2.628M` Ayush | Freeze unless sensitivity audit rejects it | Keep feature-regime value layer |
| A approval policy | Test approval rate | `56.9%` active, `44.2%` Ayush | Defensible by NPV sign and support audit | Avoid arbitrary flat PD threshold |
| B file validity | Expected cohort-age rows | 169 | 169, monotone, validator clean | Already file-ready for active A |
| B accuracy | CDR MAE | `0.0062` after Markov-switching calibration | Keep near current level after any A rebuild | Rebuild B if A changes |
| B tail/cohort risk | Worst absolute CDR error | `0.0267`, cohort 13 age 5 | Do not worsen coverage/width tradeoff | Mention sparse-cohort risk; avoid overfitting B late cohorts |
| C file validity | Intervention query rows | 900 / 900 | 900 / 900 | Already file-ready |
| C methodology | Causal-safe intervention logic | Implemented with support checks and shrinkage | Feature class rules plus support checks | Defend in writeup |
| D writeup | 4-page PDF | Present, 4 pages, team `Global Intuit Hackers` | Present, concise, formula-backed | Rebuild PDF after any markdown edit |
| Final package | Validator result | PASS, 0 errors, 0 warnings | PASS plus D included | Re-run validator after any change |

## Decision Rule

The next highest-value work is not another AUROC bakeoff. The active A/B/C package is structurally valid, and the writeup is now the main missing scoring surface. Since D is worth 15%, and calibration is another 20%, the highest expected score gain is:

```text
1. Draft submission_D_writeup.pdf around the actual traps and formulas.
2. Run final interval coverage/width tables for A and B.
3. Decide whether to keep LightGBM/no-prior-score or fall back to the prior direct-NPV blend.
4. If A changes, rebuild B and C, then re-run the validator.
```

For model selection, use labeled-validation NPV as the first screen, then reject-region sensitivity as the tiebreaker. For final hackathon performance, optimize the weighted score: profitable loan book, accurate B trajectories, calibrated intervals, credible intervention effects, and a concise defense.
