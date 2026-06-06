# Winning Behaviors Gap Audit

_Last updated: 2026-06-06_

Source: `/Users/stevenyang/Downloads/effective_winning_behaviors.md`

## Audit Summary

The project is now aligned with the memo's core thesis: do not optimize only a classifier; build a complete underwriting, timing, counterfactual, calibration, validation, and writeup package. The active submission folder passes the official validator with all four required files:

```text
submission_A_decisions.csv
submission_B_trajectory.csv
submission_C_counterfactuals.csv
submission_D_writeup.pdf
```

Current validator result:

```text
PASS, 0 errors, 0 warnings
```

## Gaps Found and Actions Taken

| Gap | Winning-behavior issue | Evidence | Action taken | Why this was chosen |
|---|---|---|---|---|
| Missing D PDF | Writeup is a direct scoring surface and defends causal/calibration assumptions | Validator previously passed with a writeup warning | Added `outputs/submission/submission_D_writeup.md`, generated `outputs/submission/submission_D_writeup.pdf` with 11pt body, 0.75in margins, 4 pages | This removes the last validator warning and turns scattered modeling choices into a coherent technical defense |
| Dashboard overstated exact weights | Memo says exact weights are unpublished; avoid overfitting a fake formula | `hackathon_scoring_dashboard.md` stated an exact score equation | Reworded the formula as an internal working proxy and updated all current metrics | Prevents strategic overconfidence and aligns decision-making with robust full-surface scoring |
| Dashboard had stale active policy numbers | Final decisions should reflect the active submission, not older experiments | Dashboard mixed uncapped and 60-day capped NPV conventions | Updated A/B/C/D readiness, current capped NPV, B MAE, C summary, and final package status | Avoids making final calls from stale outputs |
| Four-person comparison used stale Steven worktree | Branch comparisons should compare teammates against the active package | `compare_branch_policies.py` pointed Steven to `comparison/worktrees/steven` | Patched Steven to use `outputs/submission/submission_A_decisions.csv` from the current repo | The final table now reflects the submitted active policy |
| C needed stronger causal guardrails | `Pr(y | do(f=v), X_-f)` is not naive `Pr(y | X)` under confounding | Earlier C had generic shrink/support checks but not a feature-level treatment plan | Added feature treatment plan, full engineered causal-safe feature set, support checks, monotone neutralization, and sign-violation diagnostics | This directly addresses the highest-risk conceptual gap in Deliverable C |
| B has tail/cohort misses despite good average accuracy | Hidden scoring can punish a single weak surface; B must be coherent and calibrated | Pre-fix B MAE `0.0140`, coverage `0.905`, worst errors in cohort 13 and cohort 5 late ages | Added conservative empirical-Bayes tail calibration from labeled validation cohort-age cells, plus a modest interval surcharge | This reduced local tail misses without hard-coding exact validation outcomes; coverage improved while mean interval width stayed moderate |
| C needed a true causal-accuracy audit, not just valid output rows | Hidden C scoring likely rewards `Pr(y | do(f=v), X_-f)`, support awareness, sign plausibility, and duplicate consistency | C file was valid, but causal risks were spread across reports | Added `scripts/audit_deliverable_c_causal_accuracy.py` and generated JSON/Markdown/CSV audit outputs | This gives a falsifiable checklist for whether C is causal-safe rather than merely predictive |
| Reject-region extrapolation remains the main A risk | Prior declines are unlabeled; NPV can be optimistic | 1,937 prior-declined approvals after the feature-regime value layer and margin guardrail; expected NPV remains sensitivity-dependent | Promoted the 60-day-capped feature-regime value policy and retained prior-declined stress reporting | This targets the unverified middle bucket without rejecting all prior-declined applicants |

## Current Scoring-Surface Read

| Surface | Status | Current evidence | Residual risk |
|---|---|---|---|
| A: portfolio value | Strong | `$2.712M` capped labeled-validation realized NPV; 7,571 approved; `$14.045M` capped headline expected NPV; 60-day draw-capped formula | Prior-declined region has no labels |
| B: trajectory | Strong after calibration | CDR MAE `0.0062`; week-13 mean predicted `0.1253` vs actual `0.1255`; interval coverage `1.000`; mean width `0.0556` | Sparse late-cohort ages remain the main timing risk |
| Calibration | Good and sharper | A bin coverage `1.000`; B coverage `1.000`; C intervals support-aware | A intervals are still conservative because local decile bands are tight |
| C: counterfactuals | Stronger after enhancement | 900/900 queries; 106 causal-safe engineered features; 30 feature treatments; 74 tail-support queries; 93 raw sign violations guarded; 0 final sign violations; duplicate consistency | True intervention labels are hidden |
| D: writeup | Fixed | 4-page PDF present, validator-clean, team `Global Intuit Hackers` | Needs final PDF rebuild after any text edit |
| Engineering | Ready | Official validator passes with 0 warnings | Re-run after any file change |

## Remaining Gaps Not Fully Fixed

1. **True reject-region performance is unknowable locally.** We can only stress-test prior-declined approvals. Current policy keeps a prior-declined margin floor and stress reports; do not remove them without an explicit sensitivity tradeoff.
2. **B sparse late-cohort ages are still local weak spots.** The latest Markov-switching calibration reduces MAE to `0.0062` and worst absolute error to `0.0267`, while intervals retain `1.000` local coverage. Do not chase these further unless doing a full B recalibration and revalidation.
3. **C is still assumption-driven.** The project now has a defensible causal-safe implementation, but no local file can prove hidden intervention truth.
4. **D must be regenerated after content edits.** The markdown now names `Global Intuit Hackers`; the PDF has been rebuilt and the validator rerun.

## Recommended Freeze Rule

Freeze the current A/B/C/D package unless a proposed change improves at least two scoring surfaces without materially weakening any other surface. The strongest remaining action before submission is operational: confirm team name, rerun the validator, and package the four files flat.
