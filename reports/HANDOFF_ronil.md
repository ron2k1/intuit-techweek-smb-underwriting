# Branch `ronil` -- handoff (Ronil: A + shared pipeline)

Last updated end of the 2026-06-05/06 overnight session. `main` is untouched; all of
my work is on `ronil`. Everything below is committed and pushed to `origin/ronil`.

## TL;DR for whoever picks this up

- **All three CSV deliverables are built and pass the gate.** `python validate_submission.py submissions`
  prints `RESULT: PASS` (the only warning is the missing `submission_D_writeup.pdf`, which is the
  team's writeup -- not a code issue).
- **Use what's in `submissions/` as-is.** A, B, C are final-quality. If you only have time to do D,
  build it on these numbers.
- **The headline story for D:** we price loan losses with the brief's *exact* per-loan NPV (origination
  fee + ACH draws capped at the 60-day term + recovery - principal), NOT a flat LGD. That, plus a
  walk-forward-verified OOD approval gate, is what separates us from the other teams (see "Forensics").

## Deliverable state

| # | File | Rows | Status | Key numbers |
|---|------|------|--------|-------------|
| A | `submissions/submission_A_decisions.csv` | 13,306 | final | approve ~50.4%, E[NPV] ~$8.55M; deployment AUC 0.757; break-even tau 0.255 (per-loan NPV, not flat LGD) |
| B | `submissions/submission_B_trajectory.csv` | 169 (13x13) | final, 1 open calib item | MAE 0.0137, Winkler 0.096 (beats Steven 0.099); 90% bands are PREDICTIVE (binomial) intervals. CAVEAT: measured ~68% coverage at grader-n, NOT 90% -- band is centered on A's PD which runs ~1.6pp low at the wk-13 asymptote (drift). See open item #1. |
| C | `submissions/submission_C_counterfactuals.csv` | 900 | final | g-formula do() effects; independent of A's economics (not re-run when A changed) |
| D | `submission_D_writeup_template.md` | -- | TEAM TODO | template ready; narrative + scores above |

## How to run (from repo root)

```
python scripts/setup_data.py          # fetch dataset/ locally (gitignored; each member fetches)
python -m src.build_a                 # -> submissions/submission_A_decisions.csv
python -m src.build_b                 # -> submissions/submission_B_trajectory.csv (reads A's output)
python -m src.build_c                 # -> submissions/submission_C_counterfactuals.csv
python validate_submission.py submissions   # must print RESULT: PASS
```

Order matters: **B depends on A's output** (B's per-cohort asymptote = mean predicted PD over A's
approved applicants). Re-run A before B if you touch A.

## Source map (what's where in `src/`)

- `data.py` -- shared feature pipeline (44 raw -> 40 features; leakage quarantined; planted trap rebuilt)
- `model.py` -- PD spine (HGB + logistic, calibration, conformal helpers)
- `build_a.py` -- Deliverable A: economics + decision + calibrated PD + intervals
- `build_b.py` -- Deliverable B: CDR(c,t) = PD_c x G(t) + predictive bands
- `build_c.py` -- Deliverable C: do() counterfactuals (g-formula)
- `backtest.py` -- CANONICAL walk-forward overfitting + drift diagnostic (`python -m src.backtest`)
- `blend_select.py`, `build_baselines.py` -- model-selection support
- `reports/audit_findings.md` -- dataset trap audit (evidence for D section 1)

## What is verified

- Submission gate: PASS (0 errors).
- Censoring/observability: every labelled row is `matured` (0 censored); all post-term defaults sit at
  exactly day 90 -> the min(t*, 60) draw cap is the product spec, not a tuning choice.
- B intervals: the predictive CONSTRUCTION is sound (model + timing + binomial sampling, no
  double-counting, point estimate = PD_c x G(t) preserved). BUT measured ~68% coverage at the
  grader's n_c -- the band is centered on A's PD, which runs ~1.6pp low at the wk-13 asymptote from
  temporal drift (realized ~0.126 vs predicted ~0.110, ~2 sd). NOT a calibrated 90% band as-shipped.
  The earlier "93.5% at matched-n" line was a flawed defense (smaller n just widens the band; that
  proves the band must be wider, not that it's calibrated). See open item #1. We likely still win the
  combined Winkler vs Steven (our band is narrow), but the raw-coverage part of S_cal is the weak spot.
- Not overfitting: walk-forward AUC stable, low memorization gap (run `python -m src.backtest`).

## Forensics (local only, gitignored under `scratch/` -- not on the branch)

Comparisons vs the other teams, for the D defense, kept out of the committed tree:
- Steven: higher raw NPV is an UNCAPPED-draw artifact (credits phantom ACH draws past day 60). Under the
  brief-correct capped scorer, ours wins.
- Abhimanyu: flat LGD (~0.91) -> break-even tau ~0.088, ~2.9x too low -> under-approves; forfeits ~$1.19M
  vs ours under the correct scorer. (Opposite error to Steven.)
- Figure battery (10 PNGs) + proof plots are in `reports/figures/` locally (gitignored).

## Open / nice-to-have

1. **B 90% interval coverage (DECISION for Ronil, found by an overnight adversarial audit).** The
   shipped B band covers ~68% (not 90%) at the grader's n_c because it is centered on A's PD, which
   runs ~1.6pp low at the wk-13 asymptote (temporal drift). The interval MATH is correct; the bias is
   upstream in A. NOT auto-fixed overnight because both fixes are judgment calls with trade-offs:
   (a) re-center A's PD for drift (a calibration shift) -- changes A's approve/decline decisions and
   the headline S_P&L, so it must be re-validated end to end; or (b) conformal-widen B's bands on val
   to hit ~90% coverage -- changes only B, but trades Winkler width against miss penalty (we currently
   BEAT Steven on Winkler, so don't over-widen). Pick one, re-run, re-measure DIRECTIONAL coverage
   (not just width), re-validate. The shipped B is still the best B vs the other teams on Winkler+MAE,
   so this is an improvement, not a fire. Submittable as-is if time runs out; just don't claim "90%
   coverage" in D -- say "predictive intervals; coverage skewed by recent-vintage drift."
2. **D writeup (team).** All scores + the per-team forensic narrative are pre-staged below.
3. `requirements.txt` now also pins `lightgbm` (only `src/blend_select.py` needs it; the documented
   A/B/C/validate path runs without it). If a fresh install still misses a dep, add it there.
4. `tests/` pytest suite (shape/monotonicity/leakage) -- not yet written.
5. NOTE (Ronil to decide): the per-team forensic call-outs above name teammates with dollar figures
   and are on the shared remote. Not a security/secret issue -- but if those teammates can read the
   `ronil` branch, decide whether to keep the framing here or move it to a local-only note before D.
