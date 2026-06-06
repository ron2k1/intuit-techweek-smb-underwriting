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
| B | `submissions/submission_B_trajectory.csv` | 169 (13x13) | final | MAE 0.0137, Winkler 0.096 (beats Steven 0.099); 90% bands are PREDICTIVE (binomial) intervals, 93.5% coverage at matched-n |
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
- B intervals: predictive (model + timing + binomial sampling) -> 93.5% coverage at matched-n.
- Not overfitting: walk-forward AUC stable, low memorization gap (run `python -m src.backtest`).

## Forensics (local only, gitignored under `scratch/` -- not on the branch)

Comparisons vs the other teams, for the D defense, kept out of the committed tree:
- Steven: higher raw NPV is an UNCAPPED-draw artifact (credits phantom ACH draws past day 60). Under the
  brief-correct capped scorer, ours wins.
- Abhimanyu: flat LGD (~0.91) -> break-even tau ~0.088, ~2.9x too low -> under-approves; forfeits ~$1.19M
  vs ours under the correct scorer. (Opposite error to Steven.)
- Figure battery (9 PNGs) + proof plots are in `reports/figures/` locally (gitignored).

## Open / nice-to-have (not blocking submission)

- D writeup (team).
- `requirements.txt` is present; if a fresh install misses a dep, add it there.
- `tests/` pytest suite (shape/monotonicity/leakage) -- not yet written.
