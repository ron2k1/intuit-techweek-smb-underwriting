# Deliverable B Methodology And Policy

## Objective

Deliverable B forecasts cumulative default rates by cohort week and loan age for the team's own approved loans:

```text
cumulative_default_rate(cohort_week, loan_age_weeks)
  = mean approved-applicant P(default by week a)
```

This uses the Deliverable A approved set, not all applicants.

## Portfolio Policy Update

The brief scores portfolio value from the loans we choose to fund. The decision rule is therefore economic:

```text
approve if E[NPV | approve] / requested_amount > buffer
```

A validation backtest over labeled validation rows selected:

```text
buffer_per_dollar = 0.01
```

This improved labeled validation realized NPV versus the no-buffer rule:

```text
no-buffer realized NPV:  3.695M
1% buffer realized NPV:  3.813M
```

The current `outputs/submission/submission_A_decisions.csv` uses this 1% buffer.

## B Trajectory Method

Inputs:

- `outputs/deliverable_a_curves.npz`
- `outputs/submission/submission_A_decisions.csv`
- `data/submission_B_template.csv`
- `data/cohort_week_definitions.csv`

Steps:

1. Assign validation/test applicants to `cohort_week`.
2. Use the final A decisions to select approved applicants.
3. Use the discrete-time hazard model's cumulative curve shape.
4. Normalize each applicant's week-13 cumulative default probability to its A `predicted_pd`.
5. Aggregate applicant curves by cohort week.
6. Apply mild shrinkage toward the global approved curve for sparse cohorts.
7. Add 90% intervals using binomial finite-sample uncertainty plus a small model buffer.
8. Enforce:

```text
0 <= lower <= point <= upper <= 1
cumulative_default_rate non-decreasing by loan_age_weeks
```

## Files

Generated submission:

```text
outputs/submission/submission_B_trajectory.csv
```

Reports:

```text
outputs/reports/deliverable_b_summary.json
outputs/reports/deliverable_b_policy_buffer_tuning.csv
outputs/reports/deliverable_b_cohort_diagnostics.csv
```

Script:

```text
scripts/build_deliverable_b.py
```

