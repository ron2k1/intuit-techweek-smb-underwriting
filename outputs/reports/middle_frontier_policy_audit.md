# Middle Frontier Policy Audit

The active policy keeps LightGBM as the global calibrated PD backbone, then applies an economic decision layer.
The hard region is not all applicants; it is the break-even frontier and especially prior-declined approvals with no local labels.

## Summary
- Approved validation+test applicants: 7,973
- Prior-declined approvals: 2,145
- Headline expected NPV: $14,706,220
- Labeled-validation realized NPV: $2,652,215
- Labeled-validation approved: 1,975
- Minimum prior-declined approved break-even odds gamma: 3.00x

## Interpretation
- Prior-approved near-frontier validation approvals are locally labeled and were not harmed by the tightened guardrail.
- Prior-declined frontier approvals remain unlabeled, so the guardrail now removes approvals that do not survive a 3x default-odds stress.
- This is a segmentation layer on top of LightGBM, not a replacement for the backbone model.

## Output Tables
- `middle_frontier_margin_bins.csv`
- `middle_frontier_prior_declined_gamma_bins.csv`
- `middle_frontier_reject_stress.csv`
