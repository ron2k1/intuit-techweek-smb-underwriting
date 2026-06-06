# Segment Governance Report

## Current Policy
- Feature set: `all_engineered`
- Approved total: 7,571
- Prior-declined approved: 1,937
- Labeled validation NPV: $2,711,814
- Bootstrap 90% interval: $2,460,717 to $2,952,684

## Hidden Prior-Declined Risk
- Local validation has no direct labels for prior-declined applicants, so this report treats that region as hidden-outcome risk.
- `approval_support` estimates how close an applicant is to the historically prior-approved/labeled support region.
- `break_even_pd` is the default probability that would make the applicant's expected NPV zero under the slide cash-flow formula.

## Highest-Risk Prior-Declined Approved Segments
- `cash_stress_quintile=(1.5, 4.122]`: 11 approvals, mean PD 0.181, mean support 0.555, expected NPV $12,990, headroom 0.176
- `cash_balance_quintile=(-11737.562, -2358.579]`: 11 approvals, mean PD 0.131, mean support 0.521, expected NPV $23,157, headroom 0.487
- `support_tier=low_support`: 16 approvals, mean PD 0.113, mean support 0.418, expected NPV $33,641, headroom 0.345
- `revenue_burden_quintile=(0.0155, 0.0617]`: 93 approvals, mean PD 0.179, mean support 0.662, expected NPV $114,296, headroom 0.412
- `credit_stress_quintile=(0.935, 2.092]`: 95 approvals, mean PD 0.131, mean support 0.561, expected NPV $163,767, headroom 0.392
- `payroll_quintile=(-0.001, 0.339]`: 130 approvals, mean PD 0.127, mean support 0.625, expected NPV $195,982, headroom 0.431
- `cash_balance_quintile=(-2358.579, -669.589]`: 163 approvals, mean PD 0.129, mean support 0.592, expected NPV $315,868, headroom 0.517
- `payroll_quintile=(0.339, 0.458]`: 203 approvals, mean PD 0.120, mean support 0.639, expected NPV $347,147, headroom 0.488
- `credit_stress_quintile=(0.73, 0.935]`: 201 approvals, mean PD 0.128, mean support 0.595, expected NPV $360,044, headroom 0.441
- `owner_personal_credit_band=0`: 237 approvals, mean PD 0.115, mean support 0.542, expected NPV $398,574, headroom 0.430
- `sector=4`: 212 approvals, mean PD 0.103, mean support 0.657, expected NPV $405,868, headroom 0.462
- `revenue_burden_quintile=(0.012, 0.0155]`: 287 approvals, mean PD 0.140, mean support 0.652, expected NPV $433,573, headroom 0.443

## Output Files
- `segment_governance_by_factor.csv`
- `prior_declined_hidden_risk_by_segment.csv`
- `prior_declined_stress_test.csv`
- `grouped_time_cv_diagnostics.csv`
- `split_leakage_audit.json`
- `intervention_feature_inventory.csv`
