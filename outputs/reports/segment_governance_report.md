# Segment Governance Report

## Current Policy
- Feature set: `all_engineered`
- Approved total: 9,033
- Prior-declined approved: 2,591
- Labeled validation NPV: $3,912,480
- Bootstrap 90% interval: $3,515,565 to $4,296,075

## Hidden Prior-Declined Risk
- Local validation has no direct labels for prior-declined applicants, so this report treats that region as hidden-outcome risk.
- `approval_support` estimates how close an applicant is to the historically prior-approved/labeled support region.
- `break_even_pd` is the default probability that would make the applicant's expected NPV zero under the slide cash-flow formula.

## Highest-Risk Prior-Declined Approved Segments
- `pd_risk_tier=high`: 3 approvals, mean PD 0.435, mean support 0.574, expected NPV $2,955, headroom 0.308
- `cash_balance_quintile=(-11737.562, -2358.579]`: 74 approvals, mean PD 0.193, mean support 0.484, expected NPV $110,923, headroom 0.257
- `cash_stress_quintile=(1.5, 4.122]`: 84 approvals, mean PD 0.200, mean support 0.493, expected NPV $114,041, headroom 0.206
- `support_tier=low_support`: 147 approvals, mean PD 0.188, mean support 0.394, expected NPV $210,298, headroom 0.196
- `revenue_burden_quintile=(0.0155, 0.0617]`: 184 approvals, mean PD 0.185, mean support 0.617, expected NPV $214,396, headroom 0.334
- `payroll_quintile=(-0.001, 0.339]`: 213 approvals, mean PD 0.155, mean support 0.587, expected NPV $299,001, headroom 0.369
- `credit_stress_quintile=(0.935, 2.092]`: 227 approvals, mean PD 0.175, mean support 0.503, expected NPV $328,681, headroom 0.279
- `payroll_quintile=(0.339, 0.458]`: 270 approvals, mean PD 0.150, mean support 0.611, expected NPV $418,961, headroom 0.417
- `cash_balance_quintile=(-2358.579, -669.589]`: 278 approvals, mean PD 0.177, mean support 0.564, expected NPV $467,838, headroom 0.424
- `sector=4`: 273 approvals, mean PD 0.131, mean support 0.627, expected NPV $478,369, headroom 0.400
- `payroll_quintile=(0.458, 0.572]`: 322 approvals, mean PD 0.134, mean support 0.630, expected NPV $555,639, headroom 0.471
- `owner_personal_credit_band=0`: 386 approvals, mean PD 0.157, mean support 0.500, expected NPV $568,263, headroom 0.338

## Output Files
- `segment_governance_by_factor.csv`
- `prior_declined_hidden_risk_by_segment.csv`
- `prior_declined_stress_test.csv`
- `grouped_time_cv_diagnostics.csv`
- `split_leakage_audit.json`
- `intervention_feature_inventory.csv`
