# Advanced Statistical Edges Audit

## Main Read
- The active policy is now full-engineered and prior-policy-proxy filtered.
- The strongest remaining statistical risk is reject inference: locally observed labels still do not identify outcomes for prior-declined applicants.
- Drift and recency diagnostics should be discussed in Deliverable D as validation controls, not as proof of hidden test performance.
- The life-table section is a validation audit on this hackathon data only; it does not import the external example data from the screenshots.

## Screenshot Concepts Applied
- Reject workflow: prior-approved/labeled applications are the `accepted` population; prior-declined/unlabeled applications are the `rejected` population.
- Simple augmentation: score rejects, assign hard pseudo-good/bad labels by cutoff, and refit a scorecard-style model.
- Fuzzy augmentation: duplicate each reject into bad/good rows with probability weights, then refit.
- Hazard audit: compute `h(k)=d_k/r_k` on approved labeled validation loans and compare it to the model-implied weekly hazard.

## Key Numbers
- Forbidden model features found: 0
- Train/validation business overlap: 0
- Train/test business overlap: 0
- Worst model-relevant train/test PSI: 2.352
- Worst model-relevant train/validation PSI: 2.252
- Prior-declined approvals: 2,591
- Prior-declined base expected NPV: $4,542,002
- Prior-declined expected NPV at 3x default odds: $2,533,393
- Best reject-augmentation diagnostic: `simple_gamma_1.5` with labeled validation NPV $3,847,114
- Life-table week-13 empirical CDR: 0.1475
- Life-table week-13 predicted CDR: 0.1513

## Output Files
- `advanced_leakage_audit.json`
- `advanced_suspicious_single_feature_auc.csv`
- `advanced_drift_diagnostics.csv`
- `advanced_recency_weighting_experiment.csv`
- `advanced_expected_profit_quintiles.csv`
- `advanced_approved_profit_deciles.csv`
- `advanced_reject_inference_sensitivity.csv`
- `advanced_reject_break_even_gamma_by_tier.csv`
- `advanced_reject_augmentation_experiment.csv`
- `advanced_life_table_hazard_validation.csv`
- `advanced_hazard_timing_summary.json`
