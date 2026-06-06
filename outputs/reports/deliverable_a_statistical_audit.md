# Deliverable A Statistical Audit

## Concepts Incorporated

- PD is `P(default within 90 days = 1 | application features)`, modeled by an HGB ensemble + isotonic + segment-aware meta-calibration.
- Discrete-time weekly hazard model gives `Pr(default by week w | x, approve)` for w in 1..13. Provides `E[t*|default,x]` and feeds Deliverable B's CDR curve directly.
- Recovery rate `rec_i / R_i` is modeled by a separate HGB regressor over defaulted training rows.
- Decision rule is the brief's cash-flow equation with a per-dollar robustness buffer: `d_i = 1[ E[NPV_i | approve] / requested_amount > buffer_rate ]` with
  `E[NPV] = (1-p) (F + R·r·T/365) + p (F + D·(E[t*]-1) + E[rec] - R)`,
  `D = R(1 + r·T/365)/T`, `F = 0.03 R`, `r = 0.35`, `T = 60`.
- 90% PD intervals are split-conformal: residual quantiles fit on the labeled calibration holdout, applied per PD bin for local adaptivity.
- Labels are selectively observed (prior-approved + matured). IPS weights via a prior-approval propensity model widen exposure to the under-represented decline region. `prior_decision` and `prior_approved_amount` are dropped from PD features for the same reason.

## Known Residual Risks

1. Test set is fully unlabeled. Generalization to declined applicants is unverifiable; we rely on propensity weighting + conformal coverage measured on labeled validation.
2. The hazard model's weekly bucket is the finest resolution the data and B's template support; sub-week timing is approximated by the bucket midpoint inside `E[t*]`.
3. Recovery model is trained only on defaulted rows; we assume similar recovery dynamics hold for newly-approved applicants (no censoring outside the 90-day window in the training data).

## Split Summary

```text
     split  rows  prior_approval_rate  observed_outcome_rate  default_rate_when_observed  bank_feed_link_rate
     train 85340             0.606070               0.606070                    0.174471             0.643157
validation  4489             0.568278               0.568278                    0.206194             0.634217
      test  8817             0.563003               0.000000                         NaN             0.632641
```

## Model Metrics

```json
{
  "train_calibration_holdout": {
    "rows": 10345,
    "default_rate": 0.2153697438376027,
    "auroc": 0.7907975514739661,
    "average_precision": 0.5608096811024843,
    "log_loss": 0.4180115603462702,
    "brier": 0.13211210476527105,
    "ece_10bin": 0.006879660161666221
  },
  "validation_labeled_only": {
    "rows": 2551,
    "default_rate": 0.2061936495491964,
    "auroc": 0.7549791109233441,
    "average_precision": 0.5117730419314548,
    "log_loss": 0.43064151974573134,
    "brier": 0.13530076532735805,
    "ece_10bin": 0.017934376473585503
  },
  "settings": {
    "feature_set": "baseline",
    "segment_meta_calibration": false,
    "npv_buffer_rate": 0.005
  },
  "validation_interval_coverage_90": {
    "n": 2551,
    "bin_coverage": 0.9,
    "n_bins": 10,
    "mean_width": 0.08246239971438865,
    "median_width": 0.06090868196131349
  },
  "validation_realized_npv": {
    "n_labeled": 2551,
    "approved_n": 2015,
    "realized_npv_total_under_policy": 3789087.6615743465,
    "realized_npv_total_if_approve_all": 2139702.4905419005,
    "mean_pd_approved": 0.13645025433304134
  }
}
```

## Economics

- Paid-loan margin rate: 0.087534 of principal (F + R r T/365)
- Daily-draw factor D/R: 0.017626
- Origination fee rate: 0.0300
- Active NPV buffer rate: 0.0050
