# Direct NPV Policy Improvement

## Why We Added This

The official objective is realized portfolio value:

```text
maximize E[sum_i d(x_i) * NPV_i]
```

The previous production policy estimated:

```text
PD + expected default timing + expected recovery -> decomposed expected NPV
```

That is coherent, but it can miss nonlinear value patterns. The direct-NPV experiment asks the target question more directly:

```text
features -> E[realized NPV per dollar | x]
```

without using outcome columns as features.

## Previous Active Policy

Active model label:

```text
Calibrated HGB/logistic PD + hazard/recovery + direct-NPV blend
```

Current active decision score:

```text
score = 0.3 * decomposed_npv_margin + 0.7 * direct_hgb_predicted_margin
approve if score > 0.005
```

This changes only `decision` in `submission_A_decisions.csv`.

The submitted PD values and 90% PD intervals still come from the calibrated A PD model.

## Validation Result

On labeled validation rows:

```text
previous decomposed NPV policy: $3.789M
direct-NPV blend policy:       $3.835M
lift:                          +$45K
```

Active approval profile:

```text
approved total:       9,199 / 13,306
approval rate:        69.13%
validation approval:  69.35%
test approval:        69.03%
```

## Model-Family Bakeoff Update

After reconciling the DAG/tricks memo, I added a policy-level bakeoff for HGB, LightGBM, and CatBoost:

```text
scripts/experiment_model_family_npv_bakeoff.py
```

Unlike the earlier raw-score experiment, this evaluates each model by:

```text
calibrated PD -> expected NPV using timing/recovery curves -> approval policy -> labeled-validation realized NPV
```

Top results:

```text
LightGBM + no_prior_score: $3.868M labeled-validation NPV
HGB + no_prior_score:      $3.835M labeled-validation NPV
active direct-NPV blend:   $3.835M labeled-validation NPV
CatBoost + no_prior_score: $3.781M labeled-validation NPV
```

The strongest challenger is LightGBM without prior-underwriter score/proxy features. The lift over the active policy is small, so this is a serious candidate but not an automatic replacement without a reject-region sensitivity audit.

## Promoted Policy

The active submission has now been promoted to:

```text
LightGBM + no_prior_score -> expected NPV via timing/recovery curves -> threshold 0.009380
```

Promotion script:

```text
scripts/apply_lightgbm_no_prior_policy.py
```

Current active A/B results:

```text
labeled-validation NPV:  $3.868M
approved total:          9,005 / 13,306
validation approval:     67.77%
test approval:           67.63%
prior-declined funded:   3,104
PD interval coverage:    0.90 bin-level coverage on labeled validation
```

The NPV lift over the previous direct-NPV blend is about `$33K`; the lift over the decomposed NPV-only policy is about `$79K`.

## Why This Is A Reasonable Move

The challenge was designed to penalize brute-force classification. A direct value model is aligned with the actual scoring target:

```text
realized portfolio value
```

It also addresses the observed error pattern:

- some approved loans had large negative realized NPV
- some rejected loans had large positive realized NPV
- flat PD thresholds do not account for timing, amount, and recovery

## Caveat

The blend weight and threshold were selected on labeled validation. This is allowed as tuning/calibration, but it can overfit. The lift is modest, so this is a pragmatic improvement rather than proof of test dominance.

## Related Reports

```text
outputs/reports/direct_npv_policy_experiment.csv
outputs/reports/direct_npv_model_diagnostics.csv
outputs/reports/direct_npv_policy_summary.json
outputs/reports/direct_npv_blend_active_policy_summary.json
outputs/reports/model_family_npv_bakeoff.csv
outputs/reports/model_family_npv_bakeoff_summary.json
```
