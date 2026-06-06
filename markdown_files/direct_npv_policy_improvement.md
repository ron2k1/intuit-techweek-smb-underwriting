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

## Active Policy

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
```

