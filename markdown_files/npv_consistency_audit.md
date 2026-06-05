# NPV Consistency Audit

## Finding

The lower NPV was not caused by a cash-flow formula error. It was caused by comparing stale outputs from different A/B runs.

The earlier active `submission_A_decisions.csv` was consistent with:

```text
buffer_per_dollar = 0.005
```

After the direct-NPV policy experiment, the active submission is now a blend:

```text
score = 0.3 * decomposed_npv_margin + 0.7 * direct_hgb_predicted_margin
approve if score > 0.005
```

Current active validation backtest:

```text
approved applicants: 8,953 / 13,306
overall approval rate: 67.29%
validation approval rate: 67.32%
test approval rate: 67.27%
labeled validation realized NPV: $3.789M
```

Current direct-blend policy:

```text
labeled validation realized NPV: $3.835M
approval rate: 69.13%
```

## Why The Number Changed

An earlier run reported approximately `$3.813M` for a `0.01` buffer. That came from older model artifacts and is not reproducible from the current `outputs/deliverable_a_curves.npz`.

With the current artifacts, the buffer audit is:

```text
0.005 buffer -> $3.789M
0.010 buffer -> $3.698M
0.000 buffer -> $3.749M
```

So the current best validation buffer is `0.005`, not `0.010`.

## Formula Check

The current economics code follows the brief:

```text
Repaid:  NPV = F + R * r * T / 365
Default: NPV = F + D * (t* - 1) + rec - R
```

with:

```text
R = requested_amount
r = 0.35
T = 60
F = 0.03 * R
D = R * (1 + r*T/365) / T
```

## Guardrail

Run this after regenerating A or B:

```bash
python3 scripts/audit_npv_consistency.py
```

It writes:

```text
outputs/reports/npv_consistency_summary.json
outputs/reports/npv_consistency_buffer_audit.csv
```
