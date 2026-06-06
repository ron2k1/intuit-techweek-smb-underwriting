# Branch Comparison: Deliverable A Policy Traps

This compares the four local branches from `https://github.com/ron2k1/intuit-techweek-smb-underwriting` against the same local validation/test data. Abhi's branch did not commit an A submission, so it was generated in the isolated comparison worktree from his pipeline.

Valuation columns are included with the slide formula first and the teammate-screenshot convention as a reference:

- `Brief NPV` follows the hackathon slide formula exactly: repaid `F + R*r*T/365`; default `F + D*(t*-1) + rec - R`.
- `Brief headline` is expected NPV over approved validation+test rows using each branch's submitted PDs and common timing/recovery curves.
- `Brief verifiable` is realized NPV on labeled-validation approved rows.
- `Fixed-LGD EV` and `amortizing profit` are retained only to reconcile the teammate screenshot.

| Person | Model | LGD / Break-even | Approves | Prior-declined funded | Brief headline | Val approved | Brief verifiable | Fixed-LGD EV ref | Amortizing ref | AUC | Key trap read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ronil | HistGradientBoosting + Logistic blend | 0.30 amortizing / brief-standardized / 0.226 | 5,327 (40.0%) | 943 (16.3%) | $10.30M | 1,501 | $3.08M | $6.85M | $2.28M | 0.757 | identified recovery trap; amortization-aware; funds reject region; drops prior score |
| Ayush | HistGradientBoosting bootstrap ensemble | 0.30 amortizing / 0.226 | 5,883 (44.2%) | 0 (0.0%) | $10.39M | 2,007 | $3.84M | $5.28M | $2.75M | 0.758 | identified recovery trap; amortization-aware; avoids reject region; keeps prior score |
| Abhi | CatBoost 10-fold + isotonic | 0.9086 empirical recovery trap / 0.088 | 2,229 (16.8%) | 389 (6.7%) | $4.53M | 618 | $1.36M | $3.35M | $1.13M | 0.764 | fell into recovery trap; funds reject region; keeps prior score |
| Steven | LightGBM no-prior-score PD + hazard/recovery NPV policy | brief-faithful cash-flow formula / - | 9,005 (67.7%) | 3,104 (53.6%) | $15.20M | 2,012 | $3.87M | $7.13M | $2.76M | 0.744 | funds reject region; drops prior score |

## What This Shows

The biggest spread is not model AUC. It is policy/economics: whether the branch catches amortization/LGD, and whether it chooses to fund the prior-declined region where there are no observed outcomes. Steven's active stack is now LightGBM/no-prior-score PD, weekly hazard timing, recovery regression, and an expected-NPV decision policy.

- Abhi has a reasonable CatBoost AUC, but uses empirical post-default recovery as LGD. That sets break-even PD near 8.8% and under-funds.
- Ayush catches the amortization/LGD trap but uses a conservative policy that avoids the prior-declined region, so it gives up speculative upside but avoids unlabelled-region downside.
- Ronil catches amortization and drops prior-underwriter outputs, but funds a large prior-declined region. That can create a much larger optimistic headline EV, while the verifiable labeled-validation number stays close to Ayush because the declined region has no labels.
- Steven's current branch is the active project policy: LightGBM/no-prior-score PD, brief-faithful NPV, and weekly timing/recovery models. It now has the highest labeled-validation NPV in this comparison, but it still needs a defensible story for reject-region uncertainty.

## Scoring Implication

For the hackathon, the safest defense is to separate verifiable performance from speculative reject-region extrapolation. Report the labeled-validation NPV as auditable, then present any prior-declined funding as a sensitivity analysis with pessimistic floors, not as guaranteed value.

CSV output: `outputs/reports/branch_comparison_analysis.csv`
JSON output: `outputs/reports/branch_comparison_analysis.json`
