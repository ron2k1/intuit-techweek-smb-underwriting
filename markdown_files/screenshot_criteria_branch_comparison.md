# Screenshot-Criteria Branch Comparison

This mirrors the teammate screenshot categories, but uses the hackathon slide formula as the primary NPV convention. The screenshot's fixed-LGD/amortizing convention is listed separately because it explains why Ayush's screenshot shows `$5.28M` / `$2.75M`.

| Criterion | Ronil | Ayush | Abhi | Steven |
|---|---:|---:|---:|---:|
| Model | HistGradientBoosting + Logistic blend | HistGradientBoosting bootstrap ensemble | CatBoost 10-fold + isotonic | LightGBM no-prior-score PD + hazard/recovery NPV policy |
| LGD assumption | 0.30 amortizing / brief-standardized | 0.30 amortizing | 0.9086 empirical recovery trap | brief-faithful cash-flow formula |
| Break-even PD | 0.226 | 0.226 | 0.088 | - |
| Keeps `prior_underwriter_score`? | No | Yes | Yes | No |
| Funds prior-declined region? | Yes | No | Yes | Yes |
| Headline NPV (slide formula) | $10.30M | $10.39M | $4.53M | $14.88M |
| Verifiable NPV (slide formula) | $3.08M | $3.84M | $1.36M | $3.87M |
| Headline NPV (fixed-LGD reference) | $6.85M | $5.28M | $3.35M | $7.41M |
| Verifiable NPV (amortizing reference) | $2.28M | $2.75M | $1.13M | $2.76M |
| Recovery trap | Avoided | Avoided | Fell in | Avoided |

## Notes

- Abhi's branch had no committed A submission, so this table uses his own pipeline run in the isolated comparison worktree.
- Ayush's screenshot numbers match the fixed-LGD/amortizing reference: `$5.28M` headline and `$2.75M` verifiable.
- Ronil's current cloned branch does not match the screenshot's `~$15M` headline. The current committed branch produces about `$10.30M` under the slide formula and `$6.85M` under the fixed-LGD reference, while funding 943 prior-declined applicants.
- The verifiable NPV lines intentionally ignore unlabelled prior-declined upside/downside because those outcomes are not observable in validation.
- Steven's active row now uses the promoted LightGBM/no-prior-score policy with a prior-declined guardrail: prior-approved applicants use the selected NPV threshold, while prior-declined applicants need at least `0.03` expected NPV per dollar. B still comes from a weekly hazard model, recovery is modeled separately, and decisions use the slide cash-flow formula through expected NPV.

CSV output: `outputs/reports/screenshot_criteria_branch_comparison.csv`
