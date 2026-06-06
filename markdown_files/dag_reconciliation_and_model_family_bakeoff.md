# DAG Reconciliation and Model-Family NPV Bakeoff

## What Changed

I reconciled `hackathon_dag_and_dataset_tricks_analysis.md` against the current project docs and implementation. The memo reinforces the core strategy already in the project:

- Outcomes are selectively observed through the prior lender's approval/maturity process.
- Prior-underwriter fields are predictive policy artifacts, not causal drivers.
- Bank-feed missingness is informative and should not be erased by blind imputation.
- Counterfactual interventions need a separate causal-safe feature philosophy.
- The main scoring gap is not generic AUROC; it is the economics and policy layer.

The concrete gap was that CatBoost and LightGBM had only been evaluated mostly as raw-score/AUROC challengers. I added an NPV-policy bakeoff that converts each model family's PDs into expected NPV decisions and scores the resulting policy on labeled-validation realized NPV.

## LGD Assumption

The active project does not use a single fixed LGD in the production NPV decision. It uses the brief cash-flow formula:

```text
Repaid:  F + R * r * T / 365
Default: F + D * (t* - 1) + rec - R
```

This means default loss varies by expected default timing and expected recovery. On observed defaults:

| Split | Brief-formula implied LGD | Simpler amortizing LGD |
|---|---:|---:|
| train defaults | mean 0.137, median 0.241 | mean 0.238, median 0.223 |
| validation defaults | mean 0.148, median 0.236 | mean 0.241, median 0.219 |

The teammate screenshot's `LGD = 0.30` is a conservative rounded simplification for the amortization-aware story. The planted trap is `LGD ~= 0.91`, which comes from treating only final post-default recovery as recovery and ignoring daily draws before default.

## New Bakeoff

Script:

```text
scripts/experiment_model_family_npv_bakeoff.py
```

Outputs:

```text
outputs/reports/model_family_npv_bakeoff.csv
outputs/reports/model_family_npv_bakeoff_threshold_sweeps.csv
outputs/reports/model_family_npv_bakeoff_summary.json
```

The bakeoff trains HGB, LightGBM, and CatBoost under two feature sets:

- `current_predictive`: current application-time feature set.
- `no_prior_score`: removes `prior_underwriter_score` and derived prior-score proxies (`prior_score_logit`, `selection_support_index`).

Each model is calibrated on a time-held-out slice of train, scored on validation/test, converted into expected NPV using the active timing/recovery curves, and thresholded by NPV margin. The threshold is selected by labeled-validation realized NPV, so it is a policy lead, not proof of test lift.

## Results

| Rank | Model | Feature set | Labeled-val NPV | Approved labeled val | Approved total | Prior-declined approved | AUROC |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | LightGBM | no_prior_score | $3.868M | 2,012 | 9,005 | 3,104 | 0.744 |
| 2 | HGB | no_prior_score | $3.835M | 2,012 | 9,036 | 3,115 | 0.753 |
| 3 | active submission | active project | $3.835M | 2,047 | 9,199 | n/a | 0.755 |
| 4 | HGB/LightGBM/CatBoost mean | no_prior_score | $3.820M | 2,012 | 9,022 | 3,101 | 0.753 |
| 5 | HGB/LightGBM/CatBoost mean | current_predictive | $3.813M | 2,020 | 9,046 | 3,112 | 0.752 |
| 6 | LightGBM | current_predictive | $3.812M | 1,948 | 8,621 | 2,904 | 0.743 |
| 7 | HGB | current_predictive | $3.812M | 2,020 | 9,000 | 3,072 | 0.752 |
| 8 | CatBoost | no_prior_score | $3.781M | 1,948 | 8,634 | 2,903 | 0.750 |
| 9 | CatBoost | current_predictive | $3.779M | 2,085 | 9,481 | 3,356 | 0.752 |

## Interpretation

The best challenger is `LightGBM + no_prior_score`, with about `$33K` more labeled-validation NPV than the active submitted policy. That is a small lift, and its AUROC is lower than HGB/CatBoost, which confirms the point: ranking is not the objective; NPV policy is.

The no-prior-score feature set performs well because it reduces inheritance of the old lender's approval frontier while still using bank-feed, bureau, application, and platform signals. This is consistent with the DAG memo: prior-underwriter signals are useful diagnostics but can contaminate the reject-region extrapolation.

Before promotion, the active submission was best described as `Calibrated HGB/logistic PD + hazard/recovery + direct-NPV blend`. The current active submission is `LightGBM no-prior-score PD + hazard/recovery NPV policy`: LightGBM supplies the submitted PDs, the existing hazard/recovery curves supply timing and recovery economics, and A's approval decisions use expected NPV.

## Active Promotion

The LightGBM/no-prior-score policy was promoted to the active submission with:

```text
scripts/apply_lightgbm_no_prior_policy.py
scripts/build_deliverable_b.py --use-existing-a-decisions
```

This improves labeled-validation NPV from `$3.835M` to `$3.868M` and reduces approved total from `9,199` to `9,005`. Because the single LightGBM model has less ensemble dispersion, A's 90% PD intervals now use a conservative `0.06` half-width floor; this restores labeled-validation bin-level interval coverage to `0.90`.

The largest unresolved uncertainty remains the prior-declined region with no labels. The promoted policy still funds `3,104` prior-declined applicants, so the writeup should frame that as model-based extrapolation with uncertainty, not as directly verified profit.

## Screenshot Metric Reconciliation

The teammate screenshot uses two different quantities and a simplified economics convention:

- `Headline NPV`: expected/model NPV over all approved validation + test applicants.
- `Verifiable NPV`: realized value on labeled validation only.
- The screenshot's Ayush numbers use the fixed-LGD/amortizing shorthand, not the exact slide formula.

Ayush's branch reproduces the screenshot convention exactly:

```text
fixed-LGD headline expected NPV:                  $5.279M
amortizing labeled-validation verifiable profit:  $2.755M
```

Under the exact slide formula, the same submitted decisions produce different headline/verifiable values because defaults are valued by predicted default day and recovery:

```text
Ayush slide-formula headline NPV:      $10.39M
Ayush slide-formula verifiable NPV:    $3.84M
Steven previous slide-formula headline NPV:     $14.46M
Steven previous slide-formula verifiable NPV:   $3.83M
Steven promoted LightGBM verifiable NPV:         $3.87M
```

Ronil's current cloned branch does not reproduce the screenshot's `~$15M`. The current branch produces about `$10.30M` under the slide formula and `$6.85M` under the fixed-LGD reference for 5,327 funded loans. The screenshot likely used an older branch state or a more aggressive trust-policy mode in the unlabelled prior-declined region.

## Gaps Still Open

- If time permits, run a reject-region sensitivity table: optimistic model PD, PD floored at prior-approved edge, and abstain/defer policy.
- Deliverable C still needs the same DAG logic formalized in code: cache duplicate interventions, classify intervention features, and widen intervals outside support.
