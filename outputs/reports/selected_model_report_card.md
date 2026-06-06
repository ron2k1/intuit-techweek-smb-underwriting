# Selected Model Report Card

Model: compact raw valid-prior LightGBM + brief-formula NPV policy.

The active package promotes the compact raw valid-prior LightGBM policy because it improves verifiable validation NPV while keeping prior-policy proxy fields out of the PD model. It uses the brief cash-flow formula with timing/recovery curves and a prior-declined margin guardrail.

## Active A Policy

| Metric | Value |
|---|---:|
| Approved validation+test applicants | 9,033 |
| Approval rate | 67.9% |
| Prior-declined approvals | 2,591 |
| Labeled-validation approved | 2,196 |
| Labeled-validation realized NPV | $3.912M |
| Headline expected NPV | $14.822M |
| Prior-declined expected NPV | $4.542M |
| Prior-declined expected NPV under 3x stress | $2.533M |

## PD Metrics

| Metric | Validation labeled |
|---|---:|
| AUROC | 0.7461 |
| Log loss | 0.4351 |
| Brier score | 0.1365 |
| Mean predicted PD | 0.2102 |
| Actual default rate | 0.2062 |
| A interval bin coverage | 1.000 |

## B Trajectory

| Metric | Value |
|---|---:|
| Approved labeled validation rows used for B check | 2,196 |
| Mean absolute CDR error | 0.0117 |
| Week-13 mean actual CDR | 0.1465 |
| Week-13 mean predicted CDR | 0.1470 |
| B interval coverage | 0.970 |
| Mean B interval width | 0.0734 |

## C Counterfactuals

| Metric | Value |
|---|---:|
| Queries answered | 900 / 900 |
| Unique intervention features | 30 |
| Causal-safe feature count | 106 |
| Tail-support queries | 74 |
| Unseen categorical interventions | 0 |
| Mean counterfactual PD | 0.2952 |
| Mean C interval width | 0.1136 |
| Monotone guards applied | 82 |
| Final material sign violations | 0 |

## Selection And Causal Notes

- Prior-underwriter score, prior decision, prior-approved amount, prior-score logits, and selection-support proxies are excluded from the causal-safe C model.
- Prior-declined applicants are approved only with a margin guardrail because their outcomes are not observed locally.
- B is rebuilt on the active A approved set and tail-calibrated against labeled validation cohort-age cells, so the submitted trajectories are coherent with the loan book actually funded by A.
- D now names `Global Intuit Hackers` and exists as `outputs/submission/submission_D_writeup.pdf`, so the final folder validates with 0 warnings.
