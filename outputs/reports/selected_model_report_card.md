# Selected Model Report Card

Model: capped feature-regime NPV ensemble + calibrated PD policy.

The active package keeps the calibrated PD model and replaces only the A decision layer with a capped-economics feature-regime value ensemble. The key regime feature is `observed_revenue_trend_3mo`: validation/test mostly live in the future-like weak-revenue regime, so the value ensemble is trained on matching historical rows and aggregated by trimmed mean for stability.

## Active A Policy

| Metric | Value |
|---|---:|
| Approved validation+test applicants | 7,571 |
| Approval rate | 56.9% |
| Prior-declined approvals | 1,937 |
| Labeled-validation approved | 1,918 |
| Labeled-validation realized NPV, capped | $2.712M |
| Headline expected NPV, capped | $14.045M |
| Prior-declined expected NPV, capped | $3.753M |
| Prior-declined expected NPV under 3x stress, capped | $2.731M |
| Prior-declined expected NPV under 6x stress, capped | $1.778M |

## PD Metrics

| Metric | Validation labeled |
|---|---:|
| AUROC | 0.7404 |
| Log loss | 0.4390 |
| Brier score | 0.1376 |
| Mean predicted PD | 0.2103 |
| Actual default rate | 0.2062 |
| A interval bin coverage | 1.000 |

## B Trajectory

| Metric | Value |
|---|---:|
| Approved labeled validation rows used for B check | 1,918 |
| Mean absolute CDR error | 0.0062 |
| Week-13 mean actual CDR | 0.1255 |
| Week-13 mean predicted CDR | 0.1253 |
| B interval coverage | 1.000 |
| Mean B interval width | 0.0556 |

## C Counterfactuals

| Metric | Value |
|---|---:|
| Queries answered | 900 / 900 |
| Unique intervention features | 30 |
| Causal-safe feature count | 106 |
| Tail-support queries | 74 |
| Unseen categorical interventions | 0 |
| Mean counterfactual PD | 0.2951 |
| Mean C interval width | 0.1138 |
| Monotone guards applied | 93 |
| Final material sign violations | 0 |

## Selection And Causal Notes

- Prior-underwriter score, prior decision, prior-approved amount, prior-score logits, and selection-support proxies are excluded from the causal-safe C model.
- Prior-declined applicants are controlled by a positive value-margin floor because their outcomes are not observed locally.
- B is rebuilt on the active A approved set and Markov-switching calibrated against labeled validation cohort-age residual states, so the submitted trajectories are coherent with the loan book actually funded by A.
- The final feature-regime decision layer gives up some headline expected NPV but improves verified capped validation NPV and reduces prior-declined exposure.
- D now names `Global Intuit Hackers` and exists as `outputs/submission/submission_D_writeup.pdf`, so the final folder validates with 0 warnings.
