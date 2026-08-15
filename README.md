# Intuit TechWeek NYC 2026: SMB Underwriting Challenge

Our team took 2nd place at Intuit's explainable-ML hackathon during NY Tech Week (June 5 and 6, 2026, at Intuit HQ). The premise: you are a small-business lender. Decide whom to fund to maximize portfolio profit, forecast how the funded book defaults over time, answer 900 causal what-if queries and defend every methodological choice in a graded writeup.

Official challenge repo: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026

## The four graded artifacts

Everything we were scored on lives in [`submissions/`](submissions/).

| Deliverable | File | What it is |
|---|---|---|
| A | `submission_A_decisions.csv` | Approve/decline, calibrated PD and 90% interval for all 13,306 applicants |
| B | `submission_B_trajectory.csv` | 13x13 cumulative default-rate grid over the book our policy approved |
| C | `submission_C_counterfactuals.csv` | 900 do(feature=value) counterfactual PDs with intervals |
| D | [`submission_D_writeup.md`](submissions/submission_D_writeup.md) | The methodology defense |

The writeup is the best entry point. It covers the causal DAG, the selection-bias handling and every number below.

## What we built

Deliverable A is not a classifier with a 0.5 cutoff. We trained calibrated PD models (validation AUROC 0.740, log loss 0.439, Brier 0.138) and converted each PD into expected NPV with the challenge's cash-flow formula, approving only when the NPV margin cleared a threshold. Default labels only exist for loans a prior underwriter approved and let mature, so prior-declined applicants got an extra margin guardrail instead of blind trust in the model. The final policy approved 7,571 applicants, 1,937 of them prior-declined, and realized about $2.71M in labeled-validation NPV (bootstrap 5th to 95th percentile: $2.46M to $2.95M).

Deliverable B forecasts how the book A chose to fund defaults over 13 weeks. We built applicant-level cumulative hazard curves, shrank sparse cohorts toward the global approved curve and applied Markov-switching residual calibration over cohort timing states. Validation timing came out at mean CDR error 0.0062, with week-13 predicted at 0.1253 against 0.1255 actual. The 90% intervals covered every grid cell at a mean width of 0.0556.

Deliverable C asks for interventional probabilities, not associations. We excluded prior-underwriter artifacts from the causal model, recomputed engineered features under each intervention and applied per-feature guardrails: clean business-state interventions keep the model's delta, while self-reported and measurement-process fields get shrunk toward baseline with wider intervals. The causal audit surfaced 93 material sign violations on monotone features and neutralized all 93. Zero survived to the final file.

## Why it was hard

The dataset was booby-trapped, on theme. Labels exist only for loans the prior underwriter approved, so naive training learns the wrong conditional entirely. Post-outcome columns sat right next to legitimate features waiting to leak. Self-reported revenue ran optimistically inflated relative to the bank-feed observations. Bank-feed missingness was informative rather than random, and late cohorts were right-censored. Finding those traps and defending the fixes was most of the scoring.

## Team

| Person | Owned |
|---|---|
| Ronil Basu | Shared feature pipeline and Deliverable A |
| Abhimanyu Swaroop | Deliverables B and C (survival modeling, DAG and backdoor analysis) |
| Steven Yang | Calibration and conformal layer, A iteration |
| Ayush Chadha | Audit, submission assembly, validation gate, Deliverable D scribe |

## Layout

```
submissions/             the four graded artifacts, including the writeup
dataset/                 challenge data and reference files
expected_ids/            ground-truth ID sets used by the validator
validate_submission.py   the organizer's format gate
src/                     empty. The sprint's modeling code lived in our team workspace
```

`validate_submission.py submissions/` reproduces the organizer's format check and prints PASS against the committed artifacts.
