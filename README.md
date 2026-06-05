# Intuit TechWeek NYC 2026 — SMB Underwriting Challenge

Team working repo for the Intuit "Explainable ML" hackathon (NY Tech Week, Jun 5–6, 2026).
Official challenge: https://github.com/intuit/intuit-techweek-nyc-hackathon-2026

> **You are a small-business lender.** Using a historical book of loan applications,
> decide whom to fund **to maximize portfolio profit**, forecast how that book
> defaults over time, answer causal what-if questions, and defend every choice.

## Deliverables (submit exactly these four file names)

| # | File | What | Difficulty | Scored on |
|---|------|------|-----------|-----------|
| **A** | `submission_A_decisions.csv` | approve/decline + calibrated PD + 90% interval, all 13,306 val+test applicants | easy–med | `S_P&L` realized portfolio profit |
| **B** | `submission_B_trajectory.csv` | 13×13 grid: cumulative default rate by cohort-week × loan-age of **your approved set** | med | `S_traj` trajectory accuracy |
| **C** | `submission_C_counterfactuals.csv` | 900 `do(feature=value)` counterfactual PDs | hard | `S_C` true interventional effect |
| **D** | `submission_D_writeup.pdf` | 4-page methodology defense (5 fixed sections) | writeup | `S_write` |

Plus a cross-cutting **`S_cal`**: 90% intervals on A and B must contain truth without being needlessly wide. Scoring weights are unpublished. `validate_submission.py` must print `PASS` or the submission is disqualified.

### Schemas
- **A:** `applicant_id, decision (0/1), predicted_pd, pd_lower_90, pd_upper_90` — `pd_lower_90 <= predicted_pd <= pd_upper_90`; PD required even for declines.
- **B:** `cohort_week (1-13), loan_age_weeks (1-13), cumulative_default_rate, cdr_lower_90, cdr_upper_90` — 169 rows; non-decreasing in age within each cohort.
- **C:** `query_id, predicted_pd_cf, pd_cf_lower_90, pd_cf_upper_90` — one per row of `dataset/intervention_queries.csv`.

## Loan economics (drives Deliverable A's decision)
Fixed terms: **60-day** term, daily ACH, **35% APR**, **3% origination fee**. A fully-repaid loan nets ≈ `amount × (0.35×60/365 + 0.03) ≈ 8.75%`. A default loses most of principal minus `final_recovered_amount`. **Break-even PD ≈ `0.0875 / (LGD + 0.0875)`** → ~8% (no recovery) to ~15% (50% recovery). **Approve below the profit-break-even PD, NOT below 0.5.**

**Default definition:** funded loan defaults on *any* of — 3 consecutive missed daily draws, 6 cumulative missed draws, or positive balance at day 90.

## Known traps (this is the "Explainable ML" theme — finding them = writeup §1)
1. **Reject inference / selection bias** — outcomes exist only for prior-approved & matured loans. Naive `dropna()` then train ⇒ overconfident PD on the real population.
2. **Outcome leakage** — `repayment_status, observation_status, days_to_default, days_to_full_repayment, final_recovered_amount` are post-outcome and blank in test. Never use as features.
3. **`prior_decision` is a collider / selection node** — never condition on it for causal effects (C).
4. **Self-report inflation** — `stated_*` fields are optimistically biased vs `observed_*` bank-feed. `do(stated_revenue=X)` may have ~0 true effect.
5. **MNAR missingness** — bank-feed nulls (no linked feed) are informative; add missingness indicators, don't blindly impute.
6. **Right-censoring + shift** — late cohorts under-observed ⇒ B needs survival methods, not raw fractions.
7. **Planted integrity violations** — check `prior_loans_default_count <= prior_loans_count`, `days_to_default <= 90`, flag vs `repayment_status`, `business_id` not spanning splits, engineered ratio vs raw inputs.

## Team & ownership
| Person | Role | Owns |
|--------|------|------|
| Ronil | ML spine | shared feature pipeline + **Deliverable A** |
| DS Engineer | Survival + causal | **Deliverable B**, **Deliverable C** (DAG/backdoor) |
| ML PM | Modeling + calibration | calibration/conformal layer, A iteration |
| Amazon SWE | Platform & defense | audit, submission assembly, `validate` gate, **Deliverable D** scribe |

Dependency spine: **shared pipeline → A's decisions → (B consumes them) + (C reuses the PD model) → D defends all.**

## Setup
```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# Get the data (not committed): drop dataset-compressed.zip into dataset/ then:
python scripts/setup_data.py

# Build Deliverable A:
python -m src.build_a          # writes submissions/submission_A_decisions.csv

# Validate before upload (must print PASS):
python validate_submission.py submissions/
```

## Layout
```
src/            modeling pipeline (data loading, features, per-deliverable builders)
submissions/    the four output files go here (flat, exact names) for upload
reports/        audit findings, EDA, writeup drafts
dataset/        challenge reference files (data CSVs are gitignored)
expected_ids/   ground-truth ID sets used by the validator
validate_submission.py   organizer's format gate (copied from official repo)
```

## Timeline (hard deadline)
Register on the Google Form by **8 PM Friday** (no submission link otherwise). Submit by **14:00 Saturday**. Upload by 13:45 — never at 13:59.
