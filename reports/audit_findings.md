# Audit findings — the traps, with evidence

> Source of truth for **writeup D §1 (assumptions violated)** and the shared
> feature contract in `src/data.py`. Every claim below is reproducible from
> `scratch/probe.py` + `scratch/probe2.py` (gitignored; run against the raw CSVs).

**Headline:** the data is *arithmetically* clean — **0 integrity violations** across
every invariant we checked. The traps are **structural** (who is in the data, which
columns are post-outcome, how one column was secretly built), not dirty cells. So the
correct response is principled *modeling*, not `dropna()` / blind imputation.

---

## 1. Selection bias / reject inference (the dominant trap)
Outcomes (`default_flag`) exist **only** for prior-approved & matured loans.

| split | rows | labelled (have outcome) | observed default rate |
|---|---|---|---|
| train | 85,340 | 51,722 (`prior_decision==1`) | 17.45% |
| validation | 4,489 | 2,551 | 20.62% |
| test | 8,817 | **0** | — (calibrate on validation only) |

`prior_decision` is essentially a **hard cutoff on `prior_underwriter_score` at ≈ 0.273**:

| | declined (0) | approved (1) |
|---|---|---|
| score range | 0.000 → **0.273** | **0.273** → 1.000 |

We have labels only above the cut; A must price the **40% of applicants below it**,
who are systematically riskier (utilization 0.60 vs 0.44, weaker credit band 1.50 vs
2.31, fewer years in business 5.1 vs 6.5). **Consequence:** a model trained on the
approved book is overconfident (PD too low) exactly where it is most wrong.
**Mitigation:** keep declines in the scored set; widen 90% intervals in the
never-labelled region; never `dropna()` the target.

## 2. Outcome leakage (blank in test)
`default_flag, days_to_default, days_to_full_repayment, repayment_status,
final_recovered_amount, observation_status` are all post-outcome.
`prior_approved_amount` is the prior lender's post-decision output (44% null, set only
for approved). **All quarantined from features** (`src/data.py: LEAKAGE`).

## 3. Source-mixed engineered ratio (the planted "trust me" column)
`requested_amount_to_observed_revenue` looks clean (0 nulls) but is computed from
**two different denominators with no flag** — proven on train:

| rows | how the column was built |
|---|---|
| 54,887 (has bank feed) | `requested / (observed_monthly × 12)` — **verified** |
| 30,453 (no bank feed) | `requested / stated_annual_revenue` — **self-reported, gameable** |
| 0 | observed denom without a feed |

100% clean split → deliberate. A model treats a verified leverage number and an
inflatable self-reported one as the *same feature*. **Mitigation:** drop it; rebuild
`lev_verified` (NaN when no feed), `lev_stated`, and a `report_gap` inflation signal
as separate, labelled columns.

## 4. MNAR missingness
Bank-feed block (`observed_*`, `payroll_regularity_score`) is null for ~37% — exactly
the no-`has_linked_bank_feed` rows. `days_since_last_external_decline` /
`..._inquiry_elsewhere` ~49% null ("never happened" is a signal). **Mitigation:**
`*__isna` indicator columns + `no_bank_feed`; let HistGradientBoosting read NaN
natively. **No imputation** — imputing destroys the signal.

## 5. Right-censoring + temporal shift (drives B)
- train applications: 2024-01-01 → 2025-06-29 (the past)
- validation **and** test: 2025-06-30 → 2025-09-28 (the **same** 13 cohort weeks)

Late cohorts in that window have less time to mature → B needs survival/hazard
methods (cumulative default by cohort × age), not raw fractions.

## 6. Collider (drives C)
`prior_decision` (and its driver `prior_underwriter_score`) is a **selection node** —
fine as a *predictor* for A, **forbidden** as a conditioning variable for causal C.
`do(x)` must use backdoor adjustment, not naive re-prediction.

## 7. Integrity checks — ALL CLEAN (state this in the writeup as a deliberate misdirection)
- `prior_loans_default_count ≤ prior_loans_count`: 0 violations
- `days_to_default ∈ [1, 90]`: 0 violations
- `default_flag` ⟺ `repayment_status`: perfect agreement
- `business_id` / `applicant_id` spanning splits: **0 overlap** (no entity leakage)
- self-report vs observed where both exist: median ratio 0.979 (the *inflation*
  head-fake lives only in the unverifiable no-feed 37%)

## 8. Recovery / LGD (drives A's threshold)
Among defaults, `final_recovered_amount / principal` ≈ 9% (mean 0.093, median 0.073).
With the 3% origination fee kept, effective **LGD ≈ 0.88** ⇒ theoretical break-even
PD ≈ `0.0875 / (0.0875 + 0.88) ≈ 9%`. **Approve below the profit break-even, not 0.5.**
A tunes the exact threshold on *realized validation profit*.
