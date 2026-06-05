# EDA + Trap Audit — SMB Underwriting

## 0. Shapes & column groups

- train: 85,340 x 44
- validation: 4,489 x 44
- test: 8,817 x 44
- Deliverable A universe (val+test): 13,306

- **application_context** (6): application_timestamp, application_channel, multi_lender_inquiry_count_30d, days_since_last_inquiry_elsewhere, repeat_application_count, requested_amount_to_observed_revenue
- **bank_feed** (7): has_linked_bank_feed, observed_monthly_revenue_avg_3mo, observed_revenue_trend_3mo, observed_revenue_volatility, observed_cash_balance_p10, observed_overdraft_count_3mo, payroll_regularity_score
- **bureau_credit** (5): aggregate_credit_utilization, recent_inquiries_count_6mo, existing_debt_obligations, owner_personal_credit_band, days_since_last_external_decline
- **business_identity** (6): business_id, applicant_id, sector, geography_region, vintage_years, employee_count_bucket
- **outcome** (6): default_flag, days_to_default, days_to_full_repayment, repayment_status, final_recovered_amount, observation_status
- **platform_engagement** (7): account_age_days, platform_active_months, bookkeeping_recency_days, invoice_payment_delinquency_rate, prior_loans_count, prior_loans_default_count, prior_loans_amount_total
- **prior_underwriter** (3): prior_underwriter_score, prior_decision, prior_approved_amount
- **self_reported** (4): stated_annual_revenue, stated_time_in_business, requested_amount, intended_use_of_funds

## 1. Selection bias / reject inference

Outcomes (`default_flag`) exist only for prior-approved & matured loans. If we train on observed rows and score the full population, PD is biased.

### train (n=85,340)
- prior_decision: {1: 51722, 0: 33618}
- rows with default_flag observed: 51,722 (60.6%)
- observation_status: {'matured': 51722, nan: 33618}
- **observed default rate: 0.1745**

### validation (n=4,489)
- prior_decision: {1: 2551, 0: 1938}
- rows with default_flag observed: 2,551 (56.8%)
- observation_status: {'matured': 2551, nan: 1938}
- **observed default rate: 0.2062**

### test (n=8,817)
- prior_decision: {1: 4964, 0: 3853}
- rows with default_flag observed: 0 (0.0%)
- observation_status: {nan: 8817}

### Outcome availability by prior_decision (train)
```
has_outcome     False  True 
prior_decision              
0               33618      0
1                   0  51722
```
> If outcomes appear ONLY under one prior_decision value, that confirms the selection node: naive dropna()+train learns P(default | approved), not P(default | applied).

## 2. Outcome leakage

These columns are post-outcome and must never be features:
- default_flag, days_to_default, days_to_full_repayment, repayment_status, final_recovered_amount, observation_status

Null-rate of outcome columns per split (should be ~100% null in test):

```
split                     test  train    val
column                                      
days_to_default         100.0%  89.4%  88.3%
days_to_full_repayment  100.0%  50.0%  54.9%
default_flag            100.0%  39.4%  43.2%
final_recovered_amount  100.0%  89.4%  88.3%
observation_status      100.0%  39.4%  43.2%
repayment_status        100.0%  39.4%  43.2%
```

## 3. MNAR missingness (bank-feed)

- has_linked_bank_feed: {True: 54887, False: 30453}

Null rate of bank-feed columns (should track has_linked_bank_feed=False):
- observed_monthly_revenue_avg_3mo: 0.357 null
- observed_revenue_trend_3mo: 0.357 null
- observed_revenue_volatility: 0.357 null
- observed_cash_balance_p10: 0.357 null
- observed_overdraft_count_3mo: 0.357 null
- payroll_regularity_score: 0.357 null

### Is bank-feed missingness informative about default? (observed rows only)
```
                          mean  count
has_linked_bank_feed                 
False                 0.190737  16971
True                  0.166528  34751
```
> A gap here ⇒ missingness is MNAR and informative ⇒ add a `has_linked_bank_feed` / per-column missing indicator instead of blind imputation.

## 4. Self-report inflation (stated vs observed)

- rows with both stated & observed revenue: 54,887
- stated / (observed*12) ratio — median: 0.98, mean: 1.01, p90: 1.31
> Ratio >> 1 ⇒ stated revenue is optimistically inflated. Important for Deliverable C: do(stated_revenue=X) likely has ~0 true causal effect.

## 5. Planted integrity violations

- prior_loans_default_count > prior_loans_count: **0** rows
- days_to_default outside [1,90]: **0** rows (of 9,550 non-null)
- default_flag vs repayment_status (train, observed):
```
default_flag        0.0   1.0
repayment_status             
defaulted             0  9024
paid_in_full      42698     0
```
- business_id overlap train∩val: 0, train∩test: 0, val∩test: 0 (should be 0)
- requested_amount_to_observed_revenue mismatch vs raw (>1% rel err): **63,312** of 63,312 checked

## 6. Loan economics & break-even PD

- Net margin on a fully-repaid loan ≈ 0.0875 (35% APR × 60/365 + 3% fee)

- defaults in train: 9,024
- recovery fraction (recovered/amount) — median: 0.072, mean: 0.091
- implied LGD — median: 0.928, mean: 0.909
- break-even PD at LGD=0.91: **0.088**
- break-even PD at LGD=1.00: **0.080**
- break-even PD at LGD=0.50: **0.149**
> Approve when predicted_pd < break-even PD (≈8–15%), NOT < 0.5.

