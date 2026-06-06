# Deliverable C Causal Accuracy Proxy Audit

True counterfactual labels are hidden, so this audit checks local failure modes that would make C easy to penalize.

## Summary

```json
{
  "rows_expected": 900,
  "rows_submitted": 900,
  "unique_query_ids": 900,
  "interval_order_violations": 0,
  "duplicate_intervention_groups": 4,
  "duplicate_predictions_consistent": true,
  "features_with_treatment_rules": 30,
  "treatment_missing_features": [],
  "outside_train_min_max_queries": 0,
  "tail_support_queries": 74,
  "unseen_category_queries": 0,
  "raw_material_sign_violations": 93,
  "monotonic_guard_applied": 93,
  "final_material_sign_violations": 0,
  "mean_interval_width": 0.11382407340682439,
  "mean_abs_delta_final": 0.016699952275135866,
  "largest_abs_delta_final": 0.4281317081156084,
  "classes": {
    "amount_burden_intervention": 40,
    "application_context_proxy": 79,
    "credit_context_intervention": 49,
    "credit_state_intervention": 144,
    "historical_or_proxy": 165,
    "measurement_process_intervention": 9,
    "observed_business_state": 260,
    "platform_state_intervention": 50,
    "self_report_proxy": 104
  }
}
```

## Feature-Level Audit

| feature_name | count | intervention_class | outside_min_max_rate | outside_p01_p99_rate | unseen_category_rate | raw_sign_violation_count | monotonic_guard_count | sign_violation_count | mean_delta_final | mean_interval_width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stated_time_in_business | 68 | self_report_proxy | 0.0000 | 0.0000 | 0.0000 | 22 | 22 | 0 | -0.0004 | 0.1315 |
| observed_revenue_trend_3mo | 53 | observed_business_state | 0.0000 | 0.0000 | 0.0000 | 4 | 4 | 0 | -0.0002 | 0.0887 |
| existing_debt_obligations | 52 | credit_state_intervention | 0.0000 | 0.0000 | 0.0000 | 26 | 26 | 0 | -0.0009 | 0.0967 |
| observed_revenue_volatility | 52 | observed_business_state | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | -0.0070 | 0.0852 |
| observed_cash_balance_p10 | 51 | observed_business_state | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | -0.0124 | 0.0928 |
| invoice_payment_delinquency_rate | 50 | platform_state_intervention | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | -0.0311 | 0.0965 |
| recent_inquiries_count_6mo | 49 | credit_context_intervention | 0.0000 | 0.5510 | 0.0000 | 3 | 3 | 0 | 0.0021 | 0.1361 |
| aggregate_credit_utilization | 47 | credit_state_intervention | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | -0.0249 | 0.0740 |
| observed_overdraft_count_3mo | 47 | observed_business_state | 0.0000 | 0.3617 | 0.0000 | 0 | 0 | 0 | 0.0453 | 0.1373 |
| application_channel | 45 | application_context_proxy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0001 | 0.1239 |
| owner_personal_credit_band | 45 | credit_state_intervention | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0020 | 0.0925 |
| observed_monthly_revenue_avg_3mo | 42 | observed_business_state | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | -0.0033 | 0.1037 |
| requested_amount | 40 | amount_burden_intervention | 0.0000 | 0.0000 | 0.0000 | 2 | 2 | 0 | -0.0105 | 0.0995 |
| stated_annual_revenue | 36 | self_report_proxy | 0.0000 | 0.0000 | 0.0000 | 7 | 7 | 0 | 0.0050 | 0.1450 |
| multi_lender_inquiry_count_30d | 34 | application_context_proxy | 0.0000 | 0.3529 | 0.0000 | 7 | 7 | 0 | 0.0011 | 0.1299 |
| account_age_days | 20 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 7 | 7 | 0 | -0.0013 | 0.1498 |
| platform_active_months | 19 | historical_or_proxy | 0.0000 | 0.4211 | 0.0000 | 3 | 3 | 0 | -0.0009 | 0.1517 |
| days_since_last_inquiry_elsewhere | 16 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | 0.0000 | 0.1281 |
| payroll_regularity_score | 15 | observed_business_state | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | -0.0013 | 0.1006 |
| employee_count_bucket | 14 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | -0.0004 | 0.1317 |
| prior_loans_default_count | 13 | historical_or_proxy | 0.0000 | 0.3846 | 0.0000 | 1 | 1 | 0 | 0.0016 | 0.1384 |
| intended_use_of_funds | 12 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0006 | 0.1279 |
| sector | 12 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0006 | 0.1094 |
| vintage_years | 12 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 2 | 2 | 0 | -0.0005 | 0.1178 |
| bookkeeping_recency_days | 11 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 2 | 2 | 0 | -0.0002 | 0.1384 |
| geography_region | 11 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0005 | 0.1313 |
| days_since_last_external_decline | 9 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 1 | 1 | 0 | -0.0022 | 0.1353 |
| has_linked_bank_feed | 9 | measurement_process_intervention | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | -0.0018 | 0.2046 |
| prior_loans_amount_total | 9 | historical_or_proxy | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.0018 | 0.1163 |
| prior_loans_count | 7 | historical_or_proxy | 0.0000 | 0.7143 | 0.0000 | 0 | 0 | 0 | 0.0002 | 0.1504 |

## Interpretation

- No local audit can prove true causal accuracy because the scoring interventions are hidden.
- The submitted C file is structurally complete, duplicate-deterministic, and support-aware.
- Material opposite-sign effects for monotone risk features are neutralized and intervals widened.
- Historical/proxy and measurement-process features are intentionally shrunk toward baseline rather than overclaimed.
