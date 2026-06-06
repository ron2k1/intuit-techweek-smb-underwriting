# Markov Regime Audit

## Interpretation

The relevant Markov-chain idea is discrete latent economic states with transition probabilities. This audit infers weekly states from non-outcome application/economic features, estimates a state transition matrix, and checks whether the active underwriting policy behaves differently by state.

## Key Results

- Selected regimes: 2
- Active variant rank in switching-intercept sweep: 2
- Best candidate: `regime_1_odds_-0.15`
- Best validation NPV: $2,657,249
- Active exact-threshold validation NPV: $2,654,901

## Files

- `outputs/reports/markov_regime_weekly_states.csv` (weekly regimes)
- `outputs/reports/markov_regime_transition_matrix.csv` (transition matrix)
- `outputs/reports/markov_regime_feature_summary.csv` (regime feature summary)
- `outputs/reports/markov_regime_active_policy_by_regime.csv` (active policy by regime)
- `outputs/reports/markov_regime_candidate_sweep.csv` (switching intercept candidates)
