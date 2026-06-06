# Latent Factor Diagnostics

## Read
- The sponsor hint is consistent with latent business-health factors behind correlated columns.
- This report identifies correlation clusters and PCA-like factor axes using only hackathon data.
- The conclusion is not that we need more raw feature engineering; it is that we should use latent factors for stable governance and reject-inference controls.

## Key Numbers
- Numeric features used: 59
- PC1 explained variance: 0.236
- PC1-PC5 cumulative variance: 0.581
- Strong correlation edges abs(corr)>=0.60: 126
- Largest correlation cluster size: 34
- Best latent-factor-only model: logistic_raw_numeric AUROC 0.757

## Files
- `latent_factor_explained_variance.csv`
- `latent_factor_loadings.csv`
- `latent_factor_correlation_edges.csv`
- `latent_factor_correlation_clusters.csv`
- `latent_factor_selection_default_effects.csv`
- `latent_factor_predictive_experiment.csv`
