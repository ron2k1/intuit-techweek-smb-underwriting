#!/usr/bin/env python3
"""Diagnose correlated latent structure in the synthetic SMB data.

The dataset hint says many fields are related because the synthetic data is
based on real data. This report looks for latent factors, correlation clusters,
and whether explicit factor scores add signal beyond the engineered features.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DELIVERABLE_A_FEATURE_SET", "all_engineered")

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402


CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTCOME_COLUMNS = {
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
}
PRIOR_POLICY_TOKENS = (
    "prior_underwriter",
    "prior_decision",
    "prior_approved",
    "prior_score",
    "selection_support",
)


def time_split_labeled(train: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def connected_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    seen = set()
    components = []
    for node in graph:
        if node in seen:
            continue
        stack = [node]
        comp = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in graph.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return sorted(components, key=len, reverse=True)


def describe_component(component: list[str]) -> str:
    text = " ".join(component)
    if any(token in text for token in ["cash", "overdraft", "payroll", "revenue_trend", "volatility"]):
        return "cashflow/liquidity"
    if any(token in text for token in ["utilization", "delinquency", "inquiries", "debt", "default"]):
        return "credit_stress"
    if any(token in text for token in ["revenue", "requested", "amount", "burden"]):
        return "scale_or_burden"
    if any(token in text for token in ["age", "vintage", "time_in_business", "platform"]):
        return "maturity"
    return "mixed"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, _ = feature_columns(train_fe)
    numeric = [
        c
        for c in numeric
        if c not in OUTCOME_COLUMNS
        and not any(token in c for token in PRIOR_POLICY_TOKENS)
        and train_fe[c].nunique(dropna=True) > 3
    ]

    frames = [train_fe, validation_fe, test_fe]
    x_train = train_fe[numeric].replace([np.inf, -np.inf], np.nan)
    x_val = validation_fe[numeric].replace([np.inf, -np.inf], np.nan)
    labeled_idx, cal_idx = time_split_labeled(train)

    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    z_train = prep.fit_transform(x_train)
    z_val = prep.transform(x_val)
    z_labeled = z_train[labeled_idx]
    z_cal = z_train[cal_idx]

    pca = PCA(n_components=min(12, len(numeric)), random_state=2026)
    pca.fit(z_train)
    train_scores = pca.transform(z_train)
    val_scores = pca.transform(z_val)

    explained = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    loading_rows = []
    for i, component in enumerate(pca.components_[:8]):
        order = np.argsort(np.abs(component))[::-1][:12]
        for rank, j in enumerate(order, start=1):
            loading_rows.append(
                {
                    "component": f"PC{i+1}",
                    "rank": rank,
                    "feature": numeric[j],
                    "loading": float(component[j]),
                    "abs_loading": float(abs(component[j])),
                }
            )
    loadings = pd.DataFrame(loading_rows)

    corr_sample = pd.DataFrame(z_train, columns=numeric).sample(min(25000, len(z_train)), random_state=2026)
    corr = corr_sample.corr().abs()
    edges = []
    edge_rows = []
    for i, a in enumerate(numeric):
        vals = corr.iloc[i, i + 1 :]
        for b, value in vals[vals >= 0.60].items():
            edges.append((a, b))
            edge_rows.append({"feature_a": a, "feature_b": b, "abs_corr": float(value)})
    edge_df = pd.DataFrame(edge_rows).sort_values("abs_corr", ascending=False) if edge_rows else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
    components = connected_components(edges)
    component_rows = [
        {
            "cluster_id": i + 1,
            "size": len(comp),
            "theme": describe_component(comp),
            "features": ", ".join(comp),
        }
        for i, comp in enumerate(components)
    ]
    component_df = pd.DataFrame(component_rows)

    y_fit = train.loc[labeled_idx, "default_flag"].astype(int).to_numpy()
    y_cal = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()
    val_mask = validation["default_flag"].notna().to_numpy()
    y_val = validation.loc[val_mask, "default_flag"].astype(int).to_numpy()

    experiments = []
    for k in [3, 5, 8, 12]:
        clf = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
        clf.fit(train_scores[labeled_idx, :k], y_fit)
        p_cal = clf.predict_proba(train_scores[cal_idx, :k])[:, 1]
        p_val = clf.predict_proba(val_scores[val_mask, :k])[:, 1]
        row = {"model": f"logistic_pc_{k}", "n_features": k, **metric_summary(y_val, p_val)}
        row["cal_auroc"] = float(roc_auc_score(y_cal, p_cal))
        experiments.append(row)

    raw_clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.25, max_iter=1000, solver="lbfgs")),
        ]
    )
    raw_clf.fit(x_train.iloc[labeled_idx], y_fit)
    p_val_raw = raw_clf.predict_proba(x_val.loc[val_mask])[:, 1]
    experiments.append({"model": "logistic_raw_numeric", "n_features": len(numeric), **metric_summary(y_val, p_val_raw)})
    experiment_df = pd.DataFrame(experiments).sort_values("auroc", ascending=False)

    factor_effect_rows = []
    for i in range(min(8, train_scores.shape[1])):
        score = train_scores[:, i]
        labeled = train["default_flag"].notna().to_numpy()
        approved = train["prior_decision"].to_numpy() == 1
        defaulted = train["default_flag"].fillna(0).to_numpy() == 1
        factor_effect_rows.append(
            {
                "component": f"PC{i+1}",
                "mean_prior_approved": float(score[approved].mean()),
                "mean_prior_declined": float(score[~approved].mean()),
                "selection_smd": float((score[approved].mean() - score[~approved].mean()) / score.std(ddof=1)),
                "mean_defaulted_labeled": float(score[labeled & defaulted].mean()),
                "mean_nondefault_labeled": float(score[labeled & ~defaulted].mean()),
                "default_smd": float((score[labeled & defaulted].mean() - score[labeled & ~defaulted].mean()) / score[labeled].std(ddof=1)),
            }
        )
    factor_effects = pd.DataFrame(factor_effect_rows)

    explained.to_csv(REPORT_DIR / "latent_factor_explained_variance.csv", index=False)
    loadings.to_csv(REPORT_DIR / "latent_factor_loadings.csv", index=False)
    edge_df.to_csv(REPORT_DIR / "latent_factor_correlation_edges.csv", index=False)
    component_df.to_csv(REPORT_DIR / "latent_factor_correlation_clusters.csv", index=False)
    factor_effects.to_csv(REPORT_DIR / "latent_factor_selection_default_effects.csv", index=False)
    experiment_df.to_csv(REPORT_DIR / "latent_factor_predictive_experiment.csv", index=False)

    summary = {
        "numeric_features_used": int(len(numeric)),
        "pc1_explained_variance": float(explained.iloc[0]["explained_variance_ratio"]),
        "pc5_cumulative_explained_variance": float(explained.iloc[min(4, len(explained) - 1)]["cumulative_explained_variance"]),
        "correlation_edges_abs_ge_0p60": int(len(edge_df)),
        "correlation_clusters": int(len(component_df)),
        "largest_cluster_size": int(component_df.iloc[0]["size"]) if len(component_df) else 0,
        "top_predictive_factor_model": experiment_df.iloc[0].to_dict(),
        "interpretation": [
            "The engineered factors are useful for explanation and stability, but they are mostly low-dimensional summaries of raw correlated fields.",
            "Tree models already exploit much of this structure, so extra hand-built factors do not guarantee higher NPV.",
            "The strongest opportunity is to use latent factors for reject-region governance, causal shrinkage, and monotonic/scorecard constraints.",
        ],
    }
    (REPORT_DIR / "latent_factor_diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Latent Factor Diagnostics",
        "",
        "## Read",
        "- The sponsor hint is consistent with latent business-health factors behind correlated columns.",
        "- This report identifies correlation clusters and PCA-like factor axes using only hackathon data.",
        "- The conclusion is not that we need more raw feature engineering; it is that we should use latent factors for stable governance and reject-inference controls.",
        "",
        "## Key Numbers",
        f"- Numeric features used: {summary['numeric_features_used']}",
        f"- PC1 explained variance: {summary['pc1_explained_variance']:.3f}",
        f"- PC1-PC5 cumulative variance: {summary['pc5_cumulative_explained_variance']:.3f}",
        f"- Strong correlation edges abs(corr)>=0.60: {summary['correlation_edges_abs_ge_0p60']}",
        f"- Largest correlation cluster size: {summary['largest_cluster_size']}",
        f"- Best latent-factor-only model: {summary['top_predictive_factor_model']['model']} AUROC {summary['top_predictive_factor_model']['auroc']:.3f}",
        "",
        "## Files",
        "- `latent_factor_explained_variance.csv`",
        "- `latent_factor_loadings.csv`",
        "- `latent_factor_correlation_edges.csv`",
        "- `latent_factor_correlation_clusters.csv`",
        "- `latent_factor_selection_default_effects.csv`",
        "- `latent_factor_predictive_experiment.csv`",
    ]
    (REPORT_DIR / "latent_factor_diagnostics_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
