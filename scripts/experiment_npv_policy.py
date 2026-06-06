#!/usr/bin/env python3
"""Experiment with direct realized-NPV policy models.

Motivation:
The official objective is realized portfolio value, not AUROC. The current
production policy decomposes value into PD + timing + recovery. This experiment
adds direct value models trained on observed historical cash-flow NPV and tests
whether they improve validation realized NPV without using outcome columns as
features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


def make_preprocessor(numeric: list[str], categorical: list[str], *, scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )


def policy_table(
    scores: dict[str, np.ndarray],
    validation_realized: np.ndarray,
    amount: np.ndarray,
    current_decision: np.ndarray,
) -> pd.DataFrame:
    rows = []
    candidates = np.unique(
        np.r_[
            np.linspace(-0.10, 0.12, 45),
            np.quantile(scores["decomposed_margin"], np.linspace(0.02, 0.98, 49)),
        ]
    )
    for name, score in scores.items():
        for threshold in candidates:
            decision = score > threshold
            if decision.sum() == 0:
                continue
            rows.append(
                {
                    "policy": name,
                    "threshold": float(threshold),
                    "approved": int(decision.sum()),
                    "approval_rate": float(decision.mean()),
                    "realized_npv": float(validation_realized[decision].sum()),
                    "mean_realized_npv_approved": float(validation_realized[decision].mean()),
                    "observed_default_rate_approved": float(np.mean(validation_realized[decision] < 0)),
                    "overlap_with_current_decisions": float(np.mean(decision == current_decision)),
                }
            )
    return pd.DataFrame(rows).sort_values("realized_npv", ascending=False)


def main() -> None:
    root = Path(".")
    report_dir = root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(root / "data" / "csv-files" / "train.csv")
    validation = pd.read_csv(root / "data" / "csv-files" / "validation.csv")
    test = pd.read_csv(root / "data" / "csv-files" / "test.csv")
    submission_a = pd.read_csv(root / "outputs" / "submission" / "submission_A_decisions.csv")
    curves = np.load(root / "outputs" / "deliverable_a_curves.npz")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)
    _, numeric, categorical = feature_columns(train_fe)
    feature_cols = numeric + categorical

    labeled_train = train["default_flag"].notna()
    labeled_val = validation["default_flag"].notna()
    x_train = train_fe.loc[labeled_train, feature_cols]
    x_val = validation_fe.loc[labeled_val, feature_cols]

    y_train_npv = realized_npv(train.loc[labeled_train])
    amount_train = train.loc[labeled_train, "requested_amount"].to_numpy(float)
    y_train_margin = y_train_npv / np.maximum(amount_train, 1.0)
    y_train_profitable = (y_train_npv > 0).astype(int)

    y_val_npv = realized_npv(validation.loc[labeled_val])
    amount_val = validation.loc[labeled_val, "requested_amount"].to_numpy(float)
    current_decision_val = (
        submission_a.iloc[: len(validation)]["decision"].to_numpy(int)[labeled_val.to_numpy()] == 1
    )

    decomposed_npv_val = expected_npv(
        validation.loc[labeled_val, "requested_amount"].to_numpy(float),
        curves["validation_pd"][labeled_val.to_numpy()],
        curves["validation_t_star"][labeled_val.to_numpy()],
        curves["validation_recovery"][labeled_val.to_numpy()],
    )
    decomposed_margin_val = decomposed_npv_val / np.maximum(amount_val, 1.0)

    hgb_margin = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_iter=260,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    min_samples_leaf=45,
                    l2_regularization=0.08,
                    loss="squared_error",
                    random_state=701,
                ),
            ),
        ]
    )
    hgb_abs = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            (
                "reg",
                HistGradientBoostingRegressor(
                    max_iter=260,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    min_samples_leaf=45,
                    l2_regularization=0.08,
                    loss="absolute_error",
                    random_state=703,
                ),
            ),
        ]
    )
    ridge = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=True)),
            ("reg", Ridge(alpha=80.0)),
        ]
    )
    huber = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=True)),
            ("reg", HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=500)),
        ]
    )
    profit_clf = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=220,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=45,
                    l2_regularization=0.08,
                    random_state=709,
                ),
            ),
        ]
    )
    logit_profit = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale=True)),
            ("clf", LogisticRegression(C=0.15, max_iter=1000)),
        ]
    )

    models = {
        "direct_margin_hgb_squared": hgb_margin,
        "direct_margin_hgb_absolute": hgb_abs,
        "direct_margin_ridge": ridge,
        "direct_margin_huber": huber,
    }
    pred_scores = {"decomposed_margin": decomposed_margin_val}
    diagnostics = []
    for name, model in models.items():
        model.fit(x_train, y_train_margin)
        pred = model.predict(x_val)
        pred_scores[name] = pred
        diagnostics.append(
            {
                "model": name,
                "target": "realized_npv_per_dollar",
                "mae_margin": float(mean_absolute_error(y_val_npv / np.maximum(amount_val, 1.0), pred)),
                "corr_with_realized_margin": float(np.corrcoef(y_val_npv / np.maximum(amount_val, 1.0), pred)[0, 1]),
            }
        )

    for name, model in {"profitable_hgb": profit_clf, "profitable_logistic": logit_profit}.items():
        model.fit(x_train, y_train_profitable)
        pred = model.predict_proba(x_val)[:, 1]
        pred_scores[name] = pred
        diagnostics.append(
            {
                "model": name,
                "target": "realized_npv_positive",
                "auroc_profitable": float(roc_auc_score((y_val_npv > 0).astype(int), pred)),
                "corr_with_realized_margin": float(np.corrcoef(y_val_npv / np.maximum(amount_val, 1.0), pred)[0, 1]),
            }
        )

    # Blends: decomposed economics plus learned direct value/profit support.
    for alpha in np.linspace(0.1, 0.9, 9):
        pred_scores[f"blend_decomp_direct_hgb_{alpha:.1f}"] = (
            (1 - alpha) * decomposed_margin_val + alpha * pred_scores["direct_margin_hgb_squared"]
        )
    # Profit classifier is on [0,1], so convert to centered support term.
    for gamma in np.linspace(0.02, 0.12, 6):
        pred_scores[f"decomp_plus_profit_support_{gamma:.2f}"] = (
            decomposed_margin_val + gamma * (pred_scores["profitable_hgb"] - 0.5)
        )

    table = policy_table(pred_scores, y_val_npv, amount_val, current_decision_val)
    table.to_csv(report_dir / "direct_npv_policy_experiment.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(report_dir / "direct_npv_model_diagnostics.csv", index=False)

    current_row = {
        "policy": "active_submission",
        "threshold": np.nan,
        "approved": int(current_decision_val.sum()),
        "approval_rate": float(current_decision_val.mean()),
        "realized_npv": float(y_val_npv[current_decision_val].sum()),
        "mean_realized_npv_approved": float(y_val_npv[current_decision_val].mean()),
        "observed_default_rate_approved": float(np.mean(y_val_npv[current_decision_val] < 0)),
        "overlap_with_current_decisions": 1.0,
    }
    best = table.iloc[0].to_dict()
    summary = {
        "current_policy": current_row,
        "best_candidate": best,
        "oracle_labeled_validation_npv": float(y_val_npv[y_val_npv > 0].sum()),
        "approve_all_labeled_validation_npv": float(y_val_npv.sum()),
        "caveat": "Thresholds are selected on labeled validation; use as a policy lead, not proof of test performance.",
    }
    (report_dir / "direct_npv_policy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

