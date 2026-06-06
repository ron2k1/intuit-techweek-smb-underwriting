#!/usr/bin/env python3
"""Model-family bakeoff scored as an NPV approval policy.

This is deliberately not just an AUROC leaderboard. Each model family produces
calibrated PDs, those PDs are converted to expected NPV using the active
timing/recovery curves, and approval buffers are selected by labeled-validation
realized NPV.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deliverable_a_pipeline import add_application_features, feature_columns  # noqa: E402
from src.economics import expected_npv, realized_npv  # noqa: E402


REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CSV_DIR = PROJECT_ROOT / "data" / "csv-files"
CURVES_PATH = PROJECT_ROOT / "outputs" / "deliverable_a_curves.npz"
SUBMISSION_A_PATH = PROJECT_ROOT / "outputs" / "submission" / "submission_A_decisions.csv"

DROP_PRIOR_SCORE_PROXY_TOKENS = (
    "prior_underwriter",
    "prior_decision",
    "prior_approved",
    "prior_score",
    "selection_support",
)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def time_split_labeled(train: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    labeled = train[train["default_flag"].notna()].copy()
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def feature_set_columns(train_fe: pd.DataFrame, feature_set: str) -> tuple[list[str], list[str], list[str]]:
    _, numeric, categorical = feature_columns(train_fe)
    if feature_set == "current_predictive":
        return numeric + categorical, numeric, categorical
    if feature_set == "no_prior_score":
        numeric = [c for c in numeric if not any(token in c for token in DROP_PRIOR_SCORE_PROXY_TOKENS)]
        categorical = [c for c in categorical if not any(token in c for token in DROP_PRIOR_SCORE_PROXY_TOKENS)]
        return numeric + categorical, numeric, categorical
    raise ValueError(feature_set)


def fit_isotonic(raw_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(raw_cal, y_cal)
    return calibrator


def calibrate(calibrator: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    return np.clip(calibrator.predict(raw), 0.001, 0.999)


def train_sklearn_hgb(train_x, train_y, cal_x, cal_y, val_x, test_x, numeric, categorical):
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    min_samples_leaf=35,
                    l2_regularization=0.04,
                    random_state=1101,
                ),
            ),
        ]
    )
    pipe.fit(train_x, train_y)
    raw_cal = pipe.predict_proba(cal_x)[:, 1]
    iso = fit_isotonic(raw_cal, cal_y)
    return {
        "cal": calibrate(iso, raw_cal),
        "val": calibrate(iso, pipe.predict_proba(val_x)[:, 1]),
        "test": calibrate(iso, pipe.predict_proba(test_x)[:, 1]),
    }


def train_lightgbm(train_x, train_y, cal_x, cal_y, val_x, test_x, categorical):
    frames = [train_x.copy(), cal_x.copy(), val_x.copy(), test_x.copy()]
    for col in categorical:
        for frame in frames:
            frame[col] = frame[col].astype("category")
    x_train, x_cal, x_val, x_test = frames
    model = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.35,
        random_state=1103,
        verbosity=-1,
    )
    model.fit(x_train, train_y, categorical_feature=categorical)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    iso = fit_isotonic(raw_cal, cal_y)
    return {
        "cal": calibrate(iso, raw_cal),
        "val": calibrate(iso, model.predict_proba(x_val)[:, 1]),
        "test": calibrate(iso, model.predict_proba(x_test)[:, 1]),
    }


def train_catboost(train_x, train_y, cal_x, cal_y, val_x, test_x, categorical):
    frames = [train_x.copy(), cal_x.copy(), val_x.copy(), test_x.copy()]
    for col in categorical:
        for frame in frames:
            frame[col] = frame[col].astype(str).fillna("__missing__")
    x_train, x_cal, x_val, x_test = frames
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=1105,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, train_y, cat_features=categorical)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    iso = fit_isotonic(raw_cal, cal_y)
    return {
        "cal": calibrate(iso, raw_cal),
        "val": calibrate(iso, model.predict_proba(x_val)[:, 1]),
        "test": calibrate(iso, model.predict_proba(x_test)[:, 1]),
    }


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_pd": float(np.mean(p)),
        "actual_default_rate": float(np.mean(y)),
    }


def policy_sweep(
    score_margin: np.ndarray,
    realized: np.ndarray,
    current_decision: np.ndarray | None = None,
) -> pd.DataFrame:
    candidates = np.unique(
        np.r_[
            np.linspace(-0.05, 0.08, 53),
            np.quantile(score_margin, np.linspace(0.01, 0.99, 79)),
        ]
    )
    rows = []
    for threshold in candidates:
        decision = score_margin > threshold
        if decision.sum() == 0:
            continue
        row = {
            "threshold": float(threshold),
            "approved": int(decision.sum()),
            "approval_rate": float(decision.mean()),
            "realized_npv": float(realized[decision].sum()),
            "mean_realized_npv_approved": float(realized[decision].mean()),
            "observed_default_rate_approved": float(np.mean(realized[decision] < 0)),
        }
        if current_decision is not None:
            row["overlap_with_current_decisions"] = float(np.mean(decision == current_decision))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("realized_npv", ascending=False)


def build_policy_row(
    name: str,
    feature_set: str,
    val_pd: np.ndarray,
    test_pd: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    curves: np.lib.npyio.NpzFile,
    labeled_val_mask: np.ndarray,
    realized_val: np.ndarray,
    current_labeled_decision: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    val_enpv = expected_npv(
        validation["requested_amount"].to_numpy(float),
        val_pd,
        curves["validation_t_star"],
        curves["validation_recovery"],
    )
    test_enpv = expected_npv(
        test["requested_amount"].to_numpy(float),
        test_pd,
        curves["test_t_star"],
        curves["test_recovery"],
    )
    val_margin = val_enpv / np.maximum(validation["requested_amount"].to_numpy(float), 1.0)
    test_margin = test_enpv / np.maximum(test["requested_amount"].to_numpy(float), 1.0)
    sweep = policy_sweep(
        val_margin[labeled_val_mask],
        realized_val,
        current_decision=current_labeled_decision,
    )
    best = sweep.iloc[0]
    threshold = float(best["threshold"])
    val_decision = val_margin > threshold
    test_decision = test_margin > threshold
    all_prior_decision = pd.concat([validation["prior_decision"], test["prior_decision"]], ignore_index=True)
    all_decision = np.r_[val_decision, test_decision]
    prior_declined = all_prior_decision.to_numpy() == 0
    row = {
        "model": name,
        "feature_set": feature_set,
        "best_threshold_margin": threshold,
        "validation_labeled_realized_npv": float(best["realized_npv"]),
        "validation_labeled_approved": int(best["approved"]),
        "validation_labeled_approval_rate": float(best["approval_rate"]),
        "validation_labeled_default_rate_approved": float(best["observed_default_rate_approved"]),
        "overlap_with_current_labeled_decisions": float(best.get("overlap_with_current_decisions", math.nan)),
        "validation_all_approval_rate": float(val_decision.mean()),
        "test_approval_rate": float(test_decision.mean()),
        "approved_total": int(all_decision.sum()),
        "prior_declined_approved_total": int((all_decision & prior_declined).sum()),
        "prior_declined_approval_rate": float((all_decision & prior_declined).sum() / max(prior_declined.sum(), 1)),
        "mean_validation_pd": float(np.mean(val_pd)),
        "mean_test_pd": float(np.mean(test_pd)),
    }
    return row, sweep, val_decision, test_decision


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")
    test = pd.read_csv(CSV_DIR / "test.csv")
    curves = np.load(CURVES_PATH)
    current_submission = pd.read_csv(SUBMISSION_A_PATH)

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    test_fe = add_application_features(test)

    model_idx, cal_idx = time_split_labeled(train)
    val_mask = validation["default_flag"].notna().to_numpy()
    val_y = validation.loc[val_mask, "default_flag"].astype(int).to_numpy()
    realized_val = realized_npv(validation.loc[val_mask])
    current_labeled_decision = (
        current_submission.iloc[: len(validation)]["decision"].to_numpy(int)[val_mask] == 1
    )

    all_rows: list[dict[str, object]] = []
    all_sweeps: list[pd.DataFrame] = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for feature_set in ["current_predictive", "no_prior_score"]:
        feature_cols, numeric, categorical = feature_set_columns(train_fe, feature_set)
        train_x_all = train_fe[feature_cols]
        validation_x = validation_fe[feature_cols]
        test_x = test_fe[feature_cols]
        train_x = train_x_all.loc[model_idx]
        train_y = train.loc[model_idx, "default_flag"].astype(int)
        cal_x = train_x_all.loc[cal_idx]
        cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()

        trainers = {
            "hgb": lambda: train_sklearn_hgb(train_x, train_y, cal_x, cal_y, validation_x, test_x, numeric, categorical),
            "lightgbm": lambda: train_lightgbm(train_x, train_y, cal_x, cal_y, validation_x, test_x, categorical),
            "catboost": lambda: train_catboost(train_x, train_y, cal_x, cal_y, validation_x, test_x, categorical),
        }
        feature_predictions: dict[str, dict[str, np.ndarray]] = {}
        for model_name, trainer in trainers.items():
            pred = trainer()
            feature_predictions[model_name] = pred
            predictions[(feature_set, model_name)] = (pred["val"], pred["test"])
            metrics = metric_summary(val_y, pred["val"][val_mask])
            row, sweep, _, _ = build_policy_row(
                model_name,
                feature_set,
                pred["val"],
                pred["test"],
                validation,
                test,
                curves,
                val_mask,
                realized_val,
                current_labeled_decision,
            )
            row.update(metrics)
            all_rows.append(row)
            sweep.insert(0, "model", model_name)
            sweep.insert(1, "feature_set", feature_set)
            all_sweeps.append(sweep)

        blend_val = np.mean([feature_predictions[m]["val"] for m in ["hgb", "lightgbm", "catboost"]], axis=0)
        blend_test = np.mean([feature_predictions[m]["test"] for m in ["hgb", "lightgbm", "catboost"]], axis=0)
        predictions[(feature_set, "family_mean_blend")] = (blend_val, blend_test)
        metrics = metric_summary(val_y, blend_val[val_mask])
        row, sweep, _, _ = build_policy_row(
            "family_mean_blend",
            feature_set,
            blend_val,
            blend_test,
            validation,
            test,
            curves,
            val_mask,
            realized_val,
            current_labeled_decision,
        )
        row.update(metrics)
        all_rows.append(row)
        sweep.insert(0, "model", "family_mean_blend")
        sweep.insert(1, "feature_set", feature_set)
        all_sweeps.append(sweep)

    active_val_pd = curves["validation_pd"]
    active_test_pd = curves["test_pd"]
    active_metrics = metric_summary(val_y, active_val_pd[val_mask])
    active_row, active_sweep, _, _ = build_policy_row(
        "active_curves_pd_retuned",
        "active_project",
        active_val_pd,
        active_test_pd,
        validation,
        test,
        curves,
        val_mask,
        realized_val,
        current_labeled_decision,
    )
    active_row.update(active_metrics)
    active_row["note"] = "active PD curves retuned by validation NPV margin"
    all_rows.append(active_row)
    active_sweep.insert(0, "model", "active_curves_pd_retuned")
    active_sweep.insert(1, "feature_set", "active_project")
    all_sweeps.append(active_sweep)

    active_decisions = current_submission.iloc[: len(validation)]["decision"].to_numpy(int)[val_mask] == 1
    active_submission_row = {
        "model": "active_submission_direct_npv_blend",
        "feature_set": "active_project",
        "best_threshold_margin": np.nan,
        "validation_labeled_realized_npv": float(realized_val[active_decisions].sum()),
        "validation_labeled_approved": int(active_decisions.sum()),
        "validation_labeled_approval_rate": float(active_decisions.mean()),
        "validation_labeled_default_rate_approved": float(np.mean(realized_val[active_decisions] < 0)),
        "overlap_with_current_labeled_decisions": 1.0,
        "validation_all_approval_rate": float(current_submission.iloc[: len(validation)]["decision"].mean()),
        "test_approval_rate": float(current_submission.iloc[len(validation) :]["decision"].mean()),
        "approved_total": int(current_submission["decision"].sum()),
        "prior_declined_approved_total": np.nan,
        "prior_declined_approval_rate": np.nan,
        **active_metrics,
        "note": "actual current submitted decisions",
    }
    all_rows.append(active_submission_row)

    results = pd.DataFrame(all_rows).sort_values(
        ["validation_labeled_realized_npv", "auroc"], ascending=False
    )
    sweeps = pd.concat(all_sweeps, ignore_index=True)
    results_path = REPORT_DIR / "model_family_npv_bakeoff.csv"
    sweeps_path = REPORT_DIR / "model_family_npv_bakeoff_threshold_sweeps.csv"
    summary_path = REPORT_DIR / "model_family_npv_bakeoff_summary.json"
    results.to_csv(results_path, index=False)
    sweeps.to_csv(sweeps_path, index=False)
    summary = {
        "best_by_validation_npv": results.iloc[0].to_dict(),
        "active_submission": active_submission_row,
        "caveats": [
            "Thresholds are selected on labeled validation rows, so the top row is an optimization lead rather than proof of test performance.",
            "Prior-declined applicants still have no observed outcomes; reject-region funding should be defended with sensitivity analysis.",
            "The active submission can be promoted with scripts/apply_lightgbm_no_prior_policy.py.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(results.head(20).to_string(index=False))
    print("Wrote", results_path)
    print("Wrote", sweeps_path)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
