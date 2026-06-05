from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deliverable_a_pipeline import (
    CSV_DIR,
    REPORT_DIR,
    add_application_features,
    feature_columns,
    metric_summary,
)


def time_split_labeled(labeled: pd.DataFrame, train_fraction: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    ordered = labeled.sort_values("application_timestamp").index.to_numpy()
    split_at = int(len(ordered) * train_fraction)
    return ordered[:split_at], ordered[split_at:]


def transformed_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool = False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
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


def rank_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prob = np.clip(score, 0.001, 0.999)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
    }


def train_sklearn_hgb(train_x, train_y, cal_x, val_x, numeric, categorical):
    pipe = Pipeline(
        [
            ("prep", transformed_preprocessor(numeric, categorical, scale_numeric=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=260,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    l2_regularization=0.03,
                    min_samples_leaf=35,
                    random_state=11,
                ),
            ),
        ]
    )
    pipe.fit(train_x, train_y)
    return {
        "name": "sklearn_hgb",
        "cal_raw": pipe.predict_proba(cal_x)[:, 1],
        "val_raw": pipe.predict_proba(val_x)[:, 1],
    }


def train_logistic(train_x, train_y, cal_x, val_x, numeric, categorical):
    pipe = Pipeline(
        [
            ("prep", transformed_preprocessor(numeric, categorical, scale_numeric=True)),
            ("clf", LogisticRegression(C=0.35, max_iter=1200)),
        ]
    )
    pipe.fit(train_x, train_y)
    return {
        "name": "logistic",
        "cal_raw": pipe.predict_proba(cal_x)[:, 1],
        "val_raw": pipe.predict_proba(val_x)[:, 1],
    }


def train_lightgbm(train_x, train_y, cal_x, val_x, categorical: list[str]):
    x_train = train_x.copy()
    x_cal = cal_x.copy()
    x_val = val_x.copy()
    for col in categorical:
        for frame in (x_train, x_cal, x_val):
            frame[col] = frame[col].astype("category")
    model = LGBMClassifier(
        objective="binary",
        n_estimators=850,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=55,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.30,
        random_state=17,
        verbosity=-1,
    )
    model.fit(x_train, train_y, categorical_feature=categorical)
    return {
        "name": "lightgbm",
        "cal_raw": model.predict_proba(x_cal)[:, 1],
        "val_raw": model.predict_proba(x_val)[:, 1],
    }


def train_catboost(train_x, train_y, cal_x, val_x, categorical: list[str]):
    x_train = train_x.copy()
    x_cal = cal_x.copy()
    x_val = val_x.copy()
    for col in categorical:
        for frame in (x_train, x_cal, x_val):
            frame[col] = frame[col].astype(str).fillna("__missing__")
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=29,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, train_y, cat_features=categorical)
    return {
        "name": "catboost",
        "cal_raw": model.predict_proba(x_cal)[:, 1],
        "val_raw": model.predict_proba(x_val)[:, 1],
    }


def optimize_blend(y_cal: np.ndarray, pred_map: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray]:
    names = list(pred_map)
    best_weights = None
    best_score = -np.inf

    if len(names) == 1:
        return {names[0]: 1.0}, pred_map[names[0]]

    grid = np.linspace(0, 1, 11)
    if len(names) == 4:
        candidates = []
        for a in grid:
            for b in grid:
                for c in grid:
                    d = 1 - a - b - c
                    if d < -1e-9:
                        continue
                    candidates.append([a, b, c, max(0, d)])
    else:
        candidates = []
        for a in grid:
            for b in grid:
                c = 1 - a - b
                if c < -1e-9:
                    continue
                candidates.append([a, b, max(0, c)])

    for weights in candidates:
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        blended = sum(w * pred_map[n] for w, n in zip(weights, names))
        auc = roc_auc_score(y_cal, blended)
        if auc > best_score:
            best_score = auc
            best_weights = weights

    assert best_weights is not None
    return {n: float(w) for n, w in zip(names, best_weights)}, sum(
        w * pred_map[n] for w, n in zip(best_weights, names)
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(CSV_DIR / "train.csv")
    validation = pd.read_csv(CSV_DIR / "validation.csv")

    train_fe = add_application_features(train)
    validation_fe = add_application_features(validation)
    _, numeric, categorical = feature_columns(train_fe)
    train_x_all = train_fe[numeric + categorical]
    validation_x = validation_fe[numeric + categorical]

    labeled = train[train["default_flag"].notna()].copy()
    model_idx, cal_idx = time_split_labeled(labeled)
    train_x = train_x_all.loc[model_idx]
    train_y = train.loc[model_idx, "default_flag"].astype(int)
    cal_x = train_x_all.loc[cal_idx]
    cal_y = train.loc[cal_idx, "default_flag"].astype(int).to_numpy()

    val_mask = validation["default_flag"].notna()
    val_y = validation.loc[val_mask, "default_flag"].astype(int).to_numpy()
    val_x_labeled = validation_x.loc[val_mask]

    results = []
    models = []
    for trainer in [
        lambda: train_sklearn_hgb(train_x, train_y, cal_x, val_x_labeled, numeric, categorical),
        lambda: train_logistic(train_x, train_y, cal_x, val_x_labeled, numeric, categorical),
        lambda: train_lightgbm(train_x, train_y, cal_x, val_x_labeled, categorical),
        lambda: train_catboost(train_x, train_y, cal_x, val_x_labeled, categorical),
    ]:
        model_result = trainer()
        models.append(model_result)
        raw_metrics = rank_metrics(val_y, model_result["val_raw"])
        raw_metrics["model"] = model_result["name"]
        raw_metrics["score_type"] = "raw"

        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        calibrator.fit(model_result["cal_raw"], cal_y)
        cal_val = calibrator.predict(model_result["val_raw"])
        cal_metrics = metric_summary(val_y, np.clip(cal_val, 0.001, 0.999))
        cal_metrics["model"] = model_result["name"]
        cal_metrics["score_type"] = "isotonic"
        results.extend([raw_metrics, cal_metrics])

    cal_pred_map = {m["name"]: m["cal_raw"] for m in models}
    val_pred_map = {m["name"]: m["val_raw"] for m in models}
    weights, _ = optimize_blend(cal_y, cal_pred_map)
    raw_blend_val = sum(weights[name] * val_pred_map[name] for name in weights)
    raw_blend_metrics = rank_metrics(val_y, raw_blend_val)
    raw_blend_metrics["model"] = "raw_score_blend"
    raw_blend_metrics["score_type"] = "raw"
    raw_blend_metrics["weights"] = json.dumps(weights)

    raw_blend_cal = sum(weights[name] * cal_pred_map[name] for name in weights)
    blend_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    blend_calibrator.fit(raw_blend_cal, cal_y)
    calibrated_blend_val = np.clip(blend_calibrator.predict(raw_blend_val), 0.001, 0.999)
    calibrated_blend_metrics = metric_summary(val_y, calibrated_blend_val)
    calibrated_blend_metrics["model"] = "raw_score_blend"
    calibrated_blend_metrics["score_type"] = "isotonic"
    calibrated_blend_metrics["weights"] = json.dumps(weights)
    results.extend([raw_blend_metrics, calibrated_blend_metrics])

    table = pd.DataFrame(results)
    table = table.sort_values(["auroc", "average_precision"], ascending=False)
    out_path = REPORT_DIR / "raw_score_ensemble_experiment.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
