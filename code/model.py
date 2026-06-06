"""
model.py — CatBoost training utilities.

Public API
----------
run_cv(X_train, y_train, X_val_all, X_test, cat_indices, ...)
    -> (oof_preds, val_preds_folds, test_preds_folds, models)

train_final_model(X_train, y_train, cat_indices, ensemble=True, n_ensemble=5, ...)
    -> CatBoostEnsemble  (ensemble=True, default)  — averages N models with varied seeds/splits
    -> CatBoostClassifier (ensemble=False)          — single model, original behaviour

CatBoostEnsemble
    Thin wrapper that exposes predict_proba() and tree_count_ so it is a drop-in
    replacement for a bare CatBoostClassifier everywhere downstream (bootstrap,
    calibration, decision).
"""

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from config import CB_PARAMS, N_FOLDS, SEED


# ── Ensemble wrapper ──────────────────────────────────────────────────────────

class CatBoostEnsemble:
    """
    Averages predictions from multiple CatBoostClassifiers.

    Acts as a drop-in replacement for a bare CatBoostClassifier: exposes
    predict_proba(X) and tree_count_ so all downstream code (bootstrap
    calibration, decision module) works without modification.
    """

    def __init__(self, models: list):
        self.models      = models
        # Representative tree count: mean across members (used by bootstrap to
        # set iteration budget for its own resampled models)
        self.tree_count_ = int(round(np.mean([m.tree_count_ for m in models])))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average class-probability matrices across all ensemble members."""
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def __len__(self):
        return len(self.models)

    def __repr__(self):
        return (f'CatBoostEnsemble({len(self.models)} models, '
                f'~{self.tree_count_} trees each)')


# ── Internal helper ───────────────────────────────────────────────────────────

def _train_single(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cat_indices: list,
    cb_params: dict,
    seed: int,
    val_fraction: float,
) -> CatBoostClassifier:
    """
    Train one CatBoostClassifier on all of X_train.

    Uses a stratified holdout (val_fraction) drawn with the given seed for
    early stopping to discover the right tree count, then refits on 100% of
    the data at that fixed count.
    """
    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train, y_train,
        test_size=val_fraction, random_state=seed, stratify=y_train,
    )
    # dict-merge so random_seed isn't passed twice (CB_PARAMS already contains it)
    es_params = {**cb_params, 'random_seed': seed}
    model_es  = CatBoostClassifier(**es_params)
    model_es.fit(
        X_tr, y_tr,
        cat_features=cat_indices,
        eval_set=Pool(X_es, y_es, cat_features=cat_indices),
    )
    n_trees = model_es.tree_count_

    params = {**cb_params, 'random_seed': seed, 'iterations': n_trees}
    params.pop('od_type', None)
    params.pop('od_wait', None)
    model = CatBoostClassifier(**params)
    model.fit(Pool(X_train, label=y_train, cat_features=cat_indices))
    return model


# ── Public API ────────────────────────────────────────────────────────────────

def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_all: np.ndarray,
    X_test: np.ndarray,
    cat_indices: list,
    cb_params: dict  = CB_PARAMS,
    n_splits: int    = N_FOLDS,
    seed: int        = SEED,
):
    """
    Train one CatBoostClassifier per fold and collect out-of-fold (OOF)
    predictions plus per-fold predictions on all val rows and test rows.

    Parameters
    ----------
    X_train, y_train    : labeled training data (already filtered — no NaN labels)
    X_val_all           : ALL val rows (4 489); used so calibration has full coverage
    X_test              : test rows (8 817)
    cat_indices         : integer positions of categorical features in the feature array
    cb_params           : CatBoost constructor kwargs (see config.CB_PARAMS)
    n_splits            : number of CV folds (default 10)
    seed                : random seed for fold splitting

    Returns
    -------
    oof_preds        ndarray (n_train,)      raw OOF probabilities on training data
    val_preds_folds  ndarray (n_val_all, k)  per-fold raw probs on all val rows
    test_preds_folds ndarray (n_test, k)     per-fold raw probs on test rows
    models           list[CatBoostClassifier] one fitted model per fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds        = np.zeros(len(X_train))
    val_preds_folds  = np.zeros((len(X_val_all), n_splits))
    test_preds_folds = np.zeros((len(X_test),    n_splits))
    fold_aucs        = []
    models           = []

    for fold_idx, (tr_idx, oof_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr,  X_oof = X_train[tr_idx],  X_train[oof_idx]
        y_tr,  y_oof = y_train[tr_idx],  y_train[oof_idx]

        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_tr, y_tr,
            cat_features = cat_indices,
            eval_set     = Pool(X_oof, y_oof, cat_features=cat_indices),
        )

        oof_preds[oof_idx]             = model.predict_proba(X_oof)[:, 1]
        val_preds_folds[:, fold_idx]   = model.predict_proba(X_val_all)[:, 1]
        test_preds_folds[:, fold_idx]  = model.predict_proba(X_test)[:, 1]
        models.append(model)

        fold_auc = roc_auc_score(y_oof, oof_preds[oof_idx])
        fold_aucs.append(fold_auc)
        print(f'  Fold {fold_idx + 1:2d}/{n_splits}  AUC={fold_auc:.4f}  trees={model.tree_count_}')

    overall_auc = roc_auc_score(y_train, oof_preds)
    print(f'\n  OOF AUC  : {overall_auc:.4f}')
    print(f'  Fold mean: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}')

    return oof_preds, val_preds_folds, test_preds_folds, models


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cat_indices: list,
    cb_params: dict     = CB_PARAMS,
    seed: int           = SEED,
    ensemble: bool      = True,
    n_folds: int        = 10,
    val_fraction: float = 0.1,
):
    """
    Train a final model (or ensemble) on all labeled training data.

    Parameters
    ----------
    ensemble     : True (default) — StratifiedKFold ensemble; each of the K members
                   is trained on a different (K-1)/K subset with the held-out fold
                   as the early-stopping eval set.  Returns a CatBoostEnsemble.
                   False — original single-model path; returns a CatBoostClassifier.
    n_folds      : K for the stratified K-fold split (default 10, ignored when
                   ensemble=False).
    val_fraction : holdout fraction used for early stopping in the single-model path
                   (ignored when ensemble=True).

    Why K-fold over same-data / different-seed ensemble?
    Each fold member trains on a genuinely different subset of the data, giving
    structural diversity rather than just random-state diversity.  The fold holdout
    also serves as a natural early-stopping signal, so no data is permanently wasted.

    Returns
    -------
    CatBoostEnsemble   if ensemble=True
    CatBoostClassifier if ensemble=False
    """
    if not ensemble:
        # ── Original single-model path (kept for easy revert) ────────────────
        model = _train_single(X_train, y_train, cat_indices, cb_params,
                              seed, val_fraction)
        print(f'Final model : {model.tree_count_} trees '
              f'(early-stopped on {val_fraction:.0%} holdout, seed={seed})')
        return model

    # ── K-fold ensemble path ──────────────────────────────────────────────────
    skf         = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    members     = []
    tree_counts = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # Each fold uses a different random seed so tree-splitting randomness
        # also varies across members
        params = {**cb_params, 'random_seed': seed + fold_idx}
        model  = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            cat_features = cat_indices,
            eval_set     = Pool(X_val, y_val, cat_features=cat_indices),
        )
        members.append(model)
        tree_counts.append(model.tree_count_)
        print(f'  Fold {fold_idx + 1:2d}/{n_folds}  '
              f'train={len(tr_idx):,}  val={len(val_idx):,}  '
              f'trees={model.tree_count_}')

    ensemble_model = CatBoostEnsemble(members)
    print(f'Ensemble: {n_folds} members, '
          f'{min(tree_counts)}–{max(tree_counts)} trees '
          f'(mean {ensemble_model.tree_count_})')
    return ensemble_model
