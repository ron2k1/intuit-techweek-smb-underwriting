"""
model.py — 10-fold stratified CatBoost cross-validation.

Public API
----------
run_cv(X_train, y_train, X_val_all, X_test, cat_indices, ...)
    -> (oof_preds, val_preds_folds, test_preds_folds, models)
"""

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import CB_PARAMS, N_FOLDS, SEED


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
