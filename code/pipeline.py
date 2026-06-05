"""
pipeline.py — end-to-end runner for Deliverable A.

Executes every step in sequence and writes submission_A_decisions.csv.
Equivalent to running all cells in deliverable_A.ipynb.

Usage (from repo root):
    python code/pipeline.py
"""

import sys
import subprocess
import warnings
from pathlib import Path

# Make sibling modules importable when running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import config
from data        import load_raw, prepare_splits, engineer_features, build_feature_cols, build_matrices
from integrity   import check_integrity, check_split_leakage
from model       import run_cv
from calibration import fit_calibrator, apply_calibration, conformal_intervals, build_intervals
from decision    import compute_breakeven_pd, make_decisions
from submission  import build_submission_a


def _header(step: int, total: int, title: str) -> None:
    print(f'\n[{step}/{total}] {title}')
    print('  ' + '-' * 50)


def main() -> None:
    TOTAL = 8
    print('=' * 60)
    print('  Deliverable A — CatBoost 10-Fold Default Probability')
    print('=' * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    _header(1, TOTAL, 'Loading data')
    train_raw, val_raw, test = load_raw()
    train, val_all, val_labeled, val_labeled_positions = prepare_splits(train_raw, val_raw)
    print(f'  train (labeled) : {len(train):>6,}  default rate {train["default_flag"].mean():.4f}')
    print(f'  val_all         : {len(val_all):>6,}  (→ submission)')
    print(f'  val_labeled     : {len(val_labeled):>6,}  (→ calibration)')
    print(f'  test            : {len(test):>6,}')

    # ── 2. Integrity ─────────────────────────────────────────────────────────
    _header(2, TOTAL, 'Integrity checks')
    check_integrity(train, 'train (labeled)')
    check_split_leakage(train, val_all, test)

    # ── 3. Feature engineering ───────────────────────────────────────────────
    _header(3, TOTAL, 'Feature engineering')
    train_fe       = engineer_features(train)
    val_all_fe     = engineer_features(val_all)
    val_labeled_fe = engineer_features(val_labeled)
    test_fe        = engineer_features(test)
    feature_cols, cat_indices = build_feature_cols(train_fe)
    print(f'  {len(feature_cols)} features total, {len(cat_indices)} categorical')

    arrs = build_matrices(train_fe, val_all_fe, val_labeled_fe, test_fe, feature_cols)

    # ── 4. 10-fold CV ────────────────────────────────────────────────────────
    _header(4, TOTAL, f'{config.N_FOLDS}-fold CatBoost CV')
    oof_preds, val_preds_folds, test_preds_folds, models = run_cv(
        arrs['X_train'], arrs['y_train'],
        arrs['X_val_all'], arrs['X_test'],
        cat_indices,
    )

    # ── 5. Calibration ───────────────────────────────────────────────────────
    _header(5, TOTAL, 'Isotonic calibration on val_labeled')
    val_raw_all = val_preds_folds.mean(axis=1)
    test_raw    = test_preds_folds.mean(axis=1)

    iso = fit_calibrator(val_raw_all, arrs['y_val_labeled'], val_labeled_positions)
    val_cal_all, val_cal_labeled, test_cal = apply_calibration(
        iso, val_raw_all, test_raw, val_labeled_positions
    )
    y_vl = arrs['y_val_labeled']
    print(f'  AUC     : {roc_auc_score(y_vl, val_cal_labeled):.4f}')
    print(f'  Logloss : {log_loss(y_vl, val_cal_labeled):.4f}')
    print(f'  Brier   : {brier_score_loss(y_vl, val_cal_labeled):.4f}')
    print(f'  Mean PD : {val_cal_labeled.mean():.4f}  (actual {y_vl.mean():.4f})')

    # ── 6. Conformal intervals ───────────────────────────────────────────────
    _header(6, TOTAL, 'Conformal prediction intervals (90 %)')
    q_90, coverage = conformal_intervals(y_vl, val_cal_labeled)
    print(f'  q_90 (half-width) : {q_90:.4f}')
    print(f'  Empirical coverage: {coverage:.4f}  (target ≥ {1-config.ALPHA:.2f})')
    all_pds = np.concatenate([val_cal_all, test_cal])
    lower, upper = build_intervals(all_pds, q_90)

    # ── 7. Decisions ─────────────────────────────────────────────────────────
    _header(7, TOTAL, 'Break-even threshold and approval decisions')
    breakeven_pd, lgd, avg_recovery = compute_breakeven_pd(train_raw)
    print(f'  Avg recovery  : {avg_recovery:.4f}')
    print(f'  LGD           : {lgd:.4f}')
    print(f'  Break-even PD : {breakeven_pd:.4f}  ({breakeven_pd * 100:.2f} %)')
    decisions = make_decisions(all_pds, breakeven_pd)
    print(f'  Approval rate : {decisions.mean():.4f}  ({decisions.sum():,} / {len(decisions):,})')

    # ── 8. Submission ────────────────────────────────────────────────────────
    _header(8, TOTAL, 'Saving submission_A_decisions.csv')
    submission, out_path = build_submission_a(
        val_all, test, val_cal_all, test_cal, decisions, lower, upper
    )
    print(f'  {len(submission):,} rows → {out_path}')

    # Validate
    print('\n--- Format validation ---')
    result = subprocess.run(
        [sys.executable, 'validate_submission.py', str(config.SUB_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr)


if __name__ == '__main__':
    main()
