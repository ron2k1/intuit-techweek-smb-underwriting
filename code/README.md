# code/ — Deliverable A Module Reference

All modelling logic for **Deliverable A** (calibrated default probability + approval decisions)
lives here as importable Python modules.  
The notebook `deliverable_A.ipynb` (repo root) is a thin orchestration layer that calls these.

---

## Quick start

Run the full pipeline from the repo root without opening the notebook:

```bash
python code/pipeline.py
# writes submissions/submission_A_decisions.csv
# then runs validate_submission.py automatically
```

---

## Data flow

```
load_raw()
    │
    ▼
prepare_splits()
    ├── train          (51 722 labeled rows, NaN outcomes dropped)
    ├── val_all        (4 489 rows — ALL val, used for submission)
    ├── val_labeled    (2 551 rows — known outcomes, calibration & optimisation)
    └── val_labeled_positions  (integer row indices mapping labeled → all)
    │
    ▼
engineer_features()  ×4 (train, val_all, val_labeled, test)
    │
    ▼
build_feature_cols()  →  feature_cols (47), cat_indices (6)
build_matrices()      →  X_train, y_train, X_val_all, X_val_labeled, y_val_labeled, X_test
    │
    ▼
run_cv()  (10-fold StratifiedKFold + CatBoost)
    ├── oof_preds         (51 722,)     — OOF probs on train
    ├── val_preds_folds   (4 489, 10)   — per-fold probs on all val
    └── test_preds_folds  (8 817, 10)   — per-fold probs on test
    │
    ▼
fit_calibrator()     — isotonic regression on val_labeled
apply_calibration()
    ├── val_cal_all      (4 489,)  → submission
    ├── val_cal_labeled  (2 551,)  → conformal
    └── test_cal         (8 817,)  → submission
    │
    ▼
conformal_intervals()  →  q_90 (half-width for 90 % coverage)
build_intervals()      →  lower, upper  (13 306,)
    │
    ▼
compute_breakeven_pd()  →  breakeven_pd  (≈ 8.7 %)
make_decisions()        →  decisions  (13 306,)
    │
    ▼
build_submission_a()  →  submissions/submission_A_decisions.csv
```

---

## Module reference

| File | Responsibility | Key exports |
|------|---------------|-------------|
| [config.py](config.py) | All constants, paths, hyper-parameters. **Edit here first when experimenting.** | `DATA_DIR`, `CB_PARAMS`, `REVENUE_RATE`, `ALPHA` |
| [data.py](data.py) | CSV loading, train/val splitting, feature engineering, numpy matrix builder | `load_raw`, `prepare_splits`, `engineer_features`, `build_feature_cols`, `build_matrices` |
| [integrity.py](integrity.py) | Data quality checks (planted violations, cross-split leakage) | `check_integrity`, `check_split_leakage` |
| [model.py](model.py) | 10-fold stratified CatBoost CV | `run_cv` |
| [calibration.py](calibration.py) | Isotonic calibration + split conformal prediction intervals | `fit_calibrator`, `apply_calibration`, `conformal_intervals`, `build_intervals` |
| [decision.py](decision.py) | Break-even PD from loan economics, binary approval decisions | `compute_breakeven_pd`, `make_decisions` |
| [submission.py](submission.py) | Assemble and write `submission_A_decisions.csv` | `build_submission_a` |
| [pipeline.py](pipeline.py) | End-to-end orchestrator (no notebook needed) | `main()` |

---

## Key design decisions

### Two val copies
`val_all` (4 489 rows) feeds the submission file.  
`val_labeled` (2 551 rows, NaN `default_flag` dropped) feeds calibration and conformal
intervals. This prevents using unlabeled rows as if outcomes were known.

### Train filtered upfront
`default_flag` NaN rows (declined / immature) are dropped in `prepare_splits` so no
downstream code accidentally trains on rows without a label.

### Calibration on val_labeled
Isotonic regression is fit on `val_labeled`, which is completely independent from the
training fold — no data leakage.

### Conformal intervals (split conformal)
Half-width `q_90 = quantile(|y - p_hat|, corrected-0.90)` guarantees ≥ 90 % marginal
coverage under exchangeability. See `calibration.conformal_intervals` for the
finite-sample correction formula.

### Break-even PD (not 0.5)
`approve when PD < revenue_rate / (LGD + revenue_rate)`.  
LGD is estimated from observed recovery rates in training defaults (~8.7 % threshold).

---

## Changing hyper-parameters

Everything is in `config.py`. The most impactful knobs:

```python
# Tune CatBoost
CB_PARAMS['iterations']     = 2000   # more trees
CB_PARAMS['learning_rate']  = 0.03   # slower learning
CB_PARAMS['depth']          = 8      # wider trees

# Change the approval threshold
# (override after calling compute_breakeven_pd, or set a fixed value)
breakeven_pd = 0.10   # 10 % fixed threshold

# Change conformal coverage
ALPHA = 0.05   # 95 % intervals instead of 90 %
```
