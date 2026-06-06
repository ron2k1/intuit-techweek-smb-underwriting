#!/usr/bin/env python3
"""wf4_model_techniques — 5-fold OOF test of TIER-1 #4/#9 + TIER-2 #10 model levers.

Baseline to beat (deployed Deliverable A blend):
  calibrated 0.4*HGB-ensemble + 0.6*L2-logistic on build_features (keeps prior_uw_score).
  Honest 5-fold OOF: AUC 0.7752, Brier 0.1168, LogLoss 0.3818, ECE 0.0018 (cross-fit isotonic).

This script reproduces an HONEST OOF harness and pits the following against it:
  (a) MONOTONIC constraints on the HGB arm (signs from wf3_causal_c.MONO_SIGN)
  (b) CLASS WEIGHTING / scale_pos_weight on the HGB arm (class_weight='balanced')
  (c) TARGET ENCODING / WoE of integer categoricals (cross-fit) vs native categorical
  (d) STACKING meta-learner: logistic on nested-OOF base preds vs fixed 0.4/0.6 blend
  (e) FOCAL-style loss approximation via sample weights on the HGB arm

For every variant we report PER-FOLD: AUC, Brier, ECE, LogLoss.
Calibration is applied HONESTLY: a cross-fit isotonic mapping where, inside each
outer fold, the isotonic is fit on the held-out OOF preds of the OTHER folds (i.e.
isotonic is fit on data disjoint from the rows it calibrates). This mirrors the
baseline's "cross-fit isotonic" ECE protocol so calibration metrics are honest.

Run:
  set PYTHONPATH=...; .venv/Scripts/python.exe -m src.wf4_model_techniques
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_a import (  # noqa: E402
    build_cat_dtypes, build_features, numeric_frame, CATEGORICAL,
)
from src.wf3_causal_c import MONO_SIGN  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SEED = 17
N_SPLITS = 5
W_HGB = 0.4  # deployed blend weight


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def make_hgb(seed: int, mono: list[int] | None = None,
             class_weight=None) -> HistGradientBoostingClassifier:
    kw = dict(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        categorical_features="from_dtype",
        random_state=seed,
    )
    if mono is not None:
        kw["monotonic_cst"] = mono
    if class_weight is not None:
        kw["class_weight"] = class_weight
    return HistGradientBoostingClassifier(**kw)


def make_logit():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=3000),
    )


def mono_vector(cols: list[str]) -> list[int]:
    return [MONO_SIGN.get(c, 0) for c in cols]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ece(p, y, n_bins=10):
    """Expected calibration error (equal-width bins on [0,1])."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return e


def metrics_block(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(
        auc=roc_auc_score(y, p),
        brier=brier_score_loss(y, p),
        logloss=log_loss(y, p),
        ece=ece(p, y),
    )


# --------------------------------------------------------------------------- #
# Honest cross-fit isotonic calibration of an OOF score vector.
# Inside each outer fold, fit isotonic on the OOF preds of the OTHER folds and
# apply to this fold's preds -> isotonic never sees the rows it calibrates.
# --------------------------------------------------------------------------- #
def crossfit_isotonic(oof_raw, y, fold_assign):
    cal = np.zeros_like(oof_raw, dtype=float)
    folds = np.unique(fold_assign)
    for f in folds:
        te = fold_assign == f
        tr = ~te
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof_raw[tr], y[tr])
        cal[te] = np.clip(iso.predict(oof_raw[te]), 1e-6, 1 - 1e-6)
    return cal


# --------------------------------------------------------------------------- #
# Target / WoE encoding of integer categoricals (cross-fit within a training
# fold via inner KFold so encodings are leakage-free), with smoothing.
# --------------------------------------------------------------------------- #
def woe_target_encode(Xtr_raw, ytr, Xte_raw, cat_cols, mode="woe",
                      smoothing=50.0, seed=SEED):
    """Return numeric frames (Xtr_enc, Xte_enc) where each cat col is replaced by
    its target/WoE encoding. Train encoding is built with inner 5-fold OOF to
    avoid leakage; test encoding uses the full-train mapping."""
    rng = np.random.default_rng(seed)
    prior = ytr.mean()
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    Xtr_enc = Xtr_raw.copy()
    Xte_enc = Xte_raw.copy()

    def encode_map(s_tr, y_tr):
        """category-value -> encoded number, with smoothing toward prior."""
        df = pd.DataFrame({"c": s_tr.astype("object"), "y": y_tr})
        grp = df.groupby("c")["y"]
        cnt = grp.count()
        pos = grp.sum()
        if mode == "target":
            enc = (pos + smoothing * prior) / (cnt + smoothing)
        else:  # woe
            # smoothed event rate -> log(odds_cat / odds_prior)
            rate = (pos + smoothing * prior) / (cnt + smoothing)
            rate = np.clip(rate, 1e-4, 1 - 1e-4)
            base = np.clip(prior, 1e-4, 1 - 1e-4)
            enc = np.log(rate / (1 - rate)) - np.log(base / (1 - base))
        return enc.to_dict()

    tr_idx = np.arange(len(ytr))
    for c in cat_cols:
        # inner OOF encoding for the training rows
        col_tr = Xtr_raw[c].astype("object").to_numpy()
        enc_tr = np.full(len(ytr), prior if mode == "target" else 0.0, dtype=float)
        for itr, ite in inner.split(tr_idx, ytr):
            m = encode_map(pd.Series(col_tr[itr]), ytr[itr])
            default = prior if mode == "target" else 0.0
            enc_tr[ite] = [m.get(v, default) for v in col_tr[ite]]
        Xtr_enc[c] = enc_tr
        # full-train mapping for the test rows
        m_full = encode_map(pd.Series(col_tr), ytr)
        default = prior if mode == "target" else 0.0
        Xte_enc[c] = [m_full.get(v, default) for v in Xte_raw[c].astype("object").to_numpy()]
    return Xtr_enc, Xte_enc


# --------------------------------------------------------------------------- #
# Focal-style sample weights: w_i = alpha * (1 - p_t)^gamma with p_t the model's
# own probability of the true class. We approximate with a 2-pass scheme: fit a
# quick HGB to get p, derive focal weights, refit. (Single refit; cheap proxy.)
# --------------------------------------------------------------------------- #
def focal_weights(p_pos, y, gamma=2.0):
    p_t = np.where(y == 1, p_pos, 1 - p_pos)
    return (1 - p_t) ** gamma


def main() -> int:
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    cats = build_cat_dtypes(train, val, test)
    X = build_features(train, cats)

    obs = train["default_flag"].notna().to_numpy()
    Xo = X.loc[obs].reset_index(drop=True)
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    n = len(y)
    print(f"[data] {n:,} observed rows, default rate {y.mean():.4f}, {Xo.shape[1]} feats")

    mono = mono_vector(list(Xo.columns))
    n_mono = sum(s != 0 for s in mono)
    print(f"[mono] {n_mono} monotone-constrained cols "
          f"(+:{sum(s>0 for s in mono)}, -:{sum(s<0 for s in mono)})")

    # categorical columns actually present in the frame
    cat_cols = [c for c in CATEGORICAL if c in Xo.columns]
    print(f"[cats] integer categoricals: {cat_cols}")

    # Numeric frame (for logit and for WoE-as-numeric variants).
    Xnum = numeric_frame(Xo)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_assign = np.full(n, -1)

    # OOF raw-score accumulators per arm/variant.
    oof = {
        "hgb": np.zeros(n),
        "hgb_mono": np.zeros(n),
        "hgb_cw": np.zeros(n),          # class_weight balanced
        "hgb_focal": np.zeros(n),       # focal-style sample weights (gamma=2, OOF pass-1)
        "hgb_focal1": np.zeros(n),      # focal-style sample weights (gamma=1, OOF pass-1)
        "logit": np.zeros(n),
        "logit_woe": np.zeros(n),       # logit on WoE-encoded cats
        "logit_tgt": np.zeros(n),       # logit on target-encoded cats
        "hgb_woe": np.zeros(n),         # HGB with WoE numeric cats instead of native
    }

    for k, (tr, te) in enumerate(skf.split(Xo, y)):
        fold_assign[te] = k
        ytr = y[tr]

        # ---- (baseline) native HGB --------------------------------------- #
        m = make_hgb(SEED + k)
        m.fit(Xo.iloc[tr], ytr)
        oof["hgb"][te] = m.predict_proba(Xo.iloc[te])[:, 1]

        # ---- (a) monotone HGB -------------------------------------------- #
        mm = make_hgb(SEED + k, mono=mono)
        mm.fit(Xo.iloc[tr], ytr)
        oof["hgb_mono"][te] = mm.predict_proba(Xo.iloc[te])[:, 1]

        # ---- (b) class_weight balanced HGB ------------------------------- #
        mc = make_hgb(SEED + k, class_weight="balanced")
        mc.fit(Xo.iloc[tr], ytr)
        oof["hgb_cw"][te] = mc.predict_proba(Xo.iloc[te])[:, 1]

        # ---- (e) focal-style weighted HGB (2-pass, HONEST inner-OOF pass1) -- #
        # pass 1: get OUT-OF-FOLD probabilities for the train rows via an inner
        # KFold, so the focal difficulty weight reflects genuine generalization
        # difficulty, NOT in-sample overfit (predicting on one's own train rows
        # makes p_t degenerate and inverts the ranking). This is the correct way
        # to approximate focal loss with sample weights.
        p_tr_oof = np.zeros(len(tr))
        inner_f = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + k)
        tr_pos = np.arange(len(tr))
        for itr, ite in inner_f.split(tr_pos, ytr):
            mpi = make_hgb(SEED + 100 + k)
            mpi.fit(Xo.iloc[tr].iloc[itr], ytr[itr])
            p_tr_oof[ite] = mpi.predict_proba(Xo.iloc[tr].iloc[ite])[:, 1]
        # gamma=2 focal
        w_focal = focal_weights(p_tr_oof, ytr, gamma=2.0)
        w_focal = w_focal * (len(w_focal) / w_focal.sum())
        mf = make_hgb(SEED + 200 + k)
        mf.fit(Xo.iloc[tr], ytr, sample_weight=w_focal)
        oof["hgb_focal"][te] = mf.predict_proba(Xo.iloc[te])[:, 1]
        # gamma=1 focal (milder)
        w_focal1 = focal_weights(p_tr_oof, ytr, gamma=1.0)
        w_focal1 = w_focal1 * (len(w_focal1) / w_focal1.sum())
        mf1 = make_hgb(SEED + 300 + k)
        mf1.fit(Xo.iloc[tr], ytr, sample_weight=w_focal1)
        oof["hgb_focal1"][te] = mf1.predict_proba(Xo.iloc[te])[:, 1]

        # ---- (baseline logit) -------------------------------------------- #
        lg = make_logit()
        lg.fit(Xnum.iloc[tr], ytr)
        oof["logit"][te] = lg.predict_proba(Xnum.iloc[te])[:, 1]

        # ---- (c) WoE / target encoding ----------------------------------- #
        # Build encoded numeric frames (start from native numeric frame, then
        # overwrite the categorical columns with their encodings).
        Xtr_woe, Xte_woe = woe_target_encode(
            Xnum.iloc[tr], ytr, Xnum.iloc[te], cat_cols, mode="woe", seed=SEED + k)
        lg_woe = make_logit()
        lg_woe.fit(Xtr_woe, ytr)
        oof["logit_woe"][te] = lg_woe.predict_proba(Xte_woe)[:, 1]

        Xtr_tgt, Xte_tgt = woe_target_encode(
            Xnum.iloc[tr], ytr, Xnum.iloc[te], cat_cols, mode="target", seed=SEED + k)
        lg_tgt = make_logit()
        lg_tgt.fit(Xtr_tgt, ytr)
        oof["logit_tgt"][te] = lg_tgt.predict_proba(Xte_tgt)[:, 1]

        # ---- (c') HGB on WoE-encoded cats (numeric, no native categorical) -- #
        # Replace the native-categorical HGB with one trained on WoE-numeric cols.
        mw = HistGradientBoostingClassifier(
            loss="log_loss", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.1, random_state=SEED + k,
        )
        mw.fit(Xtr_woe, ytr)
        oof["hgb_woe"][te] = mw.predict_proba(Xte_woe)[:, 1]

        print(f"[fold {k}] done (n_te={len(te):,})")

    # ===================================================================== #
    # Build CALIBRATED OOF predictions (cross-fit isotonic) and the blends.
    # ===================================================================== #
    print("\n" + "=" * 90)
    print("RAW-OOF AUC per arm (uncalibrated; AUC is calibration-invariant):")
    for name, p in oof.items():
        print(f"  {name:12s} AUC={roc_auc_score(y, p):.4f}")

    # ---- Blend definitions ---------------------------------------------- #
    # Baseline blend: 0.4*HGB + 0.6*logit (native).
    blends = {
        "BASELINE(0.4HGB+0.6logit)": W_HGB * oof["hgb"] + (1 - W_HGB) * oof["logit"],
        "MONO(0.4HGBmono+0.6logit)": W_HGB * oof["hgb_mono"] + (1 - W_HGB) * oof["logit"],
        "CW(0.4HGBcw+0.6logit)": W_HGB * oof["hgb_cw"] + (1 - W_HGB) * oof["logit"],
        "FOCALg2(0.4HGBfocal+0.6logit)": W_HGB * oof["hgb_focal"] + (1 - W_HGB) * oof["logit"],
        "FOCALg1(0.4HGBfocal+0.6logit)": W_HGB * oof["hgb_focal1"] + (1 - W_HGB) * oof["logit"],
        "WOE(0.4HGB+0.6logitWOE)": W_HGB * oof["hgb"] + (1 - W_HGB) * oof["logit_woe"],
        "TGT(0.4HGB+0.6logitTGT)": W_HGB * oof["hgb"] + (1 - W_HGB) * oof["logit_tgt"],
        "HGBWOE(0.4HGBwoe+0.6logit)": W_HGB * oof["hgb_woe"] + (1 - W_HGB) * oof["logit"],
    }

    # ---- (d) STACKING meta-learner: logistic on OOF base preds ---------- #
    # Honest: fit the meta-logit per outer fold on OOF preds of the OTHER folds,
    # apply to this fold. Base preds (oof['hgb'], oof['logit']) are already OOF, so
    # using them directly as meta-features is leakage-free for the outer fold split.
    meta_feats = np.column_stack([oof["hgb"], oof["logit"]])
    oof_stack = np.zeros(n)
    for f in range(N_SPLITS):
        te = fold_assign == f
        tr = ~te
        meta = LogisticRegression(C=1.0, max_iter=3000)
        meta.fit(meta_feats[tr], y[tr])
        oof_stack[te] = meta.predict_proba(meta_feats[te])[:, 1]
    blends["STACK(logit on [HGB,logit])"] = oof_stack

    # Also a 3-input stack adding the mono HGB.
    meta_feats3 = np.column_stack([oof["hgb"], oof["logit"], oof["hgb_mono"]])
    oof_stack3 = np.zeros(n)
    for f in range(N_SPLITS):
        te = fold_assign == f
        tr = ~te
        meta = LogisticRegression(C=1.0, max_iter=3000)
        meta.fit(meta_feats3[tr], y[tr])
        oof_stack3[te] = meta.predict_proba(meta_feats3[te])[:, 1]
    blends["STACK3(logit on [HGB,logit,HGBmono])"] = oof_stack3

    # ===================================================================== #
    # Calibrate each blend with cross-fit isotonic and compute per-fold metrics.
    # ===================================================================== #
    print("\n" + "=" * 90)
    print("CALIBRATED (cross-fit isotonic) PER-FOLD METRICS  [mean over 5 folds]")
    print("=" * 90)
    print(f"{'variant':42s} {'AUC':>8s} {'Brier':>8s} {'LogLoss':>9s} {'ECE':>8s}")

    results = {}
    per_fold = {}
    for name, raw in blends.items():
        cal = crossfit_isotonic(raw, y, fold_assign)
        # per-fold metrics on calibrated preds
        rows = []
        for f in range(N_SPLITS):
            te = fold_assign == f
            rows.append(metrics_block(cal[te], y[te]))
        fdf = pd.DataFrame(rows)
        per_fold[name] = fdf
        mean = fdf.mean()
        results[name] = mean
        print(f"{name:42s} {mean['auc']:8.4f} {mean['brier']:8.4f} "
              f"{mean['logloss']:9.4f} {mean['ece']:8.4f}")

    # ---- Per-fold detail + paired deltas vs baseline -------------------- #
    base = "BASELINE(0.4HGB+0.6logit)"
    bf = per_fold[base]
    print("\n" + "=" * 90)
    print(f"PER-FOLD detail (calibrated). Baseline = {base}")
    print("=" * 90)
    for name, fdf in per_fold.items():
        tag = "  <== BASELINE" if name == base else ""
        print(f"\n{name}{tag}")
        for f in range(N_SPLITS):
            r = fdf.iloc[f]
            print(f"  fold{f}: AUC={r['auc']:.4f} Brier={r['brier']:.4f} "
                  f"LogLoss={r['logloss']:.4f} ECE={r['ece']:.4f}")
        if name != base:
            d_auc = fdf["auc"] - bf["auc"]
            d_brier = fdf["brier"] - bf["brier"]
            d_ll = fdf["logloss"] - bf["logloss"]
            d_ece = fdf["ece"] - bf["ece"]
            print(f"  PAIRED dAUC:    " + ", ".join(f"{v:+.4f}" for v in d_auc)
                  + f"  (mean {d_auc.mean():+.5f})")
            print(f"  PAIRED dBrier:  " + ", ".join(f"{v:+.4f}" for v in d_brier)
                  + f"  (mean {d_brier.mean():+.5f}, neg=better)")
            print(f"  PAIRED dLogLoss:" + ", ".join(f"{v:+.4f}" for v in d_ll)
                  + f"  (mean {d_ll.mean():+.5f}, neg=better)")
            print(f"  PAIRED dECE:    " + ", ".join(f"{v:+.4f}" for v in d_ece)
                  + f"  (mean {d_ece.mean():+.5f}, neg=better)")
            print(f"  AUC all folds >= baseline? {bool((d_auc >= -1e-9).all())}; "
                  f"AUC all folds > baseline? {bool((d_auc > 0).all())}")

    # ---- Headline summary table ----------------------------------------- #
    print("\n" + "=" * 90)
    print("SUMMARY (mean over folds; deltas vs baseline). + = better for AUC; - = better for Brier/LL/ECE")
    print("=" * 90)
    b = results[base]
    print(f"{'variant':42s} {'dAUC':>9s} {'dBrier':>9s} {'dLogLoss':>9s} {'dECE':>9s}")
    for name, m in results.items():
        if name == base:
            print(f"{name:42s} {'--':>9s} {'--':>9s} {'--':>9s} {'--':>9s}  "
                  f"(AUC {m['auc']:.4f} Brier {m['brier']:.4f} LL {m['logloss']:.4f} ECE {m['ece']:.4f})")
            continue
        print(f"{name:42s} {m['auc']-b['auc']:+9.5f} {m['brier']-b['brier']:+9.5f} "
              f"{m['logloss']-b['logloss']:+9.5f} {m['ece']-b['ece']:+9.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
