#!/usr/bin/env python3
"""wf_timing_npv — does a per-loan TIMING-AWARE funding decision beat the flat
break-even PD 0.226 on REALIZED amortizing profit, OUT-OF-FOLD?

Decision A (baseline): fund iff pd < 0.226  (closed-form break-even).
Decision B (timing):   fund iff E_NPV > 0, where E_NPV uses a per-loan model of
                       E[days_to_default | default, x] and E[recovery_frac | default, x].

We evaluate on the 51,722 observed rows with:
  (1) 5-fold random OOF
  (2) walk-forward TIME folds (ordered by application_timestamp)
For each fold we report REALIZED amortizing profit for A, B and delta = B - A,
using ACTUAL days_to_default & final_recovered_amount (capped per SHARED formula).

ADOPT only if delta>0 on (nearly) every fold and mean |delta| clearly exceeds the
per-fold std. Else REJECT as noise.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(r"C:\Users\ayush\intuit-hackathon\intuit-techweek-smb-underwriting")
sys.path.insert(0, str(REPO / "src"))
from build_a import build_cat_dtypes, build_features  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# ---- Economics (from SHARED) --------------------------------------------- #
TERM_INT = 0.35 * 60 / 365      # 0.057534
FEE = 0.03
NET_MARGIN = TERM_INT + FEE     # 0.087534
BREAK_EVEN = 0.226
SEED = 17

D = REPO / "dataset"
train = pd.read_csv(D / "train.csv")
val = pd.read_csv(D / "validation.csv")
test = pd.read_csv(D / "test.csv")
cats = build_cat_dtypes(train, val, test)
X_all = build_features(train, cats)

obs = train["default_flag"].notna().to_numpy()
Xo = X_all.loc[obs].reset_index(drop=True)
sub = train.loc[obs].reset_index(drop=True)
y = sub["default_flag"].astype(int).to_numpy()
amount = sub["requested_amount"].to_numpy(dtype=float)
dtd = sub["days_to_default"].to_numpy(dtype=float)            # NaN for non-default
rec = np.nan_to_num(sub["final_recovered_amount"].to_numpy(dtype=float))
ts = pd.to_datetime(sub["application_timestamp"]).to_numpy()
N = len(sub)
print(f"[data] observed rows={N:,}  default rate={y.mean():.4f}  defaulters={y.sum():,}")


def make_clf(seed):
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=seed, learning_rate=0.05,
        max_iter=400, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1)


def make_reg(seed):
    return HistGradientBoostingRegressor(
        categorical_features="from_dtype", random_state=seed, learning_rate=0.05,
        max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1)


def realized_profit(idx, fund):
    """REALIZED amortizing $ profit summed over funded rows in idx.
    fund is a boolean array aligned to idx."""
    a = amount[idx]; d = y[idx]; t = dtd[idx]; r = rec[idx]
    frac = np.clip(np.minimum(np.nan_to_num(t), 60) / 60.0, 0, 1)
    default_profit = np.minimum(a * (FEE + frac * (1 + TERM_INT) - 1) + r,
                                a * NET_MARGIN)
    profit = np.where(d == 0, a * NET_MARGIN, default_profit)
    return float(profit[fund].sum())


def expected_npv_decision(pd_hat, e_tstar, e_recfrac, a):
    """Decision B: fund iff E_NPV>0 (per-loan amortizing expected profit)."""
    frac = np.clip(np.minimum(e_tstar, 60) / 60.0, 0, 1)
    e_default_profit = np.minimum(a * (FEE + frac * (1 + TERM_INT) - 1) + e_recfrac * a,
                                  a * NET_MARGIN)
    e_npv = (1 - pd_hat) * a * NET_MARGIN + pd_hat * e_default_profit
    return e_npv > 0, e_npv


def run_split(name, splits):
    print("\n" + "=" * 78)
    print(f"{name}")
    print("=" * 78)
    rows = []
    for fold, (tr, te) in enumerate(splits):
        # PD model on train folds
        clf = make_clf(SEED + fold)
        clf.fit(Xo.iloc[tr], y[tr])
        pd_hat = clf.predict_proba(Xo.iloc[te])[:, 1]
        auc = roc_auc_score(y[te], pd_hat)

        # Timing model: E[days_to_default | default] on TRAIN-fold defaulters
        dmask = (y[tr] == 1)
        tr_def = tr[dmask]
        reg_t = make_reg(SEED + 100 + fold)
        reg_t.fit(Xo.iloc[tr_def], dtd[tr_def])
        e_tstar = reg_t.predict(Xo.iloc[te])

        # Recovery-fraction model: E[rec/amount | default] on TRAIN-fold defaulters
        recfrac_tr = rec[tr_def] / amount[tr_def]
        reg_r = make_reg(SEED + 200 + fold)
        reg_r.fit(Xo.iloc[tr_def], recfrac_tr)
        e_recfrac = np.clip(reg_r.predict(Xo.iloc[te]), 0, None)

        a_te = amount[te]
        fund_A = pd_hat < BREAK_EVEN
        fund_B, e_npv = expected_npv_decision(pd_hat, e_tstar, e_recfrac, a_te)

        pA = realized_profit(te, fund_A)
        pB = realized_profit(te, fund_B)
        delta = pB - pA

        # how often do A and B disagree?
        disagree = int((fund_A != fund_B).sum())
        # on disagreement rows, realized profit of the rows B funds but A doesn't,
        # minus rows A funds but B doesn't
        only_B = fund_B & ~fund_A
        only_A = fund_A & ~fund_B
        gain_onlyB = realized_profit(te, only_B)
        loss_onlyA = realized_profit(te, only_A)

        rows.append(dict(fold=fold, n=len(te), auc=auc,
                         nA=int(fund_A.sum()), nB=int(fund_B.sum()),
                         disagree=disagree, onlyB=int(only_B.sum()),
                         onlyA=int(only_A.sum()),
                         pA=pA, pB=pB, delta=delta,
                         gain_onlyB=gain_onlyB, loss_onlyA=loss_onlyA))
        print(f"fold {fold}: n={len(te):,} auc={auc:.4f} "
              f"| fundA={int(fund_A.sum()):,} fundB={int(fund_B.sum()):,} "
              f"disagree={disagree:,} (onlyB={int(only_B.sum())}, onlyA={int(only_A.sum())})")
        print(f"        realized A=${pA:,.0f}  B=${pB:,.0f}  "
              f"delta(B-A)=${delta:,.0f}  "
              f"[onlyB realized=${gain_onlyB:,.0f}, onlyA realized=${loss_onlyA:,.0f}]")

    df = pd.DataFrame(rows)
    deltas = df["delta"].to_numpy()
    mean_d = deltas.mean(); std_d = deltas.std(ddof=1)
    n_pos = int((deltas > 0).sum())
    print("-" * 78)
    print(f"SUMMARY {name}")
    print(f"  per-fold delta(B-A): {[f'${d:,.0f}' for d in deltas]}")
    print(f"  mean delta = ${mean_d:,.0f}   std = ${std_d:,.0f}   "
          f"folds with delta>0: {n_pos}/{len(df)}")
    if std_d > 0:
        print(f"  mean/std ratio = {mean_d/std_d:+.2f}  "
              f"(t-stat ~ {mean_d/(std_d/np.sqrt(len(df))):+.2f})")
    # total realized across folds (full OOF book)
    print(f"  total realized  A=${df['pA'].sum():,.0f}  B=${df['pB'].sum():,.0f}  "
          f"total delta=${df['delta'].sum():,.0f}")
    return df, mean_d, std_d, n_pos


# ---- (1) 5-fold random OOF ---------------------------------------------- #
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
splits_rand = list(kf.split(Xo))
df_r, md_r, sd_r, np_r = run_split("5-FOLD RANDOM OOF", splits_rand)

# ---- (2) Walk-forward TIME folds ---------------------------------------- #
order = np.argsort(ts, kind="stable")
n_tf = 5
# expanding-window walk-forward: train on all-earlier, test on next block
blocks = np.array_split(order, n_tf + 1)
splits_time = []
for k in range(n_tf):
    tr = np.concatenate(blocks[:k + 1])
    te = blocks[k + 1]
    splits_time.append((tr, te))
df_t, md_t, sd_t, np_t = run_split("WALK-FORWARD TIME FOLDS (expanding window)", splits_time)

# ---- Verdict ------------------------------------------------------------- #
print("\n" + "#" * 78)
print("VERDICT")
print("#" * 78)


def verdict(name, md, sd, npos, nfold):
    adopt = (npos >= nfold - 0) and (md > sd) and (md > 0)
    near = (npos >= nfold - 1) and (md > 0.5 * sd) and (md > 0)
    tag = "ADOPT" if adopt else ("WEAK/CAVEAT" if near else "REJECT")
    print(f"  {name}: mean=${md:,.0f} std=${sd:,.0f} pos={npos}/{nfold} -> {tag}")
    return tag


t_rand = verdict("random-OOF", md_r, sd_r, np_r, len(df_r))
t_time = verdict("walk-forward", md_t, sd_t, np_t, len(df_t))
print(f"\nOVERALL: random={t_rand}, walkforward={t_time}")
