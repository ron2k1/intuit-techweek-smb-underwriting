#!/usr/bin/env python3
"""wf_timing_npv_verify — INDEPENDENT adversarial re-run of the timing-aware
E[NPV] decision lever vs the flat break-even 0.226, on REALIZED amortizing
profit, OUT-OF-FOLD.

Decision A (baseline): fund iff pd_hat < 0.226 (closed-form break-even).
Decision B (timing):   fund iff E_NPV > 0, where E_NPV uses per-loan models of
                       E[days_to_default | default, x] and E[rec_frac | default, x].

This is a *fresh* script with DIFFERENT random_state seeds {3, 11, 29}, averaged.
For each seed and each fold we report REALIZED amortizing profit for A, B and
delta = B - A using ACTUAL days_to_default & final_recovered_amount (capped per
the SHARED formula).

Default verdict = REJECT unless delta>0 on essentially every fold AND the gain
clearly exceeds fold-to-fold noise across seeds.
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
SEEDS = [3, 11, 29]             # fresh, independent of the original SEED=17

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
print(f"[econ] NET_MARGIN={NET_MARGIN:.6f}  BREAK_EVEN={BREAK_EVEN}  seeds={SEEDS}")


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
    """REALIZED amortizing $ profit summed over funded rows in idx."""
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


def run_split(name, splits, seed):
    rows = []
    for fold, (tr, te) in enumerate(splits):
        clf = make_clf(seed + fold)
        clf.fit(Xo.iloc[tr], y[tr])
        pd_hat = clf.predict_proba(Xo.iloc[te])[:, 1]
        auc = roc_auc_score(y[te], pd_hat)

        dmask = (y[tr] == 1)
        tr_def = tr[dmask]
        reg_t = make_reg(seed + 100 + fold)
        reg_t.fit(Xo.iloc[tr_def], dtd[tr_def])
        e_tstar = reg_t.predict(Xo.iloc[te])

        recfrac_tr = rec[tr_def] / amount[tr_def]
        reg_r = make_reg(seed + 200 + fold)
        reg_r.fit(Xo.iloc[tr_def], recfrac_tr)
        e_recfrac = np.clip(reg_r.predict(Xo.iloc[te]), 0, None)

        a_te = amount[te]
        fund_A = pd_hat < BREAK_EVEN
        fund_B, e_npv = expected_npv_decision(pd_hat, e_tstar, e_recfrac, a_te)

        pA = realized_profit(te, fund_A)
        pB = realized_profit(te, fund_B)
        delta = pB - pA
        disagree = int((fund_A != fund_B).sum())
        only_B = fund_B & ~fund_A
        only_A = fund_A & ~fund_B

        rows.append(dict(fold=fold, n=len(te), auc=auc,
                         nA=int(fund_A.sum()), nB=int(fund_B.sum()),
                         disagree=disagree, onlyB=int(only_B.sum()),
                         onlyA=int(only_A.sum()),
                         pA=pA, pB=pB, delta=delta))
    df = pd.DataFrame(rows)
    deltas = df["delta"].to_numpy()
    print(f"  [{name} seed={seed}] per-fold delta(B-A): "
          f"{[f'${d:,.0f}' for d in deltas]}")
    print(f"      mean=${deltas.mean():,.0f} std=${deltas.std(ddof=1):,.0f} "
          f"pos={int((deltas>0).sum())}/{len(df)} "
          f"auc~[{df['auc'].min():.4f},{df['auc'].max():.4f}] "
          f"total delta=${df['delta'].sum():,.0f}")
    return df


def make_random_splits(seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    return list(kf.split(Xo))


def make_time_splits():
    order = np.argsort(ts, kind="stable")
    n_tf = 5
    blocks = np.array_split(order, n_tf + 1)
    splits = []
    for k in range(n_tf):
        tr = np.concatenate(blocks[:k + 1])
        te = blocks[k + 1]
        splits.append((tr, te))
    return splits


# ====================================================================== #
print("\n" + "=" * 78)
print("(1) 5-FOLD RANDOM OOF  — averaged over seeds", SEEDS)
print("=" * 78)
rand_dfs = {}
for s in SEEDS:
    rand_dfs[s] = run_split("RAND", make_random_splits(s), s)

print("\n" + "=" * 78)
print("(2) WALK-FORWARD TIME FOLDS (expanding window) — averaged over seeds", SEEDS)
print("=" * 78)
time_splits = make_time_splits()  # same time ordering; seed only varies the models
time_dfs = {}
for s in SEEDS:
    time_dfs[s] = run_split("TIME", time_splits, s)


def aggregate(name, dfs):
    # pool all per-fold deltas across seeds
    all_deltas = np.concatenate([d["delta"].to_numpy() for d in dfs.values()])
    seed_means = np.array([d["delta"].mean() for d in dfs.values()])
    seed_totals = np.array([d["delta"].sum() for d in dfs.values()])
    n_pos = int((all_deltas > 0).sum())
    n_tot = len(all_deltas)
    mean_d = all_deltas.mean()
    std_d = all_deltas.std(ddof=1)
    # per-fold mean across seeds (position-aligned for random; folds are not
    # comparable across seeds so we report pooled + per-seed-mean dispersion)
    print("\n" + "-" * 78)
    print(f"AGGREGATE — {name}")
    print(f"  pooled per-fold deltas (n={n_tot}): mean=${mean_d:,.0f} "
          f"std=${std_d:,.0f}  pos={n_pos}/{n_tot}")
    if std_d > 0:
        t = mean_d / (std_d / np.sqrt(n_tot))
        print(f"  pooled mean/std={mean_d/std_d:+.2f}  t~{t:+.2f}")
    print(f"  per-seed mean delta/fold: {[f'${m:,.0f}' for m in seed_means]} "
          f"(seed-avg=${seed_means.mean():,.0f})")
    print(f"  per-seed total delta:     {[f'${m:,.0f}' for m in seed_totals]} "
          f"(seed-avg=${seed_totals.mean():,.0f})")
    return mean_d, std_d, n_pos, n_tot


mr, sr, pr, nr = aggregate("RANDOM 5-FOLD OOF", rand_dfs)
mt, st, pt, nt = aggregate("WALK-FORWARD TIME", time_dfs)

print("\n" + "#" * 78)
print("VERDICT (default REJECT unless delta>0 on ~all folds AND > noise)")
print("#" * 78)


def verdict(name, m, s, npos, ntot):
    # ADOPT requires: positive on essentially every fold AND mean clearly > std.
    adopt = (npos >= ntot - 1) and (m > s) and (m > 0)
    tag = "ADOPT" if adopt else "REJECT"
    print(f"  {name}: mean=${m:,.0f} std=${s:,.0f} pos={npos}/{ntot} -> {tag}")
    return tag


v_r = verdict("RANDOM", mr, sr, pr, nr)
v_t = verdict("WALK-FORWARD", mt, st, pt, nt)
print(f"\nOVERALL: random={v_r}, walkforward={v_t}")
