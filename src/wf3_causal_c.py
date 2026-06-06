#!/usr/bin/env python3
"""Deliverable C PROTOTYPE -- backdoor-valid do(feature=value) counterfactual PD.

This is the ADVERSARIAL PROOF that the C approach is causal, not a naive
re-prediction. It does NOT write a submission; it fits the causal blend and
checks that the do() interventions move PD in the correct, sign-verified
direction on ~30 sample queries (one per distinct queried feature).

Causal design (defended against the README traps):

  (1) BACKDOOR / COLLIDER REMOVAL. prior_decision is a collider (selection node),
      prior_underwriter_score is the selection score, prior_approved_amount is a
      prior-decision output. Conditioning on any of them opens a backdoor path
      score -> [decision] <- X, biasing the interventional estimate. We DROP all
      three (and their __isna indicators) from the causal feature frame. They are
      fine for pure prediction (kept in build_a.py) but invalid for do().

  (2) SELF-REPORT ZERO-EFFECT. stated_annual_revenue and stated_time_in_business
      are optimistically inflated application statements, NOT manipulable business
      state. README trap #4: do(stated_*) has ~0 TRUE causal effect. We enforce
      this STRUCTURALLY by EXCLUDING the stated_* columns (and their __isna) from
      the causal model, so do(stated_*) returns EXACTLY the baseline PD (delta 0).
      This is stricter than a shrink factor and is the backdoor-valid answer.
      intended_use_of_funds: a stated categorical purpose, also self-report; the
      README treats stated_* as non-causal. It is a coarse declared label, not a
      business-state lever, so we EXCLUDE it too (delta 0). (It is a relatively
      rare query feature: 12/900.)

  (3) requested_amount IS a genuine lever. Loan size -> leverage -> default. We
      KEEP it and, on do(requested_amount), RECOMPUTE the engineered leverage.

  (4) ENGINEERED LEVERAGE RECOMPUTE. The dataset ships
      requested_amount_to_observed_revenue = requested_amount /
      (observed_monthly_revenue_avg_3mo * 12) (verified corr 1.0). It is a
      DETERMINISTIC descendant of two intervenable parents. Under do(parent=v) the
      g-formula requires recomputing the child. We replace the shipped ratio with a
      clean observed-only leverage column we control, and recompute it whenever
      requested_amount OR observed_monthly_revenue_avg_3mo is intervened.

We reuse A's modeling spine verbatim (make_model HGB ensemble + make_logit,
calibrated blend) so A and C share one model; only the feature frame differs.

Run:  .venv/Scripts/python.exe -m src.wf3_causal_c
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

# Reuse A's spine WITHOUT modifying it.
from src.build_a import (
    build_cat_dtypes,
    build_features,
    numeric_frame,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"

SEED = 17
N_BOOT = 15  # smaller than A's 25: this is a proof prototype, not the submission

# --- Causal exclusions ----------------------------------------------------- #
# (1) Backdoor / selection collider nodes -> drop (and their __isna indicators).
COLLIDER_DROP = ["prior_underwriter_score", "prior_decision", "prior_approved_amount"]
# (2) Self-report zero-effect fields -> drop so do() returns exact baseline.
SELF_REPORT_ZERO = ["stated_annual_revenue", "stated_time_in_business",
                    "intended_use_of_funds"]
# Engineered ratio shipped in the data -- we replace it with our own clean,
# observed-only leverage so we fully control the recompute under do().
SHIPPED_RATIO = "requested_amount_to_observed_revenue"
CLEAN_LEVERAGE = "leverage_obs"  # requested_amount / (observed_monthly_rev * 12)

# Features whose do() must return EXACTLY baseline (delta 0) because they are not
# in the causal model. Used by the query engine and the sign checks.
ZERO_EFFECT_FEATURES = set(SELF_REPORT_ZERO)

# --- Monotone causal-sign constraints -------------------------------------- #
# DIAGNOSIS (see wf3 diagnostic): existing_debt_obligations has a strongly POSITIVE
# raw marginal vs default (10.5% -> 25.9% across quintiles) but a NEGATIVE
# multivariate partial effect, because it is near-collinear with
# aggregate_credit_utilization (corr 0.63) and revenue (corr -0.39). A naive
# multivariate counterfactual therefore returns the WRONG SIGN for
# do(existing_debt up). The true interventional effect is positive (more debt ->
# more burden -> more default). We inject this domain sign STRUCTURALLY via
# HistGBT monotonic_cst (+1 = PD non-decreasing, -1 = non-increasing), and a
# sign-constrained logistic arm, so the do() effect is guaranteed directionally
# correct without hand-editing deltas. Only features with an unambiguous economic
# sign are constrained; everything else stays free (0).
MONO_SIGN = {
    "aggregate_credit_utilization": +1,
    "existing_debt_obligations": +1,
    "invoice_payment_delinquency_rate": +1,
    "observed_overdraft_count_3mo": +1,
    "recent_inquiries_count_6mo": +1,
    "multi_lender_inquiry_count_30d": +1,
    "prior_loans_default_count": +1,
    "leverage_obs": +1,                  # higher loan-to-revenue -> more default
    "observed_revenue_volatility": +1,
    "observed_cash_balance_p10": -1,     # more liquidity -> less default
    "observed_monthly_revenue_avg_3mo": -1,
    "observed_revenue_trend_3mo": -1,
    "payroll_regularity_score": -1,
    "vintage_years": -1,
    "account_age_days": -1,
}


def monotone_cst(ref_cols: list[str]) -> list[int]:
    """Per-column monotonic constraint vector aligned to the model columns."""
    return [MONO_SIGN.get(c, 0) for c in ref_cols]


class SignLogit:
    """L2 logistic regression with per-coefficient SIGN constraints.

    Mirrors make_logit's preprocessing (median impute -> standardize) but fits the
    coefficients under scipy L-BFGS-B box bounds so a feature with MONO_SIGN=+1 has
    a >=0 weight (PD up when it rises) and -1 has a <=0 weight. This makes the
    linear arm agree with the monotone HGB on the constrained causal signs, so the
    blended do() effect cannot be sign-flipped by collinearity (e.g. existing_debt
    vs aggregate_credit_utilization). Unconstrained features stay free.
    """

    def __init__(self, mono: list[int], C: float = 0.5):
        self.mono = np.asarray(mono)
        self.C = C
        self.median_ = None
        self.mean_ = None
        self.std_ = None
        self.coef_ = None
        self.intercept_ = 0.0

    def _prep(self, Z: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.median_ = np.nanmedian(Z, axis=0)
        Z = np.where(np.isnan(Z), self.median_, Z)
        if fit:
            self.mean_ = Z.mean(axis=0)
            self.std_ = Z.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        return (Z - self.mean_) / self.std_

    def fit(self, Z: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize

        Zs = self._prep(np.asarray(Z, float), fit=True)
        n, p = Zs.shape
        y = np.asarray(y, float)
        # parameter vector = [intercept, coefs...]; standardized X so signs survive.
        lam = 1.0 / (self.C * n)

        def negll(theta):
            b0, w = theta[0], theta[1:]
            z = b0 + Zs @ w
            # stable log-loss
            ll = np.logaddexp(0, z) - y * z
            reg = 0.5 * lam * np.sum(w * w)
            grad_z = (1.0 / (1.0 + np.exp(-z))) - y
            g0 = grad_z.sum()
            gw = Zs.T @ grad_z + lam * w
            return ll.sum() + reg, np.concatenate([[g0], gw])

        bounds = [(None, None)]  # intercept free
        for s in self.mono:
            if s > 0:
                bounds.append((0.0, None))     # PD non-decreasing -> w >= 0
            elif s < 0:
                bounds.append((None, 0.0))     # PD non-increasing -> w <= 0
            else:
                bounds.append((None, None))
        theta0 = np.zeros(p + 1)
        res = minimize(negll, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 3000})
        self.intercept_ = res.x[0]
        self.coef_ = res.x[1:]
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        Zs = self._prep(np.asarray(Z, float), fit=False)
        z = self.intercept_ + Zs @ self.coef_
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


def make_causal_model(seed: int, mono: list[int]) -> HistGradientBoostingClassifier:
    """A's HGB hyper-params (mirrors make_model) PLUS causal monotone signs.

    Built locally so build_a.make_model stays untouched. Note: sklearn forbids
    monotone constraints on categorical features, so MONO_SIGN only names numeric
    risk drivers (all categoricals map to 0 = free).
    """
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        categorical_features="from_dtype",
        monotonic_cst=mono,
        random_state=seed,
    )


def _isna_cols(cols: list[str]) -> list[str]:
    return [f"{c}__isna" for c in cols]


def add_clean_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """Add an observed-only leverage column = requested / (obs_monthly_rev*12).

    Deterministic descendant of requested_amount and observed_monthly_revenue.
    NaN (kept, HGB-native) when no bank feed or non-positive revenue -- exactly
    where the shipped ratio is also undefined.
    """
    out = df.copy()
    rev_annual = out["observed_monthly_revenue_avg_3mo"] * 12.0
    lev = out["requested_amount"] / rev_annual
    lev = lev.where(rev_annual > 0, np.nan)
    out[CLEAN_LEVERAGE] = lev.replace([np.inf, -np.inf], np.nan)
    return out


def causal_feature_frame(df: pd.DataFrame, cat_dtypes: dict[str, CategoricalDtype]
                         ) -> pd.DataFrame:
    """A's feature frame MINUS colliders, MINUS self-report-zero, with our clean
    observed-only leverage replacing the shipped engineered ratio."""
    df = add_clean_leverage(df)
    feat = build_features(df, cat_dtypes)
    drop = set(
        COLLIDER_DROP + _isna_cols(COLLIDER_DROP)
        + SELF_REPORT_ZERO + _isna_cols(SELF_REPORT_ZERO)
        + [SHIPPED_RATIO]  # remove the shipped ratio; keep our clean leverage
    )
    feat = feat.drop(columns=[c for c in drop if c in feat.columns])
    return feat


# --------------------------------------------------------------------------- #
# Intervention engine (the g-formula for deterministic descendants)
# --------------------------------------------------------------------------- #

LEVERAGE_PARENTS = {"requested_amount", "observed_monthly_revenue_avg_3mo"}


def apply_intervention(raw_row: pd.Series, feature: str, value: float) -> pd.Series:
    """Return a RAW applicant row with do(feature=value) applied.

    Sets the raw feature, then the engineered leverage is recomputed downstream
    by causal_feature_frame -> add_clean_leverage. For excluded zero-effect
    features the raw set is irrelevant (column not in the causal model), so do()
    naturally collapses to baseline.
    """
    row = raw_row.copy()
    if feature in row.index:
        row[feature] = value
    return row


def predict_blend(frame: pd.DataFrame, models, logit, iso, ref_cols, cat_dtypes
                  ) -> np.ndarray:
    """Calibrated blended PD for a feature frame, aligned to training columns."""
    f = frame.reindex(columns=ref_cols)
    for c, dt in cat_dtypes.items():
        if c in f.columns:
            f[c] = f[c].astype(dt)
    hgb = np.mean([m.predict_proba(f)[:, 1] for m in models], axis=0)
    lg = logit.predict_proba(numeric_frame(f).to_numpy())[:, 1]
    blend = 0.4 * hgb + 0.6 * lg
    return np.clip(iso.predict(blend), 1e-6, 1 - 1e-6)


def main() -> int:
    np.random.seed(SEED)
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    queries = pd.read_csv(DATA / "intervention_queries.csv")

    cat_dtypes = build_cat_dtypes(train, val, test)
    # prior_decision is dropped from the causal frame, but build_cat_dtypes still
    # references it harmlessly; intended_use_of_funds is also dropped.

    f_train = causal_feature_frame(train, cat_dtypes)
    ref_cols = list(f_train.columns)
    f_val = causal_feature_frame(val, cat_dtypes)

    print("[causal frame] kept", len(ref_cols), "features")
    print("  DROPPED colliders     :", COLLIDER_DROP)
    print("  DROPPED self-report-0 :", SELF_REPORT_ZERO)
    print("  shipped ratio replaced by clean:", CLEAN_LEVERAGE,
          "(in frame:", CLEAN_LEVERAGE in ref_cols, ")")
    assert SHIPPED_RATIO not in ref_cols, "shipped ratio leaked into causal frame"
    for c in COLLIDER_DROP + SELF_REPORT_ZERO:
        assert c not in ref_cols, f"{c} leaked into causal frame"

    # ---- Fit the calibrated blend on observed train rows ------------------ #
    obs = train["default_flag"].notna().to_numpy()
    y = train.loc[obs, "default_flag"].astype(int).to_numpy()
    X_obs = f_train.loc[obs]
    n = len(X_obs)
    rng = np.random.default_rng(SEED)

    mono = monotone_cst(ref_cols)
    print(f"[causal sign] {sum(s!=0 for s in mono)} monotone-constrained features "
          f"(+:{sum(s>0 for s in mono)}, -:{sum(s<0 for s in mono)})")

    models = []
    val_hgb = np.zeros((len(f_val), N_BOOT))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = make_causal_model(SEED + b + 1, mono).fit(X_obs.iloc[idx], y[idx])
        models.append(m)
        val_hgb[:, b] = m.predict_proba(f_val)[:, 1]
    hgb_val_mean = val_hgb.mean(axis=1)

    logit = SignLogit(mono, C=0.5).fit(numeric_frame(X_obs).to_numpy(), y)
    logit_val = logit.predict_proba(numeric_frame(f_val).to_numpy())[:, 1]
    blend_val = 0.4 * hgb_val_mean + 0.6 * logit_val

    vmask = val["default_flag"].notna().to_numpy()
    yv = val.loc[vmask, "default_flag"].astype(int).to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(blend_val[vmask], yv)

    print(f"[fit] observed train rows={n:,}  causal-frame val AUC(blend)="
          f"{roc_auc_score(yv, blend_val[vmask]):.4f}  HGB={roc_auc_score(yv, hgb_val_mean[vmask]):.4f}")

    # ---- Raw rows for every queried applicant ----------------------------- #
    raw = test.drop_duplicates("applicant_id").set_index("applicant_id")

    # ---- Build ONE sample query per distinct queried feature -------------- #
    # Prefer an applicant in the OBSERVED (prior_decision==1) region with a bank
    # feed so engineered leverage and bank-feed features are well defined.
    feats = list(queries["feature_name"].unique())
    samples = []
    for feat in feats:
        sub = queries[queries["feature_name"] == feat]
        chosen = None
        for _, qrow in sub.iterrows():
            aid = qrow["applicant_id"]
            arow = raw.loc[aid]
            has_feed = bool(arow.get("has_linked_bank_feed", False))
            approved = arow.get("prior_decision", 0) == 1
            if has_feed and approved:
                chosen = qrow
                break
        if chosen is None:
            chosen = sub.iloc[0]
        samples.append(chosen)
    sample_df = pd.DataFrame(samples).reset_index(drop=True)

    # ---- Baseline + counterfactual PD per sample query -------------------- #
    rows = []
    for _, qrow in sample_df.iterrows():
        aid = qrow["applicant_id"]
        feat = qrow["feature_name"]
        val_set = qrow["intervention_value"]
        raw_row = raw.loc[aid]

        base_frame = causal_feature_frame(pd.DataFrame([raw_row]), cat_dtypes)
        pd_base = predict_blend(base_frame, models, logit, iso, ref_cols, cat_dtypes)[0]

        if feat in ZERO_EFFECT_FEATURES:
            # STRUCTURAL zero effect: excluded from the model -> exact baseline.
            pd_cf = pd_base
            note = "EXCLUDED (self-report zero-effect) -> baseline"
        else:
            cf_raw = apply_intervention(raw_row, feat, val_set)
            cf_frame = causal_feature_frame(pd.DataFrame([cf_raw]), cat_dtypes)
            pd_cf = predict_blend(cf_frame, models, logit, iso, ref_cols, cat_dtypes)[0]
            if feat in LEVERAGE_PARENTS:
                note = "lever; leverage_obs recomputed"
            elif feat in ref_cols:
                note = "direct lever"
            else:
                # In query set but not a model feature and not a flagged zero:
                # non-intervenable historical/context proxy -> baseline (delta 0).
                pd_cf = pd_base
                note = "not in causal model (non-intervenable) -> baseline"
        rows.append({
            "feature": feat, "applicant": aid[:8], "do_value": val_set,
            "pd_base": pd_base, "pd_cf": pd_cf, "delta": pd_cf - pd_base, "note": note,
        })

    res = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

    print("\n" + "=" * 100)
    print("BASELINE vs do(feature=value) COUNTERFACTUAL PD  (one sample query per feature)")
    print("=" * 100)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(res[["feature", "applicant", "do_value", "pd_base", "pd_cf", "delta", "note"]]
              .to_string(index=False))

    # ---- EXPECTED-SIGN ADVERSARIAL CHECKS --------------------------------- #
    # For directional checks we re-run with a CONTROLLED do-value (a large move in
    # the risky/safe direction) so the sign is unambiguous regardless of whether
    # the sampled query value happened to be near the applicant's current value.
    def directional(feat: str, direction: str) -> tuple[float, float]:
        """Return (mean pd_base, mean pd_cf) across ALL queried applicants for feat,
        intervening to a strong value in `direction` ('up'/'down')."""
        col = raw[feat] if feat in raw.columns else None
        if col is not None:
            colnum = pd.to_numeric(col, errors="coerce")
            hi = np.nanpercentile(colnum, 95)
            lo = np.nanpercentile(colnum, 5)
        aids = queries.loc[queries["feature_name"] == feat, "applicant_id"].unique()
        base_list, cf_list = [], []
        for aid in aids:
            rr = raw.loc[aid]
            bf = causal_feature_frame(pd.DataFrame([rr]), cat_dtypes)
            pb = predict_blend(bf, models, logit, iso, ref_cols, cat_dtypes)[0]
            if feat in ZERO_EFFECT_FEATURES:
                pc = pb
            else:
                v = hi if direction == "up" else lo
                cf = apply_intervention(rr, feat, v)
                cff = causal_feature_frame(pd.DataFrame([cf]), cat_dtypes)
                pc = predict_blend(cff, models, logit, iso, ref_cols, cat_dtypes)[0]
            base_list.append(pb)
            cf_list.append(pc)
        return float(np.mean(base_list)), float(np.mean(cf_list))

    checks = [
        ("aggregate_credit_utilization", "up", "+"),
        ("existing_debt_obligations", "up", "+"),
        ("observed_cash_balance_p10", "up", "-"),
        ("requested_amount", "up", "+"),          # via leverage
        ("invoice_payment_delinquency_rate", "up", "+"),
        ("observed_overdraft_count_3mo", "up", "+"),
        ("stated_annual_revenue", "up", "0"),
        ("stated_time_in_business", "up", "0"),
    ]
    print("\n" + "=" * 100)
    print("EXPECTED-SIGN ADVERSARIAL CHECKS  (mean over all queried applicants; "
          "do-> p95 for 'up', p5 for 'down')")
    print("=" * 100)
    all_ok = True
    for feat, direction, want in checks:
        pb, pc = directional(feat, direction)
        d = pc - pb
        if want == "+":
            ok = d > 1e-4
        elif want == "-":
            ok = d < -1e-4
        else:  # "0"
            ok = abs(d) <= 1e-9
        all_ok &= ok
        print(f"  do({feat}={direction:<4}) expect {want:>1} : "
              f"pd_base={pb:.4f} -> pd_cf={pc:.4f}  delta={d:+.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("\nALL SIGN CHECKS PASS:", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
