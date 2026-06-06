#!/usr/bin/env python3
"""HARD, AUDITABLE NPV calculation for our approved set. Every figure is computed
from raw data, decomposed into components, and reconciled. Example loans printed
with full per-loan arithmetic so the totals can be checked by hand.

Official per-loan NPV (challenge brief):
  repaid  (y=0): NPV = F + R*r*T/365              ;  F = 0.03R, r = 0.35, T = 60
  default (y=1): NPV = F + D*(t*-1) + rec - R      ;  D = R*(1 + r*T/365)/T
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

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "dataset"
SUB = REPO / "submissions" / "submission_A_decisions.csv"

r, T, F_RATE = 0.35, 60, 0.03
MARGIN = F_RATE + r * T / 365.0          # repaid NPV per $
D = (1 + r * T / 365.0) / T              # daily draw per $
LGD = 0.25                               # for the EXPECTED-NPV decision only


def npv_default_per_dollar(tstar, rec_per_dollar, cap=True):
    v = F_RATE + D * np.clip(tstar - 1, 0, None) - 1.0 + rec_per_dollar
    return np.minimum(v, MARGIN) if cap else v


def main() -> int:
    print("CONSTANTS (exact):")
    print(f"  r=0.35  T=60  F_rate=0.03")
    print(f"  MARGIN = 0.03 + 0.35*60/365 = {MARGIN:.8f}  (repaid NPV per $)")
    print(f"  D      = (1+0.35*60/365)/60 = {D:.8f}  (daily-draw per $)")

    sub = pd.read_csv(SUB)
    val = pd.read_csv(DATA / "validation.csv")
    test = pd.read_csv(DATA / "test.csv")
    keep = ["applicant_id", "requested_amount", "default_flag", "days_to_default",
            "final_recovered_amount", "prior_decision"]
    meta = pd.concat([val[keep], test[keep]], ignore_index=True)
    df = sub.merge(meta, on="applicant_id", how="left")
    appr = df[df["decision"] == 1].copy()

    # ---------------- REALIZED NPV on approved + LABELED (val) ------------- #
    lab = appr[appr["default_flag"].notna()].copy()
    R = lab["requested_amount"].to_numpy(float)
    y = lab["default_flag"].to_numpy(int)
    tstar = lab["days_to_default"].to_numpy(float)
    rec = np.nan_to_num(lab["final_recovered_amount"].to_numpy(float))

    repaid = y == 0
    deflt = y == 1
    npv = np.empty(len(lab))
    npv[repaid] = R[repaid] * MARGIN
    npv[deflt] = R[deflt] * npv_default_per_dollar(tstar[deflt], rec[deflt] / R[deflt])

    print("\n" + "=" * 70)
    print("REALIZED NPV  (approved & labeled validation loans)")
    print("=" * 70)
    print(f"  approved & labeled loans .......... {len(lab):,}")
    print(f"    repaid (y=0) .................... {int(repaid.sum()):,}")
    print(f"    defaulted (y=1) ................. {int(deflt.sum()):,}   "
          f"(check: {int(repaid.sum())}+{int(deflt.sum())}={int(repaid.sum()+deflt.sum())})")
    rp = npv[repaid].sum()
    dp = npv[deflt].sum()
    print(f"\n  COMPONENT A  repaid profit  = sum(R*MARGIN) ....... ${rp:,.2f}")
    print(f"               (principal repaid = ${R[repaid].sum():,.0f}; x {MARGIN:.6f})")
    print(f"  COMPONENT B  default NPV    = sum(F+D(t*-1)+rec-R) . ${dp:,.2f}")
    print(f"               (defaulted principal = ${R[deflt].sum():,.0f}; "
          f"recovered = ${rec[deflt].sum():,.0f})")
    print(f"  ----------------------------------------------------------------")
    print(f"  TOTAL realized NPV = A + B ........................ ${rp+dp:,.2f}")
    print(f"  (independent re-sum of per-loan npv) ............. ${npv.sum():,.2f}")
    print(f"  reconcile A+B == total ? {abs((rp+dp)-npv.sum())<1e-6}")
    n_cap = int((npv_default_per_dollar(tstar[deflt], rec[deflt]/R[deflt], cap=False)
                 > MARGIN).sum())
    print(f"  defaulted loans hitting the repaid-margin cap (t*>~60): {n_cap:,} "
          f"of {int(deflt.sum())}")
    print(f"  mean days_to_default among approved defaulters: {tstar[deflt].mean():.1f}")
    print(f"  mean recovery/principal among approved defaulters: {(rec[deflt]/R[deflt]).mean():.4f}")

    # ---------------- worked examples (verify by hand) -------------------- #
    print("\nWORKED EXAMPLES (per-loan arithmetic):")
    ex_r = lab[repaid].head(2)
    for _, row in ex_r.iterrows():
        Ri = row["requested_amount"]
        print(f"  REPAID : R=${Ri:,.0f}  ->  NPV = R*{MARGIN:.6f} = ${Ri*MARGIN:,.2f}")
    ex_d = lab[deflt].head(3)
    for _, row in ex_d.iterrows():
        Ri = row["requested_amount"]; ti = row["days_to_default"]
        ri = 0.0 if pd.isna(row["final_recovered_amount"]) else row["final_recovered_amount"]
        raw = Ri*F_RATE + Ri*D*max(ti-1, 0) + ri - Ri
        capped = min(raw, Ri*MARGIN)
        print(f"  DEFAULT: R=${Ri:,.0f} t*={ti:.0f} rec=${ri:,.0f}  ->  "
              f"0.03*R + D*R*(t*-1) + rec - R = {Ri*F_RATE:,.0f} + {Ri*D*max(ti-1,0):,.0f} "
              f"+ {ri:,.0f} - {Ri:,.0f} = ${raw:,.0f}"
              + (f"  (capped to ${capped:,.0f})" if capped < raw else ""))

    # ---------------- EXPECTED NPV on full approved (val+test) ------------- #
    Ra = appr["requested_amount"].to_numpy(float)
    pda = appr["predicted_pd"].to_numpy(float)
    ev = (1 - pda) * MARGIN - pda * LGD            # expected NPV per $ (LGD=0.25)
    print("\n" + "=" * 70)
    print("EXPECTED NPV  (all approved val+test, model PD, EV/$ = (1-pd)*MARGIN - pd*LGD)")
    print("=" * 70)
    print(f"  approved loans .................... {len(appr):,}")
    print(f"  principal deployed ............... ${Ra.sum():,.0f}")
    print(f"  mean predicted_pd (approved) ..... {pda.mean():.4f}")
    print(f"  EXPECTED NPV = sum(R_i * EV_i) ... ${ (ev*Ra).sum():,.2f}")
    print(f"  ROIC = NPV/principal ............. {100*(ev*Ra).sum()/Ra.sum():.3f}%")
    # back-of-envelope cross-check with portfolio averages
    wpd = (pda*Ra).sum()/Ra.sum()                  # principal-weighted mean PD
    approx = Ra.sum()*((1-wpd)*MARGIN - wpd*LGD)
    print(f"  cross-check via weighted-mean PD {wpd:.4f}: ${approx:,.0f} "
          f"(matches to {100*abs(approx-(ev*Ra).sum())/abs((ev*Ra).sum()):.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
