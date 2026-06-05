"""
submission.py — assemble and write submission_A_decisions.csv.

Schema (from validate_submission.py):
    applicant_id, decision, predicted_pd, pd_lower_90, pd_upper_90

Constraints:
    decision          in {0, 1}
    predicted_pd      in [0, 1]
    pd_lower_90       in [0, 1]
    pd_upper_90       in [0, 1]
    pd_lower_90 <= predicted_pd <= pd_upper_90

Public API
----------
build_submission_a(val_all, test, val_cal_all, test_cal,
                   decisions, lower, upper, sub_dir)
    -> (submission_df, output_path)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import SUB_DIR


def build_submission_a(
    val_all: pd.DataFrame,
    test: pd.DataFrame,
    val_cal_all: np.ndarray,
    test_cal: np.ndarray,
    decisions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sub_dir=SUB_DIR,
) -> tuple:
    """
    Combine val_all + test predictions into the required 13 306-row CSV.

    Parameters
    ----------
    val_all     : full validation DataFrame (all 4 489 rows, for applicant_ids)
    test        : test DataFrame            (all 8 817 rows, for applicant_ids)
    val_cal_all : calibrated PDs for all val rows,  shape (4 489,)
    test_cal    : calibrated PDs for all test rows, shape (8 817,)
    decisions   : binary approval array,  shape (13 306,) — val then test order
    lower/upper : conformal interval bounds, shape (13 306,)
    sub_dir     : directory where the CSV is written (created if absent)

    Returns
    -------
    submission  : pd.DataFrame  the assembled submission table
    out_path    : Path          where the file was saved
    """
    sub_dir = Path(sub_dir)
    sub_dir.mkdir(exist_ok=True)

    all_ids = list(val_all['applicant_id']) + list(test['applicant_id'])
    all_pds = np.concatenate([val_cal_all, test_cal])

    submission = pd.DataFrame({
        'applicant_id' : all_ids,
        'decision'     : decisions,
        'predicted_pd' : all_pds,
        'pd_lower_90'  : lower,
        'pd_upper_90'  : upper,
    })

    out_path = sub_dir / 'submission_A_decisions.csv'
    submission.to_csv(out_path, index=False)
    return submission, out_path
