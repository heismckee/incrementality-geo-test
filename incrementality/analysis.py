"""Estimating incremental lift from a geo test.

Two estimators, both answering "what would treated geos have done without
the campaign?":

* Synthetic control: build a weighted blend of control geos that tracks the
  treated group closely before the test, then use that blend as the
  counterfactual during the test.
* Difference-in-differences (ratio form): assume treated geos keep the same
  ratio to control geos they had before the test.

Significance comes from placebo tests. We rerun the exact same analysis on
many random fake treatment groups where nothing happened. If the real
estimate is larger than almost all of those, it's unlikely to be noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def _split(wide: pd.DataFrame, treated: list[str], start: int, weeks: int):
    controls = [g for g in wide.columns if g not in treated]
    idx = np.arange(len(wide))
    pre = idx < start
    test = (idx >= start) & (idx < start + weeks)
    return controls, pre, test


def synthetic_control(wide: pd.DataFrame, treated: list[str], start: int, weeks: int):
    """Non-negative weights on control geos fit to the treated total pre-test."""
    controls, pre, test = _split(wide, treated, start, weeks)
    y = wide[treated].sum(axis=1).to_numpy()
    X = wide[controls].to_numpy()
    w, _ = nnls(X[pre], y[pre])
    counterfactual = X @ w
    return y, counterfactual, pd.Series(w, index=controls), pre, test


def diff_in_diff(wide: pd.DataFrame, treated: list[str], start: int, weeks: int):
    controls, pre, test = _split(wide, treated, start, weeks)
    y = wide[treated].sum(axis=1).to_numpy()
    c = wide[controls].sum(axis=1).to_numpy()
    ratio = y[pre].sum() / c[pre].sum()
    return y, c * ratio, None, pre, test


def estimate(wide: pd.DataFrame, treated: list[str], start: int, weeks: int,
             method: str = "synthetic_control") -> dict:
    fn = {"synthetic_control": synthetic_control, "diff_in_diff": diff_in_diff}[method]
    y, cf, weights, pre, test = fn(wide, treated, start, weeks)
    incremental = float((y[test] - cf[test]).sum())
    pre_fit_mape = float(np.mean(np.abs(y[pre] - cf[pre]) / y[pre]))
    return {
        "method": method,
        "incremental_revenue": incremental,
        "lift": incremental / float(cf[test].sum()),
        "pre_fit_mape": pre_fit_mape,
        "actual": y,
        "counterfactual": cf,
        "weights": weights,
    }


def placebo_inference(observed_lift: float, null: np.ndarray, level: float = 0.9) -> dict:
    """Two-sided p-value and confidence interval from a placebo distribution.

    The interval inverts the placebo errors: if placebo estimates spread by
    ±x around zero, the true lift is plausibly within ±x of our estimate.
    """
    p = float((np.sum(np.abs(null) >= abs(observed_lift)) + 1) / (len(null) + 1))
    lo_q, hi_q = np.quantile(null, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"p_value": p, "ci_low": observed_lift - hi_q, "ci_high": observed_lift - lo_q, "level": level}
