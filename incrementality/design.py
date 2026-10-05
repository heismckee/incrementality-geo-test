"""Test design: who gets treated, and can this test detect the effect we care about?

Two questions have to be answered before spending a dollar on a geo test:

1. Assignment. Randomizing within matched pairs of similar-sized geos keeps
   treatment and control balanced, which shrinks noise.
2. Power. Using only historical data, how small a lift could this design
   reliably detect? If the minimum detectable effect (MDE) is larger than
   the lift you expect, the test is a coin flip and shouldn't run as designed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import estimate


def to_wide(panel: pd.DataFrame, value: str = "revenue") -> pd.DataFrame:
    return panel.pivot(index="week", columns="geo", values=value).sort_index()


def matched_pair_assignment(wide_pre: pd.DataFrame, seed: int = 0) -> list[str]:
    """Pair geos by pre-period revenue and randomly treat one of each pair."""
    rng = np.random.default_rng(seed)
    order = wide_pre.sum().sort_values().index.tolist()
    treated = []
    for i in range(0, len(order) - 1, 2):
        treated.append(order[i + rng.integers(0, 2)])
    return treated


def null_distribution(wide_pre: pd.DataFrame, test_weeks: int, n_sims: int = 300,
                      method: str = "synthetic_control", seed: int = 0) -> np.ndarray:
    """Estimated lift when there is no effect, across many random assignments.

    Uses the last ``test_weeks`` of historical data as a fake test window.
    The spread of these placebo estimates is the noise floor of the design.
    """
    rng = np.random.default_rng(seed)
    fake_start = len(wide_pre) - test_weeks
    out = []
    for _ in range(n_sims):
        treated = matched_pair_assignment(wide_pre, seed=int(rng.integers(1e9)))
        out.append(estimate(wide_pre, treated, fake_start, test_weeks, method)["lift"])
    return np.array(out)


def power_curve(wide_pre: pd.DataFrame, test_weeks: int, lifts=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08),
                n_sims: int = 200, alpha: float = 0.05, method: str = "synthetic_control",
                seed: int = 0) -> pd.DataFrame:
    """Share of simulated tests that detect each true lift at significance ``alpha``."""
    rng = np.random.default_rng(seed)
    null = null_distribution(wide_pre, test_weeks, n_sims, method, seed)
    threshold = np.quantile(np.abs(null), 1 - alpha)
    fake_start = len(wide_pre) - test_weeks
    rows = []
    for lift in lifts:
        hits = 0
        for _ in range(n_sims):
            treated = matched_pair_assignment(wide_pre, seed=int(rng.integers(1e9)))
            w = wide_pre.copy()
            w.loc[w.index[fake_start:], treated] *= 1 + lift
            est = estimate(w, treated, fake_start, test_weeks, method)["lift"]
            hits += est > threshold
        rows.append({"true_lift": lift, "power": hits / n_sims})
    return pd.DataFrame(rows)


def minimum_detectable_effect(curve: pd.DataFrame, target_power: float = 0.8) -> float | None:
    """Smallest lift in the curve detected at least ``target_power`` of the time."""
    ok = curve[curve.power >= target_power]
    return float(ok.true_lift.min()) if len(ok) else None
