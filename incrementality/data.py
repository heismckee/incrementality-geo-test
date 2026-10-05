"""Simulated geo-level sales panel with a known incremental effect.

Each geo (think DMA) has its own size, trend and noise, and all geos
share national seasonality. During the test window, treated geos get
extra media spend that lifts revenue by a known amount, so every
estimate can be checked against the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GeoTruth:
    lift: float                 # true relative lift in treated geos during the test
    incremental_revenue: float  # true total incremental revenue
    incremental_spend: float    # total extra spend in treated geos

    @property
    def iroas(self) -> float:
        return self.incremental_revenue / self.incremental_spend


def simulate_panel(n_geos: int = 40, pre_weeks: int = 26, seed: int = 11) -> pd.DataFrame:
    """Baseline weekly revenue per geo before any test runs (long format)."""
    rng = np.random.default_rng(seed)
    weeks = pre_weeks + 12  # room for a test window
    t = np.arange(weeks)
    national = 1 + 0.12 * np.sin(2 * np.pi * (t + 8) / 52) + 0.05 * np.sin(2 * np.pi * t / 13)
    rows = []
    for g in range(n_geos):
        size = rng.lognormal(np.log(80_000), 0.6)
        trend = rng.normal(0, 0.002)
        sensitivity = rng.normal(1.0, 0.15)  # how strongly this geo follows national seasonality
        eps = np.zeros(weeks)
        for i in range(1, weeks):  # AR(1) noise: good and bad weeks cluster
            eps[i] = 0.5 * eps[i - 1] + rng.normal(0, 0.04)
        revenue = size * (1 + sensitivity * (national - 1)) * (1 + trend * t) * (1 + eps)
        rows.append(pd.DataFrame({"geo": f"G{g:02d}", "week": t, "revenue": revenue}))
    return pd.concat(rows, ignore_index=True)


def apply_test(panel: pd.DataFrame, treated: list[str], start: int, weeks: int,
               lift: float = 0.06, spend_share: float = 0.035, seed: int = 3) -> tuple[pd.DataFrame, GeoTruth]:
    """Inject a campaign into treated geos for ``weeks`` starting at ``start``.

    Treated geos spend ``spend_share`` of their baseline revenue on the
    campaign, and revenue rises by ``lift`` (with a little geo-level
    variation). Returns the observed panel and the truth.
    """
    rng = np.random.default_rng(seed)
    df = panel[panel.week < start + weeks].copy()
    df["spend"] = 0.0
    in_test = df.geo.isin(treated) & df.week.between(start, start + weeks - 1)
    geo_lift = {g: max(0.0, rng.normal(lift, lift * 0.3)) for g in treated}
    base = df.loc[in_test, "revenue"].to_numpy()
    lifts = df.loc[in_test, "geo"].map(geo_lift).to_numpy()
    df.loc[in_test, "revenue"] = base * (1 + lifts)
    df.loc[in_test, "spend"] = base * spend_share
    truth = GeoTruth(
        lift=float((base * lifts).sum() / base.sum()),
        incremental_revenue=float((base * lifts).sum()),
        incremental_spend=float((base * spend_share).sum()),
    )
    return df, truth
