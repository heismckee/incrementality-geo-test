"""Tests for the incrementality package. Run with: pytest -q"""
import numpy as np

from incrementality import (apply_test, estimate, matched_pair_assignment, null_distribution,
                            placebo_inference, simulate_panel, to_wide)

PRE, TW = 26, 6


def _setup(lift=0.06, seed=11):
    panel = simulate_panel(seed=seed)
    wide_pre = to_wide(panel).iloc[:PRE]
    treated = matched_pair_assignment(wide_pre, seed=42)
    observed, truth = apply_test(panel, treated, PRE, TW, lift=lift)
    return wide_pre, treated, to_wide(observed), truth


def test_matched_pairs_split_evenly():
    wide_pre, treated, _, _ = _setup()
    assert len(treated) == wide_pre.shape[1] // 2
    assert len(set(treated)) == len(treated)


def test_no_effect_means_small_estimate():
    wide_pre, treated, wide, _ = _setup(lift=0.0)
    for method in ["synthetic_control", "diff_in_diff"]:
        assert abs(estimate(wide, treated, PRE, TW, method)["lift"]) < 0.05


def test_estimators_recover_true_lift():
    _, treated, wide, truth = _setup(lift=0.06)
    for method in ["synthetic_control", "diff_in_diff"]:
        assert abs(estimate(wide, treated, PRE, TW, method)["lift"] - truth.lift) < 0.03


def test_synthetic_control_fits_pre_period_tightly():
    _, treated, wide, _ = _setup()
    assert estimate(wide, treated, PRE, TW)["pre_fit_mape"] < 0.02


def test_real_effect_is_significant():
    wide_pre, treated, wide, _ = _setup(lift=0.06)
    est = estimate(wide, treated, PRE, TW)
    inf = placebo_inference(est["lift"], null_distribution(wide_pre, TW, n_sims=150))
    assert inf["p_value"] < 0.05 and inf["ci_low"] > 0


def test_placebo_null_is_centered_on_zero():
    wide_pre, _, _, _ = _setup()
    null = null_distribution(wide_pre, TW, n_sims=150)
    assert abs(np.mean(null)) < 0.01
