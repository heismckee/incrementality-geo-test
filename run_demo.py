"""End-to-end geo incrementality test: power, design, launch, read, validate.

Run:  python run_demo.py
Writes charts and a results summary to ./outputs
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from incrementality import (apply_test, estimate, matched_pair_assignment, minimum_detectable_effect,
                            null_distribution, placebo_inference, power_curve, simulate_panel, to_wide)

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
NAVY, BLUE, ORANGE, GREEN, GRAY = "#1F3864", "#2E86AB", "#F18F01", "#6A994E", "#999999"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

PRE_WEEKS, TEST_WEEKS, PLANNED_LIFT = 26, 6, 0.06


def money(x):
    return f"${x:,.0f}"


def main():
    panel = simulate_panel(n_geos=40, pre_weeks=PRE_WEEKS, seed=11)
    wide_pre = to_wide(panel).iloc[:PRE_WEEKS]

    # 1. Power analysis on history only: can this design detect the lift we expect?
    curve = power_curve(wide_pre, TEST_WEEKS, n_sims=200)
    mde = minimum_detectable_effect(curve)
    curve.to_csv(OUT / "power_curve.csv", index=False)

    # 2. Assignment: matched pairs by pre-period revenue.
    treated = matched_pair_assignment(wide_pre, seed=42)
    pd.DataFrame({"geo": wide_pre.columns, "group": ["treatment" if g in treated else "control" for g in wide_pre.columns]}) \
        .to_csv(OUT / "assignment.csv", index=False)

    # 3. Run the test (simulated) and read it with both estimators.
    observed, truth = apply_test(panel, treated, PRE_WEEKS, TEST_WEEKS, lift=PLANNED_LIFT)
    observed.to_csv(OUT / "simulated_geo_panel.csv", index=False)
    wide = to_wide(observed)
    spend = observed.spend.sum()
    results = {}
    for method in ["synthetic_control", "diff_in_diff"]:
        est = estimate(wide, treated, PRE_WEEKS, TEST_WEEKS, method)
        null = null_distribution(wide_pre, TEST_WEEKS, n_sims=300, method=method)
        inf = placebo_inference(est["lift"], null)
        cf_test = est["counterfactual"][PRE_WEEKS:PRE_WEEKS + TEST_WEEKS].sum()
        results[method] = {**est, **inf, "null": null,
                           "iroas": est["incremental_revenue"] / spend,
                           "iroas_low": inf["ci_low"] * cf_test / spend,
                           "iroas_high": inf["ci_high"] * cf_test / spend}

    # 4. Validation: repeat the whole test 100 times with fresh noise and assignments.
    reps = []
    for s in range(100):
        p = simulate_panel(n_geos=40, pre_weeks=PRE_WEEKS, seed=1000 + s)
        wp = to_wide(p).iloc[:PRE_WEEKS]
        tr = matched_pair_assignment(wp, seed=s)
        obs, t = apply_test(p, tr, PRE_WEEKS, TEST_WEEKS, lift=PLANNED_LIFT, seed=s)
        e = estimate(to_wide(obs), tr, PRE_WEEKS, TEST_WEEKS)
        inf = placebo_inference(e["lift"], null_distribution(wp, TEST_WEEKS, n_sims=150, seed=s))
        reps.append({"true": t.lift, "est": e["lift"], "covered": inf["ci_low"] <= t.lift <= inf["ci_high"],
                     "significant": inf["p_value"] < 0.05})
    reps = pd.DataFrame(reps)
    reps.to_csv(OUT / "validation_runs.csv", index=False)

    sc = results["synthetic_control"]

    # ---------- charts ----------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve.true_lift * 100, curve.power * 100, marker="o", color=NAVY, lw=2)
    ax.axhline(80, color=GRAY, ls="--", lw=1)
    if mde is not None:
        ax.axvline(mde * 100, color=ORANGE, ls="--", lw=1.5, label=f"MDE at 80% power: {mde:.0%}")
    ax.axvline(PLANNED_LIFT * 100, color=GREEN, ls=":", lw=2, label=f"Expected lift: {PLANNED_LIFT:.0%}")
    ax.set_xlabel("True lift (%)"); ax.set_ylabel("Chance of detecting it (%)")
    ax.set_title(f"Power analysis: 40 geos, {TEST_WEEKS}-week test, matched pairs")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "01_power_curve.png"); plt.close(fig)

    weeks = np.arange(len(sc["actual"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(PRE_WEEKS - 0.5, PRE_WEEKS + TEST_WEEKS - 0.5, color=ORANGE, alpha=0.08, label="Test window")
    ax.plot(weeks, sc["actual"], color=NAVY, lw=2, label="Treated geos (actual)")
    ax.plot(weeks, sc["counterfactual"], color=BLUE, lw=2, ls="--", label="Synthetic control (what would have happened)")
    ax.set_title(f"Treated geos vs. synthetic control (pre-test fit error {sc['pre_fit_mape']:.1%})")
    ax.set_xlabel("Week"); ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e6:,.2f}M")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "02_synthetic_control.png"); plt.close(fig)

    tw = slice(PRE_WEEKS, PRE_WEEKS + TEST_WEEKS)
    cum = np.cumsum(sc["actual"][tw] - sc["counterfactual"][tw])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, TEST_WEEKS + 1), cum, color=NAVY)
    ax.axhline(truth.incremental_revenue, color=ORANGE, ls="--", label=f"True incremental: {money(truth.incremental_revenue)}")
    ax.set_title("Cumulative incremental revenue during the test")
    ax.set_xlabel("Test week"); ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e3:,.0f}K")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "03_cumulative_incremental.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sc["null"] * 100, bins=30, color=GRAY, alpha=0.7, label="Placebo tests (no real effect)")
    ax.axvline(sc["lift"] * 100, color=NAVY, lw=2.5, label=f"Observed lift: {sc['lift']:.1%} (p = {sc['p_value']:.3f})")
    ax.set_xlabel("Estimated lift (%)"); ax.set_ylabel("Placebo tests")
    ax.set_title("Is the result real? Observed lift vs. 300 placebo tests")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "04_placebo_test.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = {"synthetic_control": "Synthetic control", "diff_in_diff": "Diff-in-diff"}
    for i, (m, r) in enumerate(results.items()):
        ax.hlines(i, r["ci_low"] * 100, r["ci_high"] * 100, color=BLUE, lw=6, alpha=0.5)
        ax.scatter(r["lift"] * 100, i, color=NAVY, zorder=3)
    ax.axvline(truth.lift * 100, color=ORANGE, ls="--", label=f"True lift {truth.lift:.1%}")
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_yticks(range(len(results)), [names[m] for m in results])
    ax.set_xlabel("Lift (%) with 90% placebo interval")
    ax.set_title("Both estimators recover the true lift")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "05_method_comparison.png"); plt.close(fig)

    # ---------- written summary ----------
    lines = [
        "# Results",
        "",
        (f"* Minimum detectable effect at 80% power: **{mde:.0%}**; power at the expected {PLANNED_LIFT:.0%} lift: "
         f"**{curve.loc[np.isclose(curve.true_lift, PLANNED_LIFT), 'power'].iloc[0]:.0%}**") if mde else "* Test is underpowered at tested lifts",
        f"* True lift: **{truth.lift:.2%}** | True incremental revenue: **{money(truth.incremental_revenue)}** | True iROAS: **{truth.iroas:.2f}**",
        "",
        "| Method | Est. lift | 90% interval | p-value | Incremental revenue | iROAS (90% interval) | Pre-test fit error |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in results.items():
        lines.append(f"| {names[m]} | {r['lift']:.2%} | {r['ci_low']:.2%} – {r['ci_high']:.2%} | {r['p_value']:.3f} | "
                     f"{money(r['incremental_revenue'])} | {r['iroas']:.2f} ({r['iroas_low']:.2f} – {r['iroas_high']:.2f}) | {r['pre_fit_mape']:.2%} |")
    lines += [
        "",
        "## Validation across 100 repeated tests",
        f"* Average estimated lift: **{reps.est.mean():.2%}** vs. average true lift **{reps.true.mean():.2%}**",
        f"* 90% interval contained the truth in **{reps.covered.mean():.0%}** of tests",
        f"* Effect detected (p < 0.05) in **{reps.significant.mean():.0%}** of tests",
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
