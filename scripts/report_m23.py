"""Render every registered M23 condition, without selecting favorable outcomes."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    result = json.loads((source / "results.json").read_text())
    if result["engineering"] != "PASS":
        raise ValueError("Cannot produce success report for failed engineering run")
    output.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    primary = summary["primary"]
    keys = ["H_A", "H_B", "H_V", "H_K", "H_S"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(keys):
        row = primary[key]
        lo, hi = row["ci"]
        ax.errorbar(
            row["mean"], i, xerr=[[row["mean"] - lo], [hi - row["mean"]]], fmt="o", capsize=4
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(
        range(5),
        [
            "A: raw minus coordinate",
            "B: raw minus coordinate",
            "B: validation16 minus256",
            "B: K16 minus K4",
            "C-B: fixed-model population risk",
        ],
    )
    ax.set_xlabel("Paired MSE contrast; preregistered 99.1667% intervals")
    ax.set_title("All five primary mechanistic contrasts; signs have different meanings")
    fig.tight_layout()
    fig.savefig(output / "primary.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = dict(
        zip(("unscaled", "standardized", "energy", "mlp"), ("C0", "C1", "C2", "C3"), strict=True)
    )
    for ax, scenario in zip(axes, ("A", "B", "C"), strict=True):
        for method, color in colors.items():
            for n, style in ((16, "--"), (256, "-")):
                rows = [
                    r
                    for r in summary["reliability"]
                    if r["scenario"] == scenario
                    and r["method"] == method
                    and r["n_validation"] == n
                ]
                ax.plot(
                    [r["k"] for r in rows],
                    [r["test_mse_mean"] for r in rows],
                    style,
                    marker="o",
                    color=color,
                    label=f"{method}, val{n}",
                )
        ax.set_title(scenario)
        ax.set_xticks([4, 16])
        ax.set_xlabel("Search candidates")
        ax.set_ylabel("Mean test MSE")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "selection.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in ("unscaled", "standardized", "energy"):
        ranks = [
            np.mean([u["geometry"][s][method]["effective_rank"] for u in result["units"]])
            for s in ("A", "B", "C")
        ]
        condition = [
            np.mean([u["geometry"][s][method]["condition_number"] for u in result["units"]])
            for s in ("A", "B", "C")
        ]
        axes[0].plot(["A", "B", "C"], ranks, marker="o", label=method)
        axes[1].plot(["A", "B", "C"], condition, marker="o", label=method)
    axes[0].set_ylabel("Training effective rank")
    axes[1].set_ylabel("Training matrix condition number")
    axes[1].set_yscale("log")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "geometry.svg")
    plt.close(fig)
    rows = list(csv.DictReader((source / "candidates.csv").open()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, scenario in zip(axes, ("A", "B", "C"), strict=True):
        for method in ("unscaled", "standardized", "energy"):
            chosen = [
                r
                for r in rows
                if r["scenario"] == scenario and r["method"] == method and int(r["index"]) == 5
            ]
            ax.scatter(
                [float(r["coefficient_error"]) for r in chosen],
                [float(r["test_mse"]) for r in chosen],
                s=12,
                alpha=0.6,
                label=method,
            )
        ax.set_title(scenario)
        ax.set_xlabel("Normalized latent coefficient error")
        ax.set_ylabel("Test MSE")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "recovery.svg")
    plt.close(fig)
    lines = [
        "# M23 — representation geometry and validation selection",
        "",
        "**Engineering PASS.** Scientific decisions below use the preregistered thresholds; no improvement is assumed.",
        "",
        "## Primary decisions",
        "",
        "| Hypothesis | Paired mean | Simultaneous interval | Decision |",
        "| --- | ---: | --- | --- |",
    ]
    for key in keys:
        r = primary[key]
        lines.append(
            f"| {key} | {r['mean']:.9f} | [{r['ci'][0]:.9f}, {r['ci'][1]:.9f}] | {r['decision']} |"
        )
    lines += [
        "",
        f"**Strong fixed comparator decision:** {primary['H_G']['decision']}.",
        "",
        "| B selected coordinate vs fixed comparator | Mean gain | Conservative interval |",
        "| --- | ---: | --- |",
    ]
    for method in colors:
        r = primary["H_G"][method]
        lines.append(f"| {method} | {r['mean']:.9f} | [{r['ci'][0]:.9f}, {r['ci'][1]:.9f}] |")
    lines += [
        "",
        "Positive H_A/H_B gain is raw minus standardized MSE; H_V is small minus large validation MSE; H_K is large minus small candidate-set MSE. H_S is analytic covariance-risk change with identical fitted state. Intervals use 40 paired independent simulation units and the preregistered multiplicity correction. HG requires a lower bound above 0.10 against every fixed baseline, including standardized ridge itself.",
        "",
        "## All validation-selection conditions",
        "",
        "| Scenario | Method | Validation | K | Mean test MSE | Wins / losses / ties vs fixed | Mean population oracle regret |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in summary["reliability"]:
        regret = (
            "not defined"
            if r["population_oracle_regret_mean"] is None
            else f"{r['population_oracle_regret_mean']:.6f}"
        )
        lines.append(
            f"| {r['scenario']} | {r['method']} | {r['n_validation']} | {r['k']} | {r['test_mse_mean']:.6f} | {r['wins']} / {r['losses']} / {r['ties']} | {regret} |"
        )
    lines += [
        "",
        "Wilson intervals for win/loss rates, 95% paired effects, selected penalties, optimism and gaps are in results.json and selections.csv. Every candidate and seed is retained in candidates.csv. Oracle quantities were computed after selection and never chose a model.",
        "",
        "## Controls and diagnostics",
        "",
        f"Maximum scalar/diagonal prediction-equivalence absolute error: {summary['max_numerical_error']:.17g}, tested with rtol=1e-8/atol=1e-9.",
        f"Label-permutation A/unscaled mean Spearman: {summary['negative_control_A_unscaled']['mean']:.9f}; descriptive 95% interval {summary['negative_control_A_unscaled']['ci']}.",
        "",
        "Every fitted candidate and permutation control was independently refitted with exact state equality. A/B retain identical latent task/labels; B/C share exactly the same training states. Training-only preprocessing and evaluation-label noninterference pass. Redundant fourfold search copies do not change selections. All seed hashes, geometry diagnostics, controls and source/environment identity are recorded.",
        "",
        "![All primary contrasts](primary.svg)",
        "![All selection conditions](selection.svg)",
        "![Training representation diagnostics](geometry.svg)",
        "![Mechanistic recovery versus accuracy](recovery.svg)",
        "",
        "## Interpretation limits",
        "",
        "Scaling geometry is identified by paired changes of feature units and exact diagonal-penalty identities, not a change of underlying labels. Global energy normalization with the matched penalty is only a scalar reparameterization. The H_A equivalence criterion is distinct from failing to reject a difference. Coefficient recovery is a separate secondary diagnostic from test MSE; covariance weights coefficient errors differently.",
        "",
        "Validation size and search-grid density are independently varied. Larger candidate sets also improve available oracle opportunity; H_K is not a pure test of independent multiple comparisons. Selection optimism and population oracle regret distinguish finite validation noise from that opportunity. The MLP has the same candidate counts but not equal capacity, compute cost or effective regularization geometry. It is secondary, with no convergence or fairness-by-wall-time claim.",
        "",
        "C is covariate shift with unchanged conditional coefficients/noise. decomposition.csv separately freezes the selected B model on C test data and then changes the validation-selected configuration on that same C test set. The terms sum to the total paired difference; they are not independent interventions on every property.",
        "",
        "These controlled Gaussian tasks establish only their own numerical mechanisms. They do not explain M17/M19 biological results, validate external protein generalization, establish optimal model quality, or authorize biological follow-up. Synthetic seeds are independent simulation units here; historical protein-family seeds were not independent families. No test-informed grid, scenario, tolerance or threshold revision occurred.",
        "",
        "## Reproduction and preservation",
        "",
        "Run from the registered checkout with development dependencies and matplotlib. Use a fresh runs/m23 output path; overwrites are refused.",
        "",
        "```sh",
        "pytest -q tests/test_m23_geometry.py",
        "ruff check .",
        "ruff format --check .",
        "python scripts/verify_m23.py --local-refs",
        "python scripts/run_m23.py --output runs/m23/reproduction",
        "python scripts/verify_m23.py --result runs/m23/reproduction/results.json",
        "python scripts/report_m23.py --source runs/m23/reproduction --output runs/m23/reproduction/plots",
        "```",
        "",
        "The protocol/config were committed before implementation; implementation freeze and execution metadata are in implementation-freeze.json and verification.json. All historical research code/evidence remains intact. CI_ADDENDUM.md documents the only two CI plumbing overlays, with original bytes archived and verified explicitly. No old report, result, threshold, trainer or frozen manifest was rewritten. Existing M22 numerical/state helpers are reused. No biological data, optimization, release or merge.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
