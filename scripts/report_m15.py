"""M15 family/seed and low-label reports using existing rank-metric aggregation."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from proteinmod.data import write_json
from proteinmod.edge_report import summarize
from proteinmod.structure_pipeline import write_csv


def report(root):
    with (root / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    results = {k: [] for k in ("summary", "per_protein", "per_family", "failure_slices")}
    for fraction in sorted({float(r["fraction"]) for r in predictions}):
        rows = [r | {"cohort": "family"} for r in predictions if float(r["fraction"]) == fraction]
        values = summarize(rows)
        for name, records in results.items():
            records.extend(r | {"fraction": fraction} for r in values[name])
    assert (
        len(results["summary"])
        == len(json.loads(Path("configs/m15.json").read_text())["models"]) * 6
    )
    write_json(root / "summary.json", results["summary"])
    for name in ("per_protein", "per_family", "failure_slices"):
        write_csv(root / (name + ".csv"), results[name])
    runs = json.loads((root / "metrics.json").read_text())
    overfit = []
    for r in runs:
        s = r["selection"]
        if "epoch" in s:
            overfit.append(
                {k: r[k] for k in ("fraction", "protocol", "fold", "model", "seed")}
                | {k: s[k] for k in ("epoch", "train_score", "validation_score", "parameters")}
                | {
                    "learning_rate": s.get("learning_rate"),
                    "adapter_parameters": s.get("adapter_parameters"),
                }
            )
    write_csv(root / "overfitting.csv", overfit)
    costs = []
    for model in sorted({r["model"] for r in runs}):
        selected = [r for r in runs if r["model"] == model]
        costs.append(
            {
                "model": model,
                "selected_fits": len(selected),
                "total_seconds": sum(r["seconds"] for r in selected),
                "median_seconds": float(np.median([r["seconds"] for r in selected])),
                "trainable_parameters": sorted(
                    {
                        r["selection"]["parameters"]
                        for r in selected
                        if "parameters" in r["selection"]
                    }
                ),
                "candidate_executions": sum(
                    len(r["selection"].get("candidates", [None])) for r in selected
                ),
            }
        )
    for cost in costs:
        cost["by_fraction"] = [
            {
                "fraction": fraction,
                "selected_fits": len(selected),
                "median_seconds": float(np.median([r["seconds"] for r in selected])),
                "total_seconds": sum(r["seconds"] for r in selected),
            }
            for fraction in (0.25, 1.0)
            if (
                selected := [
                    r for r in runs if r["model"] == cost["model"] and r["fraction"] == fraction
                ]
            )
        ]
    for cost in costs:
        if not cost["trainable_parameters"]:
            run = next(r for r in runs if r["model"] == cost["model"])
            with np.load(run["state_path"]) as state:
                cost["trainable_parameters"] = [int(state["coefficient"].shape[-1]) + 1]
    write_json(root / "costs.json", costs)
    models = json.loads(Path("configs/m15.json").read_text())["models"]
    lines = [
        "# M15 measured local results",
        "",
        "Protein-macro Spearman; equal family weights (two proteins per family).",
        "Neural mean ± population seed SD; deterministic ridges use one fit. No pooled raw RMSE.",
        "",
    ]
    for fraction in (0.25, 1.0):
        lines += [
            f"## Original training-label fraction {fraction:g}",
            "",
            "| Model | G1 | G2 | G3 |",
            "|---|---:|---:|---:|",
        ]
        for model in models:
            vals = []
            for protocol in ("G1", "G2", "G3"):
                row = next(
                    r
                    for r in results["summary"]
                    if r["model"] == model
                    and r["protocol"] == protocol
                    and r["fraction"] == fraction
                )
                vals.append(
                    f"{row['macro_spearman']:.4f} ± {row['seed_sd_spearman']:.4f}"
                    if row["seeds"] > 1
                    else f"{row['macro_spearman']:.4f}"
                )
            lines.append("| " + model + " | " + " | ".join(vals) + " |")
        lines.append("")
    lines += [
        "All models use identical test IDs within each protocol/fraction. G1 has 128 test",
        "variants; G2/G3 each cover all 768. Low-label selection affects original training",
        "rows only. G1/G2/G3 gaps mix novelty, cohort composition and training diversity.",
        "Four supervised held-out families do not establish unseen-pretraining-family transfer.",
        "",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained", sharey=True)
    for axis, protocol in zip(axes, ("G1", "G2", "G3"), strict=True):
        for model in models:
            points = sorted(
                [
                    r
                    for r in results["summary"]
                    if r["model"] == model and r["protocol"] == protocol
                ],
                key=lambda r: r["fraction"],
            )
            axis.errorbar(
                [r["fraction"] for r in points],
                [r["macro_spearman"] for r in points],
                yerr=[r["seed_sd_spearman"] for r in points],
                marker="o",
                capsize=3,
                label=model,
            )
        axis.set_title(protocol)
        axis.set_xticks([0.25, 1.0], ["25%", "100%"])
        axis.set_xlabel("Original training-label fraction")
    axes[0].set_ylabel("Macro protein Spearman (seed SD bars)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.savefig(root / "learning_curves.png", dpi=180)
    plt.close(fig)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmarks/m15"))
    report(parser.parse_args().directory)
