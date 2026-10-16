"""Per-protein/family reporting without pooling incomparable assay measurements."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from proteinmod.data import write_json
from proteinmod.models import metrics
from proteinmod.structure_pipeline import write_csv


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in ("cohort", "protocol", "model", "seed", "protein_id"))].append(
            r
        )
    proteins, families, summary, slices = [], [], [], []
    for key, values in sorted(grouped.items()):
        cohort, protocol, model, seed, protein = key
        score = metrics(
            [float(r["target"]) for r in values], [float(r["prediction"]) for r in values]
        )
        if cohort == "family":
            score = {k: v if k == "spearman_rho" else None for k, v in score.items()}
        proteins.append(
            dict(zip(("cohort", "protocol", "model", "seed", "protein_id"), key, strict=True))
            | {"family": values[0]["family"], "n": len(values), **score}
        )
        for label, selected in [
            ("single", [r for r in values if int(r["n_mutations"]) == 1]),
            ("multiple", [r for r in values if int(r["n_mutations"]) > 1]),
        ]:
            if len(selected) >= 3:
                s = metrics(
                    [float(r["target"]) for r in selected],
                    [float(r["prediction"]) for r in selected],
                )
                slices.append(
                    {
                        "cohort": cohort,
                        "protocol": protocol,
                        "model": model,
                        "seed": seed,
                        "protein_id": protein,
                        "slice": label,
                        "n": len(selected),
                        "spearman_rho": s["spearman_rho"],
                        "rmse": s["rmse"] if cohort == "native" else None,
                    }
                )
    fg = defaultdict(list)
    for r in proteins:
        fg[tuple(r[k] for k in ("cohort", "protocol", "model", "seed", "family"))].append(r)
    for key, values in sorted(fg.items()):
        family = dict(zip(("cohort", "protocol", "model", "seed", "family"), key, strict=True))
        families.append(
            family
            | {
                "n": sum(r["n"] for r in values),
                "spearman_rho": float(np.mean([r["spearman_rho"] or 0 for r in values])),
                "rmse": float(np.mean([r["rmse"] for r in values])) if key[0] == "native" else None,
            }
        )
    combinations = sorted({tuple(r[k] for k in ("cohort", "protocol", "model")) for r in proteins})
    for cohort, protocol, model in combinations:
        selected = [
            r
            for r in proteins
            if (r["cohort"], r["protocol"], r["model"]) == (cohort, protocol, model)
        ]
        seeds = sorted({r["seed"] for r in selected})
        rhos = [
            np.mean([r["spearman_rho"] or 0 for r in selected if r["seed"] == seed])
            for seed in seeds
        ]
        family_scores = [
            np.mean(
                [
                    r["spearman_rho"]
                    for r in families
                    if (r["cohort"], r["protocol"], r["model"], r["family"])
                    == (cohort, protocol, model, f)
                ]
            )
            for f in sorted({r["family"] for r in selected})
        ]
        result = {
            "cohort": cohort,
            "protocol": protocol,
            "model": model,
            "seeds": len(seeds),
            "n_per_seed": sum(r["n"] for r in selected if r["seed"] == seeds[0]),
            "macro_spearman": float(np.mean(rhos)),
            "seed_sd_spearman": float(np.std(rhos)),
            "family_sd_spearman": float(np.std(family_scores)) if cohort == "family" else None,
        }
        if cohort == "native":
            rmse = [np.mean([r["rmse"] for r in selected if r["seed"] == seed]) for seed in seeds]
            result.update(
                {"macro_rmse_kcal_mol": float(np.mean(rmse)), "seed_sd_rmse": float(np.std(rmse))}
            )
        summary.append(result)
    return {
        "summary": summary,
        "per_protein": proteins,
        "per_family": families,
        "failure_slices": slices,
    }


def report(directory):
    with (directory / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    result = summarize(rows)
    write_json(directory / "summary.json", result["summary"])
    for name in ("per_protein", "per_family", "failure_slices"):
        write_csv(directory / (name + ".csv"), result[name])
    runs = json.loads((directory / "metrics.json").read_text())
    write_csv(
        directory / "overfitting.csv",
        [
            {k: r[k] for k in ("cohort", "protocol", "fold", "model", "seed")}
            | {
                k: r["selection"][k]
                for k in ("epoch", "train_score", "validation_score", "parameters")
            }
            for r in runs
            if "epoch" in r["selection"]
        ],
    )
    evidence = json.loads((directory / "audit.json").read_text())
    shifts = []
    graph_data = evidence["provenance"]["graphs"]
    for fold in evidence["roles"]:
        cohort = fold["cohort"]
        by_id = {r["id"]: r["protein_id"] for r in rows if r["cohort"] == cohort}
        means = {}
        for role, ids in fold["roles"].items():
            proteins = {by_id[identifier] for identifier in ids}
            means[role] = np.mean(
                [graph_data[cohort + ":" + p]["feature_mean"] for p in sorted(proteins)], axis=0
            )
        shifts.append(
            {k: fold[k] for k in ("cohort", "protocol", "fold")}
            | {
                "edge_features": next(iter(graph_data.values()))["edge_feature_names"],
                "protein_macro_feature_means": {k: v.tolist() for k, v in means.items()},
                "test_minus_train": (means["test"] - means["train"]).tolist(),
            }
        )
    write_json(directory / "feature_shifts.json", shifts)
    costs = []
    for model in sorted({r["model"] for r in runs}):
        selected = [r for r in runs if r["model"] == model]
        costs.append(
            {
                "model": model,
                "fits": len(selected),
                "total_seconds": sum(r["seconds"] for r in selected),
                "median_seconds": float(np.median([r["seconds"] for r in selected])),
                "trainable_parameters": sorted(
                    {
                        r["selection"]["parameters"]
                        for r in selected
                        if "parameters" in r["selection"]
                    }
                ),
                "median_selected_epoch": float(
                    np.median(
                        [r["selection"]["epoch"] for r in selected if "epoch" in r["selection"]]
                    )
                )
                if "epoch" in selected[0]["selection"]
                else None,
            }
        )
    write_json(directory / "costs.json", costs)
    lines = [
        "# M14 actual local results",
        "",
        "Family cohort: macro per-protein Spearman (two proteins per family, equal family weight).",
        "Native cohort: macro per-protein RMSE in kcal/mol, positive destabilizing.",
        "Mean ± population SD across three fixed seeds; ridges deterministic (one fit).",
        "",
        "| Model | G1 rho | G2 rho | G3 rho | S1 RMSE | S2 RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in json.loads(Path("configs/m14.json").read_text())["models"]:
        values = []
        for protocol in ("G1", "G2", "G3", "S1", "S2"):
            r = next(
                r for r in result["summary"] if r["protocol"] == protocol and r["model"] == model
            )
            key, sd = (
                ("macro_rmse_kcal_mol", "seed_sd_rmse")
                if protocol.startswith("S")
                else ("macro_spearman", "seed_sd_spearman")
            )
            values.append(f"{r[key]:.4f} ± {r[sd]:.4f}" if r["seeds"] > 1 else f"{r[key]:.4f}")
        lines.append("| " + model + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "All models share exact test IDs within each protocol. G1 has 128 primary and",
        "270 native test variants; G2/G3 have 768 primary variants; S2 has 1,800 native.",
        "G1-to-G2/G3 gaps include the change from a within-protein test subset to all variants.",
        "No cross-assay RMSE on rank predictions, no cross-family-pretraining claim.",
        "See per_family.csv, per_protein.csv, failure_slices.csv, overfitting.csv and costs.json.",
        "",
    ]
    (directory / "RESULTS.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = json.loads(Path("configs/m14.json").read_text())["models"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for axis, protocols, metric, ylabel in [
        (axes[0], ("G1", "G2", "G3"), "macro_spearman", "Macro protein Spearman"),
        (axes[1], ("S1", "S2"), "macro_rmse_kcal_mol", "Macro protein RMSE (kcal/mol)"),
    ]:
        for j, protocol in enumerate(protocols):
            vals = [
                next(
                    r[metric]
                    for r in result["summary"]
                    if r["protocol"] == protocol and r["model"] == m
                )
                for m in models
            ]
            axis.bar(
                np.arange(len(models)) + (j - (len(protocols) - 1) / 2) * 0.24,
                vals,
                width=0.24,
                label=protocol,
            )
        axis.set_xticks(
            range(len(models)),
            [m.replace("edge_", "").replace("_", "\n") for m in models],
            fontsize=8,
        )
        axis.set_ylabel(ylabel)
        axis.legend()
    fig.savefig(directory / "results.png", dpi=180)
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmarks/m14"))
    report(parser.parse_args().directory)
