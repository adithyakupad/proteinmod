"""Recompute native-unit stability results from held-out predictions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proteinmod.data import checksum, write_json
from proteinmod.models import metrics
from proteinmod.stability_data import output_directory
from proteinmod.structure_pipeline import write_csv


def report(source, output):
    output = output_directory(output)
    with (source / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    runs = json.loads((source / "runs.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["protocol"], r["model"], int(r["seed"]), r["protein_id"]].append(r)
    expected = {
        (p, protein): {r["id"] for r in records}
        for (p, model, seed, protein), records in grouped.items()
        if model == "training_mean"
    }
    per_protein, same_population, errors = [], [], []
    s1_test = set(next(r for r in audit["roles"] if r["protocol"] == "S1")["ids"]["test"])
    for (protocol, model, seed, protein), records in sorted(grouped.items()):
        if {r["id"] for r in records} != expected[protocol, protein]:
            raise ValueError("Unmatched stability test variants")
        measured = np.array([float(r["measured_ddg_kcal_mol"]) for r in records])
        prediction = np.array([float(r["prediction_ddg_kcal_mol"]) for r in records])
        if not np.isfinite(measured).all() or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite stability observations/predictions")
        row = {
            "protocol": protocol,
            "model": model,
            "seed": seed,
            "protein_id": protein,
            "n": len(records),
            **metrics(measured, prediction),
            "target_min_kcal_mol": float(measured.min()),
            "target_max_kcal_mol": float(measured.max()),
            "mean_error_kcal_mol": float(np.mean(prediction - measured)),
            "pred_min_kcal_mol": float(prediction.min()),
            "pred_max_kcal_mol": float(prediction.max()),
        }
        per_protein.append(row)
        subset = [r for r in records if r["id"] in s1_test]
        same_population.append(
            {
                "protocol": protocol,
                "model": model,
                "seed": seed,
                "protein_id": protein,
                "n": len(subset),
                **metrics(
                    [float(r["measured_ddg_kcal_mol"]) for r in subset],
                    [float(r["prediction_ddg_kcal_mol"]) for r in subset],
                ),
            }
        )
        for j in np.argsort(np.abs(prediction - measured))[-5:][::-1]:
            r = records[j]
            errors.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "seed": seed,
                    "protein_id": protein,
                    "id": r["id"],
                    "mutations": r["mutations"],
                    "measured_ddg_kcal_mol": float(measured[j]),
                    "prediction_ddg_kcal_mol": float(prediction[j]),
                    "absolute_error_kcal_mol": float(abs(prediction[j] - measured[j])),
                }
            )

    def aggregate(records):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in records:
            grouped[r["protocol"], r["model"]][r["seed"]].append(r)
        results = []
        for (protocol, model), seeds in sorted(grouped.items()):
            row = {"protocol": protocol, "model": model, "n_seeds": len(seeds)}
            for key in ("spearman_rho", "pearson_r", "mae", "rmse"):
                values = [
                    np.mean([r[key] if r[key] is not None else 0 for r in records])
                    for records in seeds.values()
                ]
                row[key] = float(np.mean(values))
                row[key + "_seed_sd"] = float(np.std(values))
            results.append(row)
        return results

    summary = aggregate(per_protein)
    write_csv(output / "per_protein.csv", per_protein)
    write_csv(output / "largest_errors.csv", errors)
    write_json(
        output / "summary.json",
        {
            "macro": summary,
            "S2_on_S1_test_ids": aggregate(same_population),
            "units": "kcal/mol; positive ΔΔG destabilizes",
            "undefined_correlations": "retained as null per protein; zero in macros",
            "seed_sd": "population SD; not a confidence interval",
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, protocol in zip(axes, ("S1", "S2"), strict=True):
        selected = [r for r in summary if r["protocol"] == protocol]
        axis.barh(
            [r["model"] for r in selected],
            [r["rmse"] for r in selected],
            xerr=[r["rmse_seed_sd"] for r in selected],
        )
        axis.set_title(protocol)
        axis.set_xlabel("Macro RMSE (kcal/mol; ± seed SD)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    proteins = sorted({r["protein_id"] for r in rows})
    fig, axes = plt.subplots(1, len(proteins), figsize=(13, 4))
    for axis, p in zip(axes, proteins, strict=True):
        selected = [
            r
            for r in rows
            if r["protocol"] == "S2"
            and r["model"] == "stability_head"
            and r["seed"] == "0"
            and r["protein_id"] == p
        ]
        x = np.array([float(r["measured_ddg_kcal_mol"]) for r in selected])
        y = np.array([float(r["prediction_ddg_kcal_mol"]) for r in selected])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        axis.scatter(x, y, s=6, alpha=0.35)
        axis.plot([lo, hi], [lo, hi], color="black", linewidth=0.7)
        axis.set_title(p + " S2 stability head, seed 0")
        axis.set_xlabel("Experimental ΔΔG (kcal/mol)")
        axis.set_ylabel("Predicted ΔΔG (kcal/mol)")
    fig.tight_layout()
    fig.savefig(output / "held_out_scatter.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for r in runs:
        if r["model"] == "stability_head":
            t = r["selection"]["trace"]
            axis.plot(
                [v["epoch"] for v in t],
                [v["validation_macro_rmse_kcal_mol"] for v in t],
                label=f"{r['protocol']}/{r['fold']}/seed{r['seed']}",
            )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation macro RMSE (kcal/mol)")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "# M6: experimental stability benchmark",
        "",
        "Native folding ΔΔG in kcal/mol, positive destabilization. Same test IDs for every model within each protocol. Mean ± population seed SD; two head seeds, one fit for classical baselines. No cross-assay target pooling or rank normalization.",
        "",
        "| Model | S1 RMSE | S2 RMSE | S1 Spearman | S2 Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sorted({r["model"] for r in summary}):
        values = []
        for key in ("rmse", "spearman_rho"):
            for protocol in ("S1", "S2"):
                r = next(
                    (r for r in summary if r["model"] == model and r["protocol"] == protocol), None
                )
                values.append(f"{r[key]:.4f} ± {r[key + '_seed_sd']:.4f}" if r else "—")
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Per-protein stability head (two-seed means)",
        "",
        "| Protocol | Protein | Spearman | Pearson | MAE kcal/mol | RMSE kcal/mol |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for protocol in ("S1", "S2"):
        for p in proteins:
            selected = [
                r
                for r in per_protein
                if r["protocol"] == protocol
                and r["protein_id"] == p
                and r["model"] == "stability_head"
            ]
            lines.append(
                f"| {protocol} | {p} | "
                + " | ".join(
                    f"{np.mean([r[k] if r[k] is not None else 0 for r in selected]):.4f}"
                    for k in ("spearman_rho", "pearson_r", "mae", "rmse")
                )
                + " |"
            )
    lines += [
        "",
        "![Native-unit comparison](comparison.png)",
        "",
        "![Actual held-out predictions](held_out_scatter.png)",
        "",
        "![Validation-only selection traces](validation_curves.png)",
        "",
        "See `interpretation.md` for failure analysis, dataset restrictions and reproducibility. `per_protein.csv` retains every model/seed's four metrics and observed/predicted ranges. `summary.json` also reports S2 restricted to S1 test IDs. Exact reproduction and transfer-checkpoint recovery are documented in `docs/m6_protocol.md`.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(
        output / "report_manifest.json",
        {
            "source_predictions_sha256": checksum(source / "predictions.csv"),
            "reporter_sha256": checksum(__file__),
            "artifacts_sha256": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "report_manifest.json"
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m6"))
    parser.add_argument("--output", default="runs/m6")
    args = parser.parse_args()
    report(args.input, args.output)
