"""Reproduce M5 matched-cohort summaries, diagnostics, stratifications and plots."""

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
from proteinmod.hybrid_experiments import output_path
from proteinmod.models import metrics
from proteinmod.multiprotein_data import collection_rows
from proteinmod.structure_pipeline import write_csv


def report(source, output):
    output = output_path({"output": str(output)})
    with (source / "predictions.csv").open() as f:
        new = list(csv.DictReader(f))
    with open("benchmarks/m4/predictions.csv") as f:
        old = [r for r in csv.DictReader(f) if r["cohort"] == "structure_matched"]
    rows = new + old
    runs = json.loads((source / "runs.json").read_text())
    audit = json.loads((source / "audit.json").read_text())
    m4 = json.loads(Path("configs/m4.json").read_text())
    originals, structural_strata, coverage = {}, {}, {}
    for spec in m4["collection"]:
        part, _ = collection_rows(spec, m4["experiment"]["cap_per_partition"])
        originals.update((r["id"], r) for r in part)
        residues = json.loads((Path(spec["structure_artifacts"]) / "residues.json").read_text())
        mapped = {
            r["reference_position"]: r
            for r in residues
            if r["status"] == "mapped" and r["ca"] is not None
        }
        bvalues = [r["bfactor_a2"] for r in mapped.values() if r["bfactor_a2"] is not None]
        threshold = float(np.median(bvalues)) if bvalues else None
        coverage[spec["protein_id"]] = {
            "reference_length": len(residues),
            "mapped_nodes": len(mapped),
            "reference_coverage": len(mapped) / len(residues),
            "reference_bfactor_median_a2": threshold,
            "confidence_kind": "experimental B factors; not calibrated confidence or pLDDT",
            "selected_variants": len(part),
            "excluded_variants": len(audit["data"][spec["protein_id"]]["excluded"]),
        }
        for r in part:
            values = [
                mapped[int(t[1:-1])]["bfactor_a2"] for t in r["mutations"] if int(t[1:-1]) in mapped
            ]
            structural_strata[r["id"]] = (
                "missing_or_wt"
                if not values or any(v is None for v in values) or threshold is None
                else "higher_reference_bfactor"
                if np.mean(values) > threshold
                else "lower_reference_bfactor"
            )
    groups = defaultdict(list)
    for r in rows:
        groups[r["protocol"], r["model"], int(r["seed"]), r["protein_id"]].append(r)
    per_protein, strata, confidence_strata = [], [], []
    g1_roles = next(r for r in audit["roles"] if r["protocol"] == "G1")["ids"]
    g1_ids = set(g1_roles["test"])
    calibration = {}
    for p in {r["protein_id"] for r in originals.values()}:
        calibration[p] = np.array(
            [originals[i]["target"] for i in g1_roles["train"] if originals[i]["protein_id"] == p]
        )
    expected = {
        (protocol, p): {r["id"] for r in records}
        for (protocol, model, seed, p), records in groups.items()
        if model == "esm_sites"
    }
    same_population = []
    for (protocol, model, seed, p), records in sorted(groups.items()):
        if {r["id"] for r in records} != expected[protocol, p]:
            raise ValueError("Comparator IDs do not match")
        measured = np.array([float(r["measured"]) for r in records])
        pred = np.array([float(r["prediction"]) for r in records])
        row = {
            "protocol": protocol,
            "model": model,
            "seed": seed,
            "protein_id": p,
            "n": len(records),
            "spearman_rho": metrics(measured, pred)["spearman_rho"],
        }
        if protocol == "G1":
            train = calibration[p]
            target = np.array(
                [
                    ((train < value).sum() + 0.5 * (train == value).sum()) / len(train)
                    for value in measured
                ]
            )
            row["training_cdf_rmse"] = float(np.sqrt(np.mean((target - pred) ** 2)))
        else:
            row["training_cdf_rmse"] = None  # no supervised test-protein calibration
        per_protein.append(row)
        subset = [r for r in records if r["id"] in g1_ids]
        same_population.append(
            {
                "protocol": protocol,
                "model": model,
                "seed": seed,
                "protein_id": p,
                "n": len(subset),
                "spearman_rho": metrics(
                    np.array([float(r["measured"]) for r in subset]),
                    np.array([float(r["prediction"]) for r in subset]),
                )["spearman_rho"],
            }
        )
        for burden in ("single", "multiple"):
            subset = [
                r
                for r in records
                if (len(originals[r["id"]]["mutations"]) == 1) == (burden == "single")
                and len(originals[r["id"]]["mutations"]) > 0
            ]
            strata.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "seed": seed,
                    "protein_id": p,
                    "mutation_burden": burden,
                    "n": len(subset),
                    "spearman_rho": metrics(
                        np.array([float(r["measured"]) for r in subset]),
                        np.array([float(r["prediction"]) for r in subset]),
                    )["spearman_rho"]
                    if len(subset) >= 2
                    else None,
                }
            )

        for category in ("lower_reference_bfactor", "higher_reference_bfactor", "missing_or_wt"):
            subset = [r for r in records if structural_strata[r["id"]] == category]
            confidence_strata.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "seed": seed,
                    "protein_id": p,
                    "reference_bfactor_stratum": category,
                    "n": len(subset),
                    "spearman_rho": metrics(
                        np.array([float(r["measured"]) for r in subset]),
                        np.array([float(r["prediction"]) for r in subset]),
                    )["spearman_rho"]
                    if len(subset) >= 2
                    else None,
                }
            )
    write_json(output / "coverage.json", coverage)
    write_csv(output / "structure_strata.csv", confidence_strata)

    def aggregate(records):
        grouped = defaultdict(lambda: defaultdict(list))
        for r in records:
            grouped[r["protocol"], r["model"]][r["seed"]].append(r["spearman_rho"] or 0.0)
        return [
            {
                "protocol": protocol,
                "model": model,
                "mean": float(np.mean(values := [np.mean(v) for v in seeds.values()])),
                "seed_sd": float(np.std(values)),
                "n_seeds": len(values),
            }
            for (protocol, model), seeds in sorted(grouped.items())
        ]

    summary = aggregate(per_protein)
    write_json(
        output / "summary.json",
        {
            "macro": summary,
            "same_g1_test_ids": aggregate(same_population),
            "per_protein": per_protein,
        },
    )
    write_csv(output / "mutation_burden.csv", strata)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, protocol in zip(axes, ("G1", "G2"), strict=True):
        selected = [r for r in summary if r["protocol"] == protocol]
        axis.barh(
            [r["model"] for r in selected],
            [r["mean"] for r in selected],
            xerr=[r["seed_sd"] for r in selected],
        )
        axis.axvline(0, color="black", linewidth=0.6)
        axis.set_title(protocol)
        axis.set_xlabel("Macro per-protein Spearman (± seed SD)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, protocol in zip(axes, ("G1", "G2"), strict=True):
        for mode in ("weighted_fusion", "concat_fusion", "bidirectional"):
            selected = [r for r in runs if r["protocol"] == protocol and r["model"] == mode]
            max_epochs = max(len(r["selection"]["trace"]) for r in selected)
            curve = [
                np.mean(
                    [
                        r["selection"]["trace"][i]["validation_macro_spearman"]
                        for r in selected
                        if len(r["selection"]["trace"]) > i
                    ]
                )
                for i in range(max_epochs)
            ]
            axis.plot(np.arange(1, max_epochs + 1), curve, label=mode)
        axis.set_title(protocol + " validation traces (surviving runs)")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation macro Spearman")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "# M5 local benchmark",
        "",
        "Exact M4 structure-matched IDs; higher Spearman is better. Two neural seeds; SD is population SD across seeds, not a confidence interval. Classical M4 comparators have one fit. G1 and G2 use different primary test populations.",
        "",
        "| Model | G1 macro ± SD | G2 macro ± SD |",
        "|---|---:|---:|",
    ]
    for model in sorted({r["model"] for r in summary}):
        values = []
        for protocol in ("G1", "G2"):
            entry = next(
                (r for r in summary if r["protocol"] == protocol and r["model"] == model), None
            )
            values.append(f"{entry['mean']:.4f} ± {entry['seed_sd']:.4f}" if entry else "—")
        lines.append(f"| {model} | {' | '.join(values)} |")
    lines += [
        "",
        "## Per-protein results (mean of seeds)",
        "",
        "| Protocol | Model | Amylase | GB1 | NucB |",
        "|---|---|---:|---:|---:|",
    ]
    for protocol in ("G1", "G2"):
        for model in sorted({r["model"] for r in per_protein}):
            values = []
            for p in ("amylase", "gb1", "nucb"):
                values.append(
                    [
                        r["spearman_rho"] or 0
                        for r in per_protein
                        if r["protocol"] == protocol
                        and r["model"] == model
                        and r["protein_id"] == p
                    ]
                )
            if any(values):
                lines.append(
                    f"| {protocol} | {model} | "
                    + " | ".join(f"{np.mean(v):.4f}" if v else "—" for v in values)
                    + " |"
                )
    lines += [
        "",
        "## Compute and selection",
        "",
        "| Model | Parameters | Total fit seconds | Selected epoch range |",
        "|---|---:|---:|---:|",
    ]
    for mode in sorted({r["model"] for r in runs}):
        selected = [r for r in runs if r["model"] == mode]
        epochs = [r["selection"]["epoch"] for r in selected]
        lines.append(
            f"| {mode} | {selected[0]['selection']['parameters']} | {sum(r['elapsed_seconds'] for r in selected):.2f} | {min(epochs)}–{max(epochs)} |"
        )
    lines += [
        "",
        "![Matched comparison](comparison.png)",
        "",
        "![Validation curves](validation_curves.png)",
        "",
        "See `interpretation.md` for failure analysis, coverage/confidence limitations and reproduction evidence. `summary.json` includes G1 training-CDF RMSE per assay; G2 RMSE is intentionally undefined. `mutation_burden.csv` retains stratum sizes and undefined correlations rather than pooling unlike assays.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(
        output / "report_manifest.json",
        {
            "inputs_sha256": {
                str(p): checksum(p)
                for p in (
                    source / "runs.json",
                    source / "predictions.csv",
                    source / "audit.json",
                    Path("benchmarks/m4/predictions.csv"),
                    Path("configs/m4.json"),
                )
            },
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
    parser.add_argument("--input", type=Path, default=Path("runs/m5"))
    parser.add_argument("--output", type=Path, default=Path("runs/m5"))
    args = parser.parse_args()
    report(args.input, args.output)
