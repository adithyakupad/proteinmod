"""Run M0 and retain measured results with sufficient provenance to reproduce."""

import csv
import importlib.metadata
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, load_dataset, verified_reference, verify_file, write_json
from proteinmod.models import fit_ridge, metrics, one_hot


def run(config):
    start = time.perf_counter()
    dataset, settings = config["dataset"], config["run"]
    path = Path(dataset["path"])
    verify_file(path, dataset)
    reference = verified_reference(config.get("reference"))
    rows, qc = load_dataset(path, reference)
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "qc.json", qc)
    partitions = {
        part: np.array([row["partition"] == part for row in rows])
        for part in ("train", "validation", "test")
    }
    train, validation, test = (partitions[part] for part in ("train", "validation", "test"))
    targets = np.array([row["target"] for row in rows])
    predictions = {"training_mean": np.full(test.sum(), targets[train].mean())}
    selections = {}
    features = {"one_hot_ridge": one_hot([row["sequence"] for row in rows])}
    skipped = {}
    if reference:
        features["mutation_count_ridge"] = np.array([[len(row["mutations"])] for row in rows])
    else:
        skipped["mutation_count_ridge"] = "No provenance-verified WT; mutation sets unavailable"
    for name, matrix in features.items():
        model, selection = fit_ridge(
            matrix[train],
            targets[train],
            matrix[validation],
            targets[validation],
            settings["alphas"],
            settings["seed"],
        )
        # Test labels are first evaluated only after validation selection is complete.
        predictions[name] = model.predict(matrix[test])
        selections[name] = selection
    results = {name: metrics(targets[test], prediction) for name, prediction in predictions.items()}
    write_json(
        output / "metrics.json",
        {
            "test": results,
            "selection": selections,
            "skipped": skipped,
            "undefined_correlations": "null for constant arrays",
        },
    )
    with (output / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "mutations", "measured", *predictions])
        for index, row in enumerate(row for row in rows if row["partition"] == "test"):
            writer.writerow(
                [
                    row["id"],
                    row["sequence"],
                    ";".join(row["mutations"]) if reference else "unavailable",
                    row["target"],
                    *(prediction[index] for prediction in predictions.values()),
                ]
            )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    manifest = {
        "config": config,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": {p.name: checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "dataset_sha256": checksum(path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("proteinmod", "numpy", "scipy", "scikit-learn")
        },
        "git_revision": revision.stdout.strip(),
        "git_dirty": bool(dirty.stdout.strip()),
        "elapsed_seconds": time.perf_counter() - start,
        "selection": "validation RMSE only; no refit on validation; no scaling",
        "artifacts": [
            "qc.json",
            "metrics.json",
            "manifest.json",
            "test_predictions.csv",
            "report.md",
        ],
    }
    write_json(output / "manifest.json", manifest)
    lines = [
        "# ProteinMod M0 — FLIP2 Alpha Amylase",
        "",
        "Single-protein activity assay; this is not cross-protein generalization.",
        "",
        f"Dataset SHA256: `{manifest['dataset_sha256']}`",
        "",
        f"Effective partitions: {qc['counts']}. Lengths: {qc['lengths']}.",
        f"Validation flag / raw set overlap: {qc['validation_flag_set_overlap_counts']}.",
        "Validation-flagged training rows are excluded from fitting. No sequence overlap.",
        f"Unique substitution overlaps: {qc['unique_substitution_overlap_counts']}.",
        (
            "The official by-mutation file is preserved; shared substitutions are reported, "
            "so this run does not claim strictly disjoint substitutions."
        ),
        f"Verified WT: {qc['reference_verified']}; mutation burden: {qc['mutation_burden']}.",
        "",
        "Ridge selected by validation RMSE, fit on training rows only. Fixed alphabet; no scaling.",
        "Constant-prediction correlations are undefined (null), not zero.",
        "",
        "| Model | Spearman rho | Pearson r | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, scores in results.items():
        cells = ["undefined" if value is None else f"{value:.6f}" for value in scores.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Selected alphas: { {name: value['alpha'] for name, value in selections.items()} }.",
        f"Skipped: {skipped}.",
        "",
        f"Source: {dataset['source']}",
        f"Attribution: {dataset['attribution']}",
        "",
        "See metrics.json, qc.json, manifest.json and test_predictions.csv for full evidence.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return results
