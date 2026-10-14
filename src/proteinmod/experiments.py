"""Matched M0/M1 ridge comparisons and nested low-label learning curves."""

import csv
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, load_dataset, verified_reference, verify_file, write_json
from proteinmod.embeddings import cache_location, fingerprint, load_features
from proteinmod.models import fit_ridge, metrics, one_hot


def nested_subsets(train_indices, fractions, seed):
    if len(train_indices) < 2 or not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError("Need at least two training rows and fractions in (0, 1]")
    if len(set(fractions)) != len(fractions) or 1.0 not in fractions:
        raise ValueError("Fractions must be unique and include 100%")
    order = np.random.default_rng(seed).permutation(train_indices)
    return {
        fraction: np.sort(order[: max(2, math.ceil(fraction * len(order)))])
        for fraction in sorted(fractions)
    }


def predict_subset(features, targets, subset, validation, test, alphas, seed):
    predictions = {"training_mean": np.full(len(test), targets[subset].mean())}
    selections = {}
    for name, matrix in features.items():
        model, selection = fit_ridge(
            matrix[subset], targets[subset], matrix[validation], targets[validation], alphas, seed
        )
        predictions[name] = model.predict(matrix[test])
        selections[name] = selection
    return predictions, selections


def summarize(records):
    summaries = []
    for model in sorted({row["model"] for row in records}):
        for fraction in sorted({row["fraction"] for row in records}):
            group = [
                row for row in records if row["model"] == model and row["fraction"] == fraction
            ]
            entry = {
                "model": model,
                "fraction": fraction,
                "n_train": group[0]["n_train"],
                "n_seeds": len(group),
            }
            for metric in ("spearman_rho", "pearson_r", "mae", "rmse"):
                values = [row["test"][metric] for row in group if row["test"][metric] is not None]
                entry[metric + "_mean"] = float(np.mean(values)) if values else None
                entry[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summaries.append(entry)
    return summaries


def verify_history(directory, dataset_sha, rows, full_scores, predictions):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["dataset_sha256"] != dataset_sha:
        raise ValueError("M0 dataset SHA differs; refusing unmatched comparison")
    historical = json.loads((directory / "metrics.json").read_text())["test"]
    for model, scores in historical.items():
        for metric, expected in scores.items():
            actual = full_scores[model][metric]
            if expected is None:
                if actual is not None:
                    raise ValueError("M0 undefined metric changed")
            elif not np.isclose(actual, expected, atol=1e-9, rtol=1e-7):
                raise ValueError(f"M0 reproduction failed: {model}/{metric}")
    with (directory / "test_predictions.csv").open() as handle:
        old_rows = list(csv.DictReader(handle))
    current = [row for row in rows if row["partition"] == "test"]
    if len(old_rows) != len(current):
        raise ValueError("M0 test row count mismatch")
    differences = {}
    for old, row in zip(old_rows, current, strict=True):
        if (
            old["id"] != row["id"]
            or old["sequence"] != row["sequence"]
            or float(old["measured"]) != row["target"]
        ):
            raise ValueError("M0 held-out row identity/label mismatch")
    for model in historical:
        previous = np.array([float(row[model]) for row in old_rows])
        differences[model] = float(np.max(np.abs(previous - predictions[model])))
        if not np.allclose(previous, predictions[model], atol=1e-9, rtol=1e-7):
            raise ValueError(f"M0 prediction reproduction failed: {model}")
    return {
        "passed": True,
        "max_absolute_prediction_difference": differences,
        "historical_sha256": {
            p.name: checksum(p) for p in sorted(directory.iterdir()) if p.is_file()
        },
    }


def plot_curves(summaries, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for model in sorted({row["model"] for row in summaries}):
        group = [row for row in summaries if row["model"] == model]
        for ax, metric, label in zip(
            axes, ("spearman_rho", "rmse"), ("Test Spearman rho", "Test RMSE"), strict=True
        ):
            if all(row[metric + "_mean"] is None for row in group):
                continue
            x = [row["n_train"] for row in group]
            y = [row[metric + "_mean"] for row in group]
            error = [row[metric + "_std"] or 0 for row in group]
            ax.errorbar(x, y, yerr=error, marker="o", capsize=3, label=model)
            ax.set_xscale("log")
            ax.set_xlabel("Labeled training variants (log scale)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        f"FLIP2 Amylase: mean ± SD across {summaries[0]['n_seeds']} nested-subset seeds\nFixed validation/test; bars are not test uncertainty; mean/delta overlap",
        fontsize=11,
    )
    fig.savefig(output / "learning_curves.png", dpi=180)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)


def write_report(output, full_scores, summaries, history, embedding_metadata, qc):
    lines = [
        "# M1: frozen ESM-2 on FLIP2 Alpha Amylase",
        "",
        f"Exact M0 dataset and split: {qc['counts']}; lengths: {qc['lengths']}.",
        "Single protein, 410 shared train/test substitutions; no cross-protein generalization.",
        "Test-sequence extraction is label-free transductive evaluation, not test-label training.",
        "",
        f"Model revision: `{embedding_metadata['contract']['settings']['revision']}`.",
        f"Extraction device: {embedding_metadata['encoder']['device']}.",
        (
            f"M0 reproduction passed: {history['passed']}; maximum prediction differences: "
            f"{history['max_absolute_prediction_difference']}."
        ),
        "",
        "| Model | Spearman rho | Pearson r | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, scores in full_scores.items():
        cells = ["undefined" if value is None else f"{value:.6f}" for value in scores.values()]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Low-label results",
        "",
        f"Each cell is mean ± sample SD across {summaries[0]['n_seeds']} deterministic nested-subset seeds.",
        "Validation/test never change. This measures training subset variation, not test uncertainty.",
        "100% subsets coincide, so seed variance there is zero. Alphas selected by validation RMSE.",
        "",
        "| Model | Fraction | Train n | Spearman rho | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        cells = []
        for metric in ("spearman_rho", "rmse"):
            value, deviation = row[metric + "_mean"], row[metric + "_std"]
            cells.append("undefined" if value is None else f"{value:.6f} ± {deviation or 0:.6f}")
        lines.append(
            f"| {row['model']} | {row['fraction']:.0%} | {row['n_train']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "## Representation interpretation",
        "",
        (
            "Mean pooling averages all residue hidden states. Delta subtracts a constant WT mean; "
            "with a fitted intercept and identical ridge penalty these are mathematically equivalent "
            "up to numerical precision on this single-WT assay. Site features concatenate the mean "
            "mutant state at changed sites, mean mutant-minus-WT state at those sites, and site count. "
            "This symmetric aggregation handles multiple mutations; WT uses zeros. It can retain "
            "local mutation signal lost by whole-sequence averaging, but does not explicitly learn epistasis."
        ),
        "",
        (
            "Pretraining may reduce the number of labeled examples needed by supplying biochemical "
            "structure. It can also discard assay-specific effects; no improvement is assumed. "
            "These results use one architecture and a predeclared grid, without test-driven redesign."
        ),
        "",
        (
            "No GNNs, structural integration, cross-attention or fine-tuning. Frozen float32 encoder; "
            "ridge only is fitted. Fixed alphabet for one-hot, no feature scaling for any representation. "
            "See selections in learning_curve_runs.json and the source/config/environment manifest."
        ),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_experiments(config):
    started = time.perf_counter()
    path = Path(config["dataset"]["path"])
    verify_file(path, config["dataset"])
    reference = verified_reference(config.get("reference"))
    if reference is None:
        raise ValueError("M1 requires verified WT for matched mutation features")
    rows, qc = load_dataset(path, reference)
    sequences = [row["sequence"] for row in rows]
    dataset_sha = checksum(path)
    directory, identity = cache_location(config, sequences, reference, dataset_sha)
    embedded, embedding_metadata = load_features(
        directory, identity, len(rows), config["embedding"]["dimension"]
    )
    features = {
        "one_hot_ridge": one_hot(sequences),
        "mutation_count_ridge": np.array(
            [[len(row["mutations"])] for row in rows], dtype=np.float64
        ),
        **{name: matrix.astype(np.float64) for name, matrix in embedded.items()},
    }
    partitions = {
        part: np.array([i for i, row in enumerate(rows) if row["partition"] == part])
        for part in ("train", "validation", "test")
    }
    targets = np.array([row["target"] for row in rows])
    settings = config["learning_curve"]
    seeds = settings["seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Learning curves require at least two distinct seeds")
    output = Path(settings["output"])
    historical = Path(settings["historical"])
    if output.resolve() == historical.resolve() or historical.resolve() in output.resolve().parents:
        raise ValueError("M1 output must not overwrite historical M0 artifacts")
    output.mkdir(parents=True, exist_ok=True)
    records, subset_ids = [], {}
    full_predictions, full_selection = predict_subset(
        features,
        targets,
        partitions["train"],
        partitions["validation"],
        partitions["test"],
        config["run"]["alphas"],
        seeds[0],
    )
    full_scores = {
        model: metrics(targets[partitions["test"]], prediction)
        for model, prediction in full_predictions.items()
    }
    history = verify_history(historical, dataset_sha, rows, full_scores, full_predictions)
    for seed in seeds:
        subset_ids[str(seed)] = {}
        for fraction, subset in nested_subsets(
            partitions["train"], settings["fractions"], seed
        ).items():
            subset_ids[str(seed)][str(fraction)] = [rows[i]["id"] for i in subset]
            if fraction == 1:
                predictions, selections = full_predictions, full_selection
            else:
                predictions, selections = predict_subset(
                    features,
                    targets,
                    subset,
                    partitions["validation"],
                    partitions["test"],
                    config["run"]["alphas"],
                    seed,
                )
            for model, prediction in predictions.items():
                records.append(
                    {
                        "model": model,
                        "seed": seed,
                        "fraction": fraction,
                        "n_train": len(subset),
                        "test": metrics(targets[partitions["test"]], prediction),
                        "selection": selections.get(model),
                    }
                )
            print(f"Evaluated seed={seed} fraction={fraction} n={len(subset)}", flush=True)
    summaries = summarize(records)
    write_json(
        output / "metrics.json",
        {"test": full_scores, "selection": full_selection, "m0_reproduction": history},
    )
    write_json(output / "learning_curve_runs.json", records)
    write_json(output / "subset_ids.json", subset_ids)
    write_json(output / "qc.json", qc)
    with (output / "learning_curve_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "measured", *full_predictions])
        for index, row in enumerate(rows[i] for i in partitions["test"]):
            writer.writerow(
                [
                    row["id"],
                    row["sequence"],
                    row["target"],
                    *(prediction[index] for prediction in full_predictions.values()),
                ]
            )
    plot_curves(summaries, output)
    write_report(output, full_scores, summaries, history, embedding_metadata, qc)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "dataset_sha256": dataset_sha,
            "embedding": embedding_metadata,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": revision.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip()),
            "elapsed_seconds": time.perf_counter() - started,
            "source_sha256": {
                p.name: checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("proteinmod", "numpy", "scipy", "scikit-learn", "matplotlib")
            },
            "partition_ids_sha256": {
                part: fingerprint([rows[i]["id"] for i in indices])
                for part, indices in partitions.items()
            },
            "protocol": "Nested seeded train subsets; immutable validation/test; validation-only RMSE selection; no scaling/refit",
            "uncertainty": "Sample SD across train subset seeds; not test uncertainty",
            "artifacts_sha256": {
                p.name: checksum(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return full_scores
