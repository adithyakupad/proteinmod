"""Matched M3 protocol: label-free eligibility, train-only fits, validation-only selection."""

import importlib.metadata
import json
import platform
import resource
import subprocess
import time
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from proteinmod.data import checksum, load_dataset, verified_reference, verify_file, write_json
from proteinmod.embeddings import cache_location, fingerprint, load_features
from proteinmod.experiments import nested_subsets, summarize
from proteinmod.models import fit_ridge, metrics, one_hot
from proteinmod.structural_graph import load_graph, prepare_inputs
from proteinmod.structure_pipeline import write_csv


def history_hashes():
    return {
        str(p): checksum(p)
        for directory in ("benchmarks/amylase", "benchmarks/amylase_m1", "benchmarks/structures")
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


def output_directory(settings):
    output = Path(settings["output"]).resolve()
    for directory in ("benchmarks/amylase", "benchmarks/amylase_m1", "benchmarks/structures"):
        protected = Path(directory).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("M3 output must not overlap historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare(config):
    settings = config["m3"]
    with open(settings["sequence_config"], "rb") as handle:
        sequence = tomllib.load(handle)
    verify_file(sequence["dataset"]["path"], sequence["dataset"])
    reference = verified_reference(sequence["reference"])
    rows, qc = load_dataset(sequence["dataset"]["path"], reference)
    graph = load_graph(settings["structure_artifacts"], reference)
    source_sha = checksum(sequence["dataset"]["path"])
    contract = {
        "dataset_sha256": source_sha,
        "graph": graph["audit"]["m2_manifest_sha256"],
        "source": checksum(Path(__file__).with_name("structural_graph.py")),
        "sequences": fingerprint([r["sequence"] for r in rows]),
    }
    cache = Path(settings["cache"]) / fingerprint(contract)
    cache.mkdir(parents=True, exist_ok=True)
    archive, metadata = cache / "features.npz", cache / "manifest.json"
    if archive.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["contract"] != contract or checksum(archive) != saved["sha256"]:
            raise ValueError("M3 feature cache checksum/contract mismatch")
        with np.load(archive, allow_pickle=False) as arrays:
            valid, dynamic, structural = (arrays[k] for k in ("valid", "dynamic", "structural"))
        exclusions = saved["exclusions"]
    else:
        valid, dynamic, structural, exclusions = prepare_inputs(rows, reference, graph)
        np.savez_compressed(archive, valid=valid, dynamic=dynamic, structural=structural)
        write_json(
            metadata, {"contract": contract, "sha256": checksum(archive), "exclusions": exclusions}
        )
    if dynamic.shape != (len(valid), len(graph["nodes"]), 25) or structural.shape[0] != len(valid):
        raise ValueError("M3 feature cache shape mismatch")
    if not np.isfinite(dynamic).all() or not np.isfinite(structural).all():
        raise ValueError("Nonfinite M3 feature cache")
    directory, identity = cache_location(
        sequence, [r["sequence"] for r in rows], reference, source_sha
    )
    embedded, embedding_metadata = load_features(
        directory, identity, len(rows), sequence["embedding"]["dimension"]
    )
    selected = [rows[i] for i in valid]
    partitions = {
        part: np.array([i for i, r in enumerate(selected) if r["partition"] == part], dtype=int)
        for part in ("train", "validation", "test")
    }
    if any(len(indices) < 2 for indices in partitions.values()):
        raise ValueError("Insufficient valid rows in a partition")
    features = {
        "one_hot_ridge": one_hot([r["sequence"] for r in selected]),
        "esm_sites": embedded["esm_sites"][valid].astype(np.float64),
        "structural_ridge": structural.astype(np.float64),
    }
    audit = {
        "graph": graph["audit"],
        "original_qc": qc,
        "original_counts": dict(Counter(r["partition"] for r in rows)),
        "valid_counts": {p: len(ix) for p, ix in partitions.items()},
        "excluded_counts": dict(Counter(r["partition"] for r in exclusions)),
        "exclusions": exclusions,
        "exclusion_policy": "Any mutation outside identity-compatible C-alpha nodes; label independent; fixed before fitting",
        "valid_partition_ids": {p: [selected[i]["id"] for i in ix] for p, ix in partitions.items()},
        "dataset_sha256": source_sha,
        "embedding_manifest_sha256": checksum(directory / "manifest.json"),
        "embedding_label_free": embedding_metadata["label_free"],
        "feature_cache": str(cache),
        "feature_cache_bytes": archive.stat().st_size,
        "leakage": {
            "duplicate_sequences": len(selected) - len({r["sequence"] for r in selected}),
            "train_test_shared_substitutions": sorted(
                {t for i in partitions["train"] for t in selected[i]["mutations"]}
                & {t for i in partitions["test"] for t in selected[i]["mutations"]}
            ),
            "selection": "Only validation RMSE; no test-based stopping or redesign",
            "preprocessing": "Structural ridge StandardScaler and GNN target mean/std fit on training subset only; M0/M1 unchanged unscaled probes",
            "structure_selection": "Preselected experimental 1UA7 M2 context; no comparison/selection of structures using fitness",
            "scope": "Single-protein transductive reference graph; not cross-protein generalization",
        },
    }
    return selected, partitions, features, dynamic, graph, audit


def baselines(config, rows, partitions, features):
    settings = config["m3"]
    targets = np.array([r["target"] for r in rows])
    records, predictions, subsets = [], [], {}
    for seed in settings["seeds"]:
        subsets[str(seed)] = {}
        for fraction, subset in nested_subsets(
            partitions["train"], settings["fractions"], seed
        ).items():
            subsets[str(seed)][str(fraction)] = [rows[i]["id"] for i in subset]
            for name, matrix in features.items():
                started = time.perf_counter()
                if name == "structural_ridge":
                    scaler = StandardScaler().fit(matrix[subset])
                    matrix = scaler.transform(matrix)
                model, selection = fit_ridge(
                    matrix[subset],
                    targets[subset],
                    matrix[partitions["validation"]],
                    targets[partitions["validation"]],
                    settings["alphas"],
                    seed,
                )
                pred = model.predict(matrix[partitions["test"]])
                record = {
                    "model": name,
                    "seed": seed,
                    "fraction": fraction,
                    "n_train": len(subset),
                    "train": metrics(targets[subset], model.predict(matrix[subset])),
                    "validation": metrics(
                        targets[partitions["validation"]],
                        model.predict(matrix[partitions["validation"]]),
                    ),
                    "test": metrics(targets[partitions["test"]], pred),
                    "selection": selection,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                records.append(record)
                predictions.extend(
                    {
                        "id": rows[i]["id"],
                        "measured": targets[i],
                        "model": name,
                        "seed": seed,
                        "fraction": fraction,
                        "prediction": float(value),
                    }
                    for i, value in zip(partitions["test"], pred, strict=True)
                )
            print(f"Ridge seed={seed} fraction={fraction} complete", flush=True)
    return records, predictions, subsets


def run_structural(config, baseline_only=False):
    started = time.perf_counter()
    before = history_hashes()
    output = output_directory(config["m3"])
    settings = config["m3"]
    if len(settings["seeds"]) < 2 or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("M3 requires at least two distinct seeds")
    rows, partitions, features, dynamic, graph, audit = prepare(config)
    write_json(output / "audit.json", audit)
    records, predictions, subsets = baselines(config, rows, partitions, features)
    write_json(output / "baseline_runs.json", records)
    write_csv(output / "baseline_predictions.csv", predictions)
    write_json(output / "subset_ids.json", subsets)
    if not baseline_only:
        from proteinmod.structural_gnn import train_gnn

        targets = np.array([r["target"] for r in rows])
        reproducibility = None
        for seed in config["m3"]["seeds"]:
            for fraction, subset in nested_subsets(
                partitions["train"], config["m3"]["fractions"], seed
            ).items():
                for layers in config["gnn"]["layers"]:
                    name = "structural_mlp" if layers == 0 else "structural_gnn"
                    result, pred, state = train_gnn(
                        graph,
                        dynamic,
                        targets,
                        subset,
                        partitions["validation"],
                        partitions["test"],
                        config["gnn"],
                        seed,
                        layers,
                    )
                    record = dict(model=name, fraction=fraction, n_train=len(subset), **result)
                    records.append(record)
                    predictions.extend(
                        {
                            "id": rows[i]["id"],
                            "measured": targets[i],
                            "model": name,
                            "seed": seed,
                            "fraction": fraction,
                            "prediction": float(value),
                        }
                        for i, value in zip(partitions["test"], pred, strict=True)
                    )
                    checkpoint_dir = (
                        Path(config["m3"]["cache"]) / "checkpoints" / fingerprint(config)
                    )
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    import torch

                    checkpoint = checkpoint_dir / f"{name}-{seed}-{fraction}.pt"
                    torch.save(state, checkpoint)
                    record["checkpoint_sha256"] = checksum(checkpoint)
                    record["checkpoint_path"] = str(checkpoint)
                    if reproducibility is None and layers > 0:
                        repeated, again, _ = train_gnn(
                            graph,
                            dynamic,
                            targets,
                            subset,
                            partitions["validation"],
                            partitions["test"],
                            config["gnn"],
                            seed,
                            layers,
                        )
                        difference = float(np.max(np.abs(pred - again)))
                        if difference > 1e-7:
                            raise ValueError("Same-device seeded GNN reproduction failed")
                        reproducibility = {
                            "seed": seed,
                            "fraction": fraction,
                            "max_prediction_difference": difference,
                            "repeat_seconds": repeated["elapsed_seconds"],
                        }
                    print(f"{name} seed={seed} fraction={fraction} complete", flush=True)
                    write_json(output / "runs.json", records)
        audit["gnn_reproducibility"] = reproducibility
    if before != history_hashes():
        raise ValueError("Historical artifacts changed")
    write_json(output / "runs.json", records)
    write_csv(output / "test_predictions.csv", predictions)
    write_csv(output / "learning_curve_summary.csv", summarize(records))
    audit["historical_sha256"] = before
    write_json(output / "audit.json", audit)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "baseline_only": baseline_only,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_revision": revision,
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
                ).stdout.strip()
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "source_sha256": {
                p.name: checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "biopython")
            }
            | ({"torch": importlib.metadata.version("torch")} if not baseline_only else {}),
            "historical_files_unchanged": len(before),
            "artifacts_sha256": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return summarize(records)
