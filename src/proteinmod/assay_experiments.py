"""M9 matched crossed-assay G1/G2 experiments with validation-only selection."""

import argparse
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from proteinmod.assay_data import acquire, collection, folds
from proteinmod.assay_features import load, prepare
from proteinmod.assay_models import evaluate, group_ridge, neural, ridge
from proteinmod.data import checksum, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.multiprotein_data import similarity_audit
from proteinmod.structure_pipeline import write_csv


def history():
    return {
        str(p): checksum(p)
        for p in sorted(Path("benchmarks").rglob("*"))
        if p.is_file()
        and (
            not p.parts[1].startswith("m")
            or not p.parts[1][1:].isdigit()
            or int(p.parts[1][1:]) < 9
        )
    }


def output_path(path):
    output = Path(path).resolve()
    root = Path("benchmarks").resolve()
    allowed = root / "m9"
    if (output == root or root in output.parents or output in root.parents) and not (
        output == allowed or allowed in output.parents
    ):
        raise ValueError("M9 would overwrite historical artifacts")
    for p in Path("benchmarks").iterdir():
        protected = p.resolve()
        if p.name != "m9" and (
            output == protected or output in protected.parents or protected in output.parents
        ):
            raise ValueError("M9 would overwrite historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run(config):
    before = history()
    started = time.perf_counter()
    output = output_path(config["output"])
    rows, audit = collection(config)
    features, feature_manifest = load(config)
    if any(len(x) != len(rows) for x in features.values()):
        raise ValueError("Feature observation count differs")
    metadata = {a["metadata"]["assay_id"]: a["metadata"] for a in config["assays"]}
    m4 = json.loads(Path("configs/m4.json").read_text())
    similarity = similarity_audit(config["proteins"] + m4["collection"])
    new = {p["protein_id"] for p in config["proteins"]}
    supervised = set(
        config["pretrained"]["function_training"] + config["pretrained"]["function_validation"]
    )
    for p in similarity:
        if (
            (
                (p["left"] in new and p["right"] in supervised)
                or (p["right"] in new and p["left"] in supervised)
            )
            and p["aligned_identity"] >= 0.3
            and min(p["left_coverage"], p["right_coverage"]) >= 0.8
        ):
            raise ValueError("New protein overlaps supervised pretraining reference")
    predictions, records, roles_audit = [], [], []
    identity = {
        "config": config,
        "features": feature_manifest["feature_sha256"],
        "source": {str(p): checksum(p) for p in Path("src/proteinmod").glob("assay*.py")},
    }
    cache = Path(config["cache"]) / "heads" / fingerprint(identity)
    cache.mkdir(parents=True, exist_ok=True)

    def record(protocol, fold, model, seed, roles, pred, selected, elapsed):
        if not np.isfinite(pred).all() or (pred < 0).any() or (pred > 1).any():
            raise ValueError("Invalid bounded function score")
        records.append(
            {
                "protocol": protocol,
                "fold": fold,
                "model": model,
                "seed": seed,
                "selection": selected,
                "test": evaluate(rows, roles["test"], pred[roles["test"]]),
                "elapsed_seconds": elapsed,
            }
        )
        predictions.extend(
            {
                "protocol": protocol,
                "fold": fold,
                "model": model,
                "seed": seed,
                "id": rows[i]["id"],
                "protein_id": rows[i]["protein_id"],
                "assay_id": rows[i]["assay_id"],
                "family": rows[i]["family"],
                "measured_native_score": rows[i]["target"],
                "predicted_train_percentile": float(pred[i]),
            }
            for i in roles["test"]
        )

    for protocol, fold, roles in folds(rows, config["protein_order"]):
        roles_audit.append(
            {
                "protocol": protocol,
                "fold": fold,
                "ids": {k: [rows[i]["id"] for i in v] for k, v in roles.items()},
                "families": {k: sorted({rows[i]["family"] for i in v}) for k, v in roles.items()},
            }
        )
        for key, matrix in features.items():
            start = time.perf_counter()
            pred, selection, _, _ = ridge(
                matrix, rows, roles, config["alphas"], config["training"]["scale_floor"]
            )
            record(
                protocol,
                fold,
                key + "_ridge",
                0,
                roles,
                pred,
                selection,
                time.perf_counter() - start,
            )
        for key in ("family", "assay_id"):
            start = time.perf_counter()
            pred, selection = group_ridge(
                features["multimodal"],
                rows,
                roles,
                config["alphas"],
                key,
                config["training"]["scale_floor"],
            )
            record(
                protocol,
                fold,
                key + "_heads",
                0,
                roles,
                pred,
                selection,
                time.perf_counter() - start,
            )
        for seed in config["seeds"]:
            parameters = []
            for mode in config["modes"]:
                start = time.perf_counter()
                pred, selection, state, missing = neural(
                    features["multimodal"], rows, roles, metadata, config["training"], seed, mode
                )
                state.update(
                    {
                        "feature_sha256": feature_manifest["feature_sha256"],
                        "config": config,
                        "protocol": protocol,
                        "fold": fold,
                        "seed": seed,
                    }
                )
                path = cache / f"{protocol}-{fold}-{mode}-{seed}.pt"
                torch.save(state, path)
                selection["checkpoint_sha256"] = checksum(path)
                record(
                    protocol, fold, mode, seed, roles, pred, selection, time.perf_counter() - start
                )
                parameters.append(selection["parameters"])
                if mode == "structured":
                    record(
                        protocol,
                        fold,
                        "structured_missing_at_test",
                        seed,
                        roles,
                        missing,
                        {**selection, "inference_only_stress": True},
                        0.0,
                    )
            if len(set(parameters)) != 1:
                raise ValueError("Conditioning controls differ in registered parameter capacity")
            print(protocol, fold, seed, "complete", flush=True)
        write_json(output / "runs.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_json(output / "rows.json", rows)
    legacy = json.loads(Path("configs/m4.json").read_text())["collection"]
    write_json(
        output / "audit.json",
        {
            "data": audit,
            "roles": roles_audit,
            "features": feature_manifest,
            "pretraining_sequence_similarity": similarity,
            "historical_sha256": before,
            "legacy_non_identifiability": {
                "proteins": len(legacy),
                "assay_ids": len({x["assay_id"] for x in legacy}),
                "one_assay_per_protein": True,
                "assay_id_transfer_identifiable": False,
            },
            "native_endpoints_not_pooled": True,
            "stability_scalar_inputs": False,
            "novel_assay_family_holdout": "not run: two families insufficient for three disjoint family roles",
        },
    )
    if history() != before:
        raise ValueError("Historical artifacts changed")
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "config_sha256": fingerprint(config),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "sources": identity["source"],
            "device": "cpu",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "scikit-learn", "biopython")
            },
            "checkpoint_root": str(cache),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "files": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["download", "prepare", "run"])
    parser.add_argument("--config", default="configs/m9.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.action == "download":
        print(acquire(config))
    elif args.action == "prepare":
        print(prepare(config))
    else:
        run(config)


if __name__ == "__main__":
    main()
