"""M8: stability-source validation separated from identical G1/G2 function evaluation."""

import argparse
import csv
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from proteinmod.conditioning_features import array_hash, baseline, mutation_features
from proteinmod.data import checksum, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.multiprotein_data import sequence_features
from proteinmod.multiprotein_experiments import macro_score, per_protein
from proteinmod.multitask_experiments import inputs
from proteinmod.stability_conditioning import (
    controls,
    fit_stability,
    neural_function,
    ridge_function,
    scalar_inputs,
)
from proteinmod.stability_experiments import scores
from proteinmod.structure_pipeline import write_csv


def history():
    return {
        str(p): checksum(p)
        for p in sorted(Path("benchmarks").rglob("*"))
        if p.is_file() and "m8" not in p.parts
    }


def output_path(path):
    output = Path(path).resolve()
    for p in Path("benchmarks").iterdir():
        protected = p.resolve()
        if p.name != "m8" and (
            output == protected or output in protected.parents or protected in output.parents
        ):
            raise ValueError("M8 output would overwrite historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output


def infer_stability(x, fx, rows, roles, alphas, identity, root):
    scaler, model, selection = fit_stability(x, rows, roles, alphas)
    weights = {
        "coefficient": model.coef_.tolist(),
        "coefficient_dtype": str(model.coef_.dtype),
        "input_dtype": str(x.dtype),
        "intercept": float(model.intercept_),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "unit": "kcal/mol",
        "convention": "positive_destabilizing",
    }
    identity = {
        **identity,
        "weights": fingerprint(weights),
        "source_features": array_hash(x),
        "function_features": array_hash(fx),
        "source_pipeline": checksum(Path(__file__)),
        "fitting_pipeline": checksum(Path("src/proteinmod/stability_conditioning.py")),
    }
    cache = Path(root) / "stability_predictions" / fingerprint(identity)
    cache.mkdir(parents=True, exist_ok=True)
    path, metadata = cache / "predictions.npy", cache / "manifest.json"
    eligible = np.flatnonzero(np.isfinite(fx).all(1))
    prediction = np.full(len(fx), np.nan)
    if path.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        if old["identity"] != identity or old["sha256"] != checksum(path):
            raise ValueError("Stability prediction cache provenance mismatch")
        prediction = np.load(path, allow_pickle=False)
        if prediction.shape != (len(fx),) or not np.isfinite(prediction[eligible]).all():
            raise ValueError("Invalid cached stability predictions")
    else:
        prediction[eligible] = model.predict(scaler.transform(fx[eligible]))
        if not np.isfinite(prediction[eligible]).all():
            raise ValueError("Nonfinite stability inference")
        np.save(path, prediction, allow_pickle=False)
    source_test = model.predict(scaler.transform(x[roles["test"]]))
    record = {
        "identity": identity,
        "weights": weights,
        "selection": selection,
        "sha256": checksum(path),
        "path": str(path),
        "stability_test": scores(rows, roles["test"], source_test),
        "source_test_predictions": source_test.tolist(),
        "ood": {
            "all_function_proteins": True,
            "confidence_available": False,
            "source_single_mutants_only": True,
            "prediction_min": float(prediction[eligible].min()),
            "prediction_max": float(prediction[eligible].max()),
            "feature_cells_outside_source_train_range": float(
                np.mean(
                    (fx[eligible] < x[roles["train"]].min(0))
                    | (fx[eligible] > x[roles["train"]].max(0))
                )
            ),
        },
    }
    write_json(metadata, record)
    return prediction, record


def run(config):
    started = time.perf_counter()
    before = history()
    output = output_path(config["output"])
    m7 = json.loads(Path(config["m7_config"]).read_text())
    m5 = json.loads(Path(m7["function_config"]).read_text())
    m4 = json.loads(Path(m5["m4_config"]).read_text())
    m6 = json.loads(Path(m7["stability_config"]).read_text())
    rows, groups, folds, audit = inputs(m7)
    fr, sr = rows["function"], rows["stability"]
    ff, fa = mutation_features(
        fr, groups["function"], m4["collection"], m4["embedding"], Path(config["cache"]) / "wt"
    )
    sf, sa = mutation_features(
        sr,
        groups["stability"],
        m6["proteins"],
        m6["embedding"],
        m6["residue_cache"],
        m6["sources"]["measurements"],
    )
    references = {s["protein_id"]: s["reference"]["sequence"] for s in m4["collection"]}
    sequence = np.array([sequence_features(r, references[r["protein_id"]]) for r in fr])
    classical = np.column_stack((sequence, ff["structural_ridge"][:, :270]))
    with Path("benchmarks/m7/predictions.csv").open() as handle:
        historical = list(csv.DictReader(handle))
    source_evidence, predictions, records, baseline_evidence, stability_values = [], [], [], [], []
    checkpoints = (
        Path(config["cache"])
        / "heads"
        / fingerprint(
            {
                "config": config,
                "historical": before,
                "source": {str(p): checksum(p) for p in Path("src/proteinmod").glob("*.py")},
            }
        )
    )
    checkpoints.mkdir(parents=True, exist_ok=True)

    def record(f, source, seed, model, control, roles, pred, selection):
        ix = roles["test"]
        if not np.isfinite(pred[ix]).all():
            raise ValueError("Missing held-out function prediction")
        records.append(
            {
                "protocol": "G1" if f == "within" else "G2",
                "fold": f,
                "source": source,
                "seed": seed,
                "model": model,
                "control": control,
                "selection": selection,
                "per_protein": per_protein(fr, ix, pred[ix]),
                "macro_spearman": macro_score(fr, ix, pred[ix]),
                "train_macro_spearman": macro_score(fr, roles["train"], pred[roles["train"]]),
            }
        )
        predictions.extend(
            {
                "protocol": "G1" if f == "within" else "G2",
                "fold": f,
                "source": source,
                "seed": seed,
                "model": model,
                "control": control,
                "id": fr[i]["id"],
                "protein_id": fr[i]["protein_id"],
                "assay_id": fr[i]["assay_id"],
                "measured": fr[i]["target"],
                "prediction": float(p),
                "prediction_unit": "training-assay percentile score",
            }
            for i, p in zip(ix, pred[ix], strict=True)
        )

    for f, s, paired in folds:
        roles = paired["function"]
        cpred, cselection = ridge_function(classical, fr, roles, config["alphas"])
        sources = {}
        for source in config["stability_sources"]:
            if source not in ("esm_sites_ridge", "structural_ridge"):
                raise ValueError("Only leakage-safe M6 sources supported")
            identity = {
                "schema": config["schema"],
                "source": source,
                "stability_fold": s,
                "source_dataset": m6["sources"]["measurements"],
                "roles": {p: [sr[i]["id"] for i in ix] for p, ix in paired["stability"].items()},
                "function_ids": [r["id"] for r in fr],
                "reference_features": {"function": fa, "stability": sa},
            }
            z, evidence = infer_stability(
                sf[source],
                ff[source],
                sr,
                paired["stability"],
                m6["alphas"],
                identity,
                config["cache"],
            )
            evidence.update({"function_fold": f, "source": source, "stability_fold": s})
            source_evidence.append(evidence)
            sources[source] = z
            for role, ix in roles.items():
                stability_values.extend(
                    {
                        "fold": f,
                        "source": source,
                        "function_role": role,
                        "id": fr[i]["id"],
                        "protein_id": fr[i]["protein_id"],
                        "predicted_ddg_kcal_mol": float(z[i]),
                        "positive_means": "destabilization",
                        "observed_stability_available": False,
                        "domain": "out_of_distribution",
                        "mutation_count": len(fr[i]["mutations"]),
                        "multi_mutant_extrapolation": len(fr[i]["mutations"]) > 1,
                        "outside_single_mutant_support": len(fr[i]["mutations"]) != 1,
                        "confidence_available": False,
                        "source_weights_sha256": evidence["identity"]["weights"],
                    }
                    for i in ix
                )
        for seed in config["seeds"]:
            h, base, provenance = baseline(
                fr, groups["function"], rows, groups, f, s, paired, m7, seed, config["cache"]
            )
            expected = {
                r["id"]: float(r["prediction"])
                for r in historical
                if r["function_fold"] == f
                and r["task"] == "function"
                and r["model"] == "independent"
                and int(r["seed"]) == seed
            }
            if set(expected) != {fr[i]["id"] for i in roles["test"]}:
                raise ValueError("Baseline evaluation IDs differ")
            delta = max(abs(base[i] - expected[fr[i]["id"]]) for i in roles["test"])
            if delta != 0:
                raise ValueError(
                    f"Fixed M7 baseline differs by {delta}; investigate before proceeding"
                )
            baseline_evidence.append(
                {
                    "fold": f,
                    "seed": seed,
                    "max_absolute_difference": delta,
                    "provenance": provenance,
                }
            )
            independent_scalar = np.zeros((len(fr), 2))
            independent_scalar[:, 1] = 1
            npred, nselection, nstate = neural_function(
                h, independent_scalar, fr, roles, config["head"], seed
            )
            record(f, "none", seed, "m7_function", "none", roles, base, {"frozen": True})
            record(f, "none", seed, "feature_ridge", "none", roles, cpred, cselection)
            record(f, "none", seed, "neural_head", "none", roles, npred, nselection)
            torch.save(nstate, checkpoints / f"{f}-{seed}-independent.pt")
            for source, z in sources.items():
                for control, values in controls(z, fr, roles, seed).items():
                    scalar, calibration = scalar_inputs(values, roles["train"])
                    for name, matrix, offset in (
                        ("stability_only", scalar, None),
                        ("feature_ridge", np.column_stack((classical, scalar)), None),
                        ("residual", scalar, base),
                    ):
                        pred, selection = ridge_function(
                            matrix, fr, roles, config["alphas"], offset
                        )
                        record(
                            f,
                            source,
                            seed,
                            name,
                            control,
                            roles,
                            pred,
                            {**selection, "scalar_calibration": calibration},
                        )
                    pred, selection, state = neural_function(
                        h, scalar, fr, roles, config["head"], seed
                    )
                    state.update(
                        {
                            "scalar_calibration": calibration,
                            "function_encoder_sha256": provenance["identity"]["checkpoint_sha256"],
                            "stability_source_sha256": next(
                                e["identity"]["weights"]
                                for e in source_evidence
                                if e["function_fold"] == f and e["source"] == source
                            ),
                            "source": source,
                            "control": control,
                            "config": config,
                        }
                    )
                    torch.save(state, checkpoints / f"{f}-{seed}-{source}-{control}.pt")
                    record(
                        f,
                        source,
                        seed,
                        "neural_head",
                        control,
                        roles,
                        pred,
                        {**selection, "scalar_calibration": calibration},
                    )
            print(f, seed, "finished", flush=True)
            write_json(output / "runs.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "predicted_stability.csv", stability_values)
    write_json(output / "source_models.json", source_evidence)
    write_json(output / "baseline_replay.json", baseline_evidence)
    write_json(
        output / "audit.json",
        {
            **audit,
            "historical_sha256": before,
            "paired_experimental_labels": 0,
            "function_stability_observations": 0,
            "function_stability_source": "predictions only, at training and inference",
            "source_function_supervision": False,
            "source_stability_training": "M6 S1/S2 exact roles; validation-only source alpha selection",
            "native_target_unit": "kcal/mol",
            "positive_means": "destabilization",
            "confidence_available": False,
            "m7_frozen_neural_stability_head_excluded": "amylase/NucB function supervision leaks into G2",
            "feature_hashes": {
                "function": {k: array_hash(v) for k, v in ff.items()},
                "stability": {k: array_hash(v) for k, v in sf.items()},
            },
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
            "source_sha256": {str(p): checksum(p) for p in Path("src/proteinmod").glob("*.py")},
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "scikit-learn", "biopython")
            },
            "python": platform.python_version(),
            "machine": platform.machine(),
            "device": "cpu",
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
    parser.add_argument("--config", default="configs/m8.json")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
