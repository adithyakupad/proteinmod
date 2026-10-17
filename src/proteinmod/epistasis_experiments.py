"""Explicit symmetric interactions on a pinned measured GB1 landscape."""

import argparse
import copy
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from proteinmod.assay_features import encoder_state
from proteinmod.assay_models import scale_fit
from proteinmod.confidence_fusion import frozen_representation
from proteinmod.data import checksum, verified_reference, write_json
from proteinmod.embeddings import FrozenESM, fingerprint, site_features
from proteinmod.epistasis_data import acquire, additive, collection, protocols
from proteinmod.models import metrics
from proteinmod.mutation_interactions import (
    GEOMETRY_NAMES,
    PAIR_WIDTH,
    SEQUENCE_WIDTH,
    pair_features,
)
from proteinmod.residue_cache import residue_cache
from proteinmod.structural_graph import AA, load_graph, variant_inputs
from proteinmod.structure_pipeline import write_csv


def historical():
    result = {}
    for root in (Path("benchmarks"), Path("configs")):
        for p in sorted(root.rglob("*")):
            name = p.parts[1] if root.name == "benchmarks" else p.stem
            if p.is_file() and not (
                name.startswith("m") and name[1:].isdigit() and int(name[1:]) >= 11
            ):
                result[str(p)] = checksum(p)
    return result


def output_path(path):
    output, root = Path(path).resolve(), Path("benchmarks").resolve()
    allowed = root / "m11"
    if (output == root or root in output.parents or output in root.parents) and not (
        output == allowed or allowed in output.parents
    ):
        raise ValueError("M11 would overwrite historical evidence")
    output.mkdir(parents=True, exist_ok=True)
    return output


def feature_root(config):
    contract = {
        k: config[k]
        for k in (
            "dataset",
            "reference",
            "full_reference",
            "structure_artifacts",
            "positions",
            "caps",
            "split_salt",
            "fitness_floor",
            "embedding",
            "pretrained",
            "seeds",
        )
    }
    sources = {
        n: checksum(Path("src/proteinmod") / n)
        for n in ("epistasis_data.py", "mutation_interactions.py", "epistasis_experiments.py")
    }
    return (
        Path(config["cache"]) / "features" / fingerprint({"config": contract, "sources": sources})
    )


def prepare(config):
    start = time.perf_counter()
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    ref = verified_reference(config["reference"])
    verified_reference(config["full_reference"])
    rows, _, audit = collection(config)
    graph = load_graph(config["structure_artifacts"], ref)
    label_free = Path(config["cache"]) / "variants.json"
    label_free.parent.mkdir(parents=True, exist_ok=True)
    token_rows = [{k: r[k] for k in ("id", "sequence", "mutations")} for r in rows]
    write_json(label_free, {"rows": token_rows, "source_sha256": config["dataset"]["sha256"]})
    spec = {"reference": config["reference"], "dataset": {"path": str(label_free)}}
    settings = config["embedding"]
    root = Path(config["cache"]) / "residues"
    try:
        tokens, token_manifest, directory = residue_cache(spec, token_rows, settings, root)
    except ValueError as error:
        if "Missing M5 residue cache" not in str(error):
            raise
        tokens, token_manifest, directory = residue_cache(
            spec, token_rows, settings, root, True, FrozenESM(settings, local_only=True)
        )
    wt = next(i for i, r in enumerate(rows) if not r["mutations"])
    x = {
        "onehot": np.zeros((len(rows), len(config["positions"]) * 20), np.float32),
        "esm": np.zeros((len(rows), 961), np.float32),
        "structural": np.zeros((len(rows), 270), np.float32),
        "hybrid": np.zeros((len(rows), 100), np.float32),
    }
    dynamic = []
    mask = np.zeros((len(rows), len(ref)), bool)
    for i, r in enumerate(rows):
        d, features, reasons = variant_inputs(r["mutations"], ref, graph)
        if reasons:
            raise ValueError("Selected mutation lacks exact structural mapping")
        dynamic.append(d)
        x["structural"][i] = features
        positions = [int(t[1:-1]) - 1 for t in r["mutations"]]
        mask[i, positions] = True
        x["esm"][i] = site_features(tokens[i], tokens[wt], positions)
        for t in r["mutations"]:
            x["onehot"][i, config["positions"].index(int(t[1:-1])) * 20 + AA[t[-1]]] = 1
    state = encoder_state(config)
    for start_batch in range(0, len(rows), 32):
        ix = slice(start_batch, min(start_batch + 32, len(rows)))
        x["hybrid"][ix] = frozen_representation(
            state, graph, np.array(tokens[ix], copy=True), np.array(dynamic[ix]), mask[ix]
        )
    context = {
        r["mutations"][0]: x["hybrid"][i] for i, r in enumerate(rows) if len(r["mutations"]) == 1
    }
    maximum = max(len(r["mutations"]) * (len(r["mutations"]) - 1) // 2 for r in rows)
    pair_mask = np.zeros((len(rows), maximum), bool)
    profiles = [("real", 0)] + [
        (c, s) for c in ("shuffled_distance", "shuffled_biochemical") for s in config["seeds"]
    ]
    for control, seed in profiles:
        key = control if control == "real" else f"{control}{seed}"
        array = np.zeros((len(rows), maximum, PAIR_WIDTH + 300), np.float32)
        for i, r in enumerate(rows):
            value = pair_features(
                r["mutations"], ref, graph, config["positions"], control, seed, context
            )
            array[i, : len(value)] = value
            pair_mask[i, : len(value)] = True
        x["pairs_" + key] = array
    x["pair_mask"] = pair_mask
    if any(not np.isfinite(v).all() for v in x.values()):
        raise ValueError("Nonfinite frozen inputs")
    output = feature_root(config)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", **x)
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "data": audit,
            "graph": graph["audit"],
            "rows_fingerprint": fingerprint(rows),
            "ordered_ids": [r["id"] for r in rows],
            "token_manifest_sha256": checksum(directory / "manifest.json"),
            "encoder": token_manifest["encoder"],
            "pretrained": config["pretrained"],
            "feature_sha256": checksum(output / "features.npz"),
            "elapsed_seconds": time.perf_counter() - start,
            "mapping_exclusions": 0,
            "pair_schema": {
                "sequence_width": SEQUENCE_WIDTH,
                "geometry_names": GEOMETRY_NAMES,
                "explicit_width": PAIR_WIDTH,
                "hybrid_width": PAIR_WIDTH + 300,
                "aggregation": "sum of symmetric, background-independent pair features",
            },
            "pair_context": "reference single-mutant frozen multimodal features; background independent",
            "stability_scalar_inputs": False,
        },
    )
    return output


def load(config, rows):
    root = feature_root(config)
    m = json.loads((root / "manifest.json").read_text())
    if (
        m["ordered_ids"] != [r["id"] for r in rows]
        or m["rows_fingerprint"] != fingerprint(rows)
        or checksum(root / "features.npz") != m["feature_sha256"]
    ):
        raise ValueError("M11 feature identity mismatch")
    with np.load(root / "features.npz", allow_pickle=False) as z:
        x = {k: z[k] for k in z.files}
    return x, m


def ridge_fit(x, y, roles, alphas, base=None, intercept=True):
    base = np.zeros(len(y)) if base is None else np.asarray(base)
    scaler = scale_fit(np.asarray(x, dtype=np.float64), roles["train"])
    z = scaler.transform(np.asarray(x, dtype=np.float64))
    # Pair residuals must have exactly zero features at WT/singles after scaling.
    if not intercept:
        scaler.mean_ = np.zeros_like(scaler.mean_)
        z = scaler.transform(np.asarray(x, dtype=np.float64))
    best = None
    candidates = []
    for alpha in sorted(alphas):
        model = Ridge(
            alpha=alpha, fit_intercept=intercept, solver="lsqr", tol=1e-9, max_iter=10000
        ).fit(z[roles["train"]], (y - base)[roles["train"]])
        score = metrics(
            y[roles["validation"]],
            base[roles["validation"]] + model.predict(z[roles["validation"]]),
        )["rmse"]
        candidates.append({"alpha": alpha, "validation_rmse": score})
        if best is None or score < best[0]:
            best = score, alpha, model
    prediction = base + best[2].predict(z)
    selected = {
        "alpha": best[1],
        "validation_rmse": best[0],
        "candidates": candidates,
        "parameters": len(best[2].coef_) + int(intercept),
    }
    state = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": best[2].coef_.tolist(),
        "intercept": float(best[2].intercept_),
        "residual": base.tolist(),
    }
    return prediction, selected, state


class PairResidual(nn.Module):
    def __init__(self, width, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, pairs, mask):
        if (
            pairs.ndim != 3
            or pairs.shape[-1] != self.net[0].in_features
            or mask.shape != pairs.shape[:2]
            or mask.dtype != torch.bool
        ):
            raise ValueError("Invalid pair tensor/mask")
        pairs = pairs.masked_fill(~mask[..., None], 0)
        if not torch.isfinite(pairs).all():
            raise ValueError("Nonfinite pair tensor")
        return self.net(pairs).squeeze(-1).masked_fill(~mask, 0).sum(1)


def neural_features(x, mode, seed):
    if mode not in (
        "sequence_pairs",
        "structural_pairs",
        "hybrid_pairs",
        "hybrid_zero3d",
        "hybrid_shuffled_distance",
        "hybrid_shuffled_biochemical",
    ):
        raise ValueError("Unknown explicit interaction mode")
    key = "pairs_" + (
        f"shuffled_distance{seed}"
        if mode == "hybrid_shuffled_distance"
        else f"shuffled_biochemical{seed}"
        if mode == "hybrid_shuffled_biochemical"
        else "real"
    )
    pairs = x[key][..., :PAIR_WIDTH] if mode in ("sequence_pairs", "structural_pairs") else x[key]
    if mode in ("sequence_pairs", "hybrid_zero3d"):
        pairs = pairs.copy()
        pairs[..., SEQUENCE_WIDTH:PAIR_WIDTH] = 0
    return pairs


def neural_fit(pairs, mask, y, base, roles, settings, seed):
    if settings["device"] != "cpu":
        raise ValueError("Deterministic CPU required")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    local = pairs[roles["train"]][mask[roles["train"]]]
    mean = local.mean(0, dtype=np.float64)
    scale = local.std(0, dtype=np.float64)
    scale[scale < settings["scale_floor"]] = 1
    z = torch.tensor((pairs - mean) / scale, dtype=torch.float32)
    m = torch.tensor(mask)
    residual = y - base
    target_scale = max(float(np.std(residual[roles["train"]])), 1e-6)
    target = torch.tensor(residual / target_scale, dtype=torch.float32)
    model = PairResidual(pairs.shape[-1], settings["hidden"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(seed)
    best = None
    stale = 0
    trace = []

    def predict():
        model.eval()
        with torch.no_grad():
            return base + model(z, m).double().numpy() * target_scale

    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = rng.permutation(roles["train"])
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            ix = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(z[ix], m[ix]) - target[ix]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite interaction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        score = metrics(y[roles["validation"]], predict()[roles["validation"]])["rmse"]
        trace.append(
            {"epoch": epoch, "training_mse_scaled": total / len(order), "validation_rmse": score}
        )
        if best is None or score < best[0]:
            best = (score, epoch, copy.deepcopy(model.state_dict()))
            stale = 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    prediction = predict()
    state = {
        "state": best[2],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "target_scale": target_scale,
        "base": base.tolist(),
        "width": pairs.shape[-1],
        "hidden": settings["hidden"],
    }
    return (
        prediction,
        {
            "epoch": best[1],
            "validation_rmse": best[0],
            "trace": trace,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        state,
    )


def contrast_predictions(rows, contrasts, prediction, test):
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    allowed = set(test)
    output = []
    for c in contrasts:
        if lookup[c["target"]] not in allowed:
            continue
        keys = [lookup[c[k]] for k in ("target", "a", "b", "background")]
        signs = np.array([1, -1, -1, 1])
        measured = float(np.array([rows[i]["target"] for i in keys]) @ signs)
        predicted = float(np.asarray(prediction)[keys] @ signs)
        if abs(predicted) < 1e-10:
            predicted = 0.0
        linear = float(np.array([rows[i]["raw_fitness"] for i in keys]) @ signs)
        output.append(
            {
                **c,
                "measured_log_epistasis": measured,
                "predicted_log_epistasis": predicted,
                "measured_linear_epistasis": linear,
            }
        )
    return output


def epsilon_metrics(records):
    if not records:
        return {"n": 0, "spearman": None, "mae": None}
    y = np.array([r["measured_log_epistasis"] for r in records])
    p = np.array([r["predicted_log_epistasis"] for r in records])
    score = metrics(y, p)
    return {
        "n": len(records),
        "spearman": score["spearman_rho"],
        "mae": float(np.mean(np.abs(y - p))),
    }


def run(config):
    before = historical()
    started = time.perf_counter()
    output = output_path(config["output"])
    rows, contrasts, data = collection(config)
    x, manifest = load(config, rows)
    y = np.array([r["target"] for r in rows])
    records = []
    predictions = []
    epistasis = []
    role_evidence = []
    sources = {p.name: checksum(p) for p in Path("src/proteinmod").glob("*epistasis*.py")}
    sources["mutation_interactions.py"] = checksum(Path("src/proteinmod/mutation_interactions.py"))
    cache = (
        Path(config["cache"])
        / "models"
        / fingerprint(
            {"config": config, "features": manifest["feature_sha256"], "sources": sources}
        )
    )
    cache.mkdir(parents=True, exist_ok=True)

    def record(protocol, name, seed, role, pred, selection, state, elapsed):
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite held-out prediction")
        eps = contrast_predictions(rows, contrasts, pred, role["test"])
        path = cache / f"{protocol}-{name}-{seed}.pt"
        torch.save(state, path)
        records.append(
            {
                "protocol": protocol,
                "model": name,
                "seed": seed,
                "selection": selection,
                "test": metrics(y[role["test"]], pred[role["test"]]),
                "train": metrics(y[role["train"]], pred[role["train"]]),
                "epistasis_WT": epsilon_metrics([r for r in eps if not r["background_mutations"]]),
                "epistasis_nonWT": epsilon_metrics([r for r in eps if r["background_mutations"]]),
                "elapsed_seconds": elapsed,
                "checkpoint_sha256": checksum(path),
            }
        )
        predictions.extend(
            {
                "protocol": protocol,
                "model": name,
                "seed": seed,
                "id": rows[i]["id"],
                "sequence": rows[i]["sequence"],
                "mutation_count": len(rows[i]["mutations"]),
                "observed_log_fitness": y[i],
                "predicted_log_fitness": pred[i],
                "observed_raw_fitness": rows[i]["raw_fitness"],
                "censored": rows[i]["raw_fitness"] < config["fitness_floor"],
            }
            for i in role["test"]
        )
        epistasis.extend({"protocol": protocol, "model": name, "seed": seed, **r} for r in eps)

    for protocol, role in protocols(rows, config):
        base, anchors = additive(rows, role["train"])
        role_evidence.append(
            {
                "protocol": protocol,
                "ids": {p: [rows[i]["id"] for i in ix] for p, ix in role.items()},
                "additive": anchors,
            }
        )
        for name, pred in [
            ("constant", np.full(len(rows), y[role["train"]].mean())),
            ("single_additive", base),
        ]:
            record(
                protocol,
                name,
                0,
                role,
                pred,
                {"parameters": 1 if name == "constant" else len(anchors["deltas"]) + 1},
                {"kind": "fixed", "prediction": pred.tolist()},
                0,
            )
        baselines = {
            "mutation_count": np.array([[len(r["mutations"])] for r in rows]),
            **{n + "_ridge": x[n] for n in ("onehot", "esm", "structural", "hybrid")},
        }
        for name, matrix in baselines.items():
            start = time.perf_counter()
            pred, selected, state = ridge_fit(matrix, y, role, config["alphas"])
            record(
                protocol,
                name,
                0,
                role,
                pred,
                selected,
                {"kind": "ridge", "feature": name, "state": state},
                time.perf_counter() - start,
            )
        for name, width in [
            ("sequence_pair_ridge", SEQUENCE_WIDTH),
            ("structural_pair_ridge", PAIR_WIDTH),
        ]:
            start = time.perf_counter()
            matrix = x["pairs_real"][..., :width].sum(1)
            pred, selected, state = ridge_fit(matrix, y, role, config["alphas"], base, False)
            record(
                protocol,
                name,
                0,
                role,
                pred,
                selected,
                {"kind": "ridge", "feature": name, "state": state},
                time.perf_counter() - start,
            )
        for seed in config["seeds"]:
            for mode in config["neural_modes"]:
                start = time.perf_counter()
                pairs = neural_features(x, mode, seed)
                pred, selected, state = neural_fit(
                    pairs, x["pair_mask"], y, base, role, config["training"], seed
                )
                record(
                    protocol,
                    mode,
                    seed,
                    role,
                    pred,
                    selected,
                    {"kind": "neural", "state": state},
                    time.perf_counter() - start,
                )
            print(protocol, seed, "complete", flush=True)
        write_json(output / "runs.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "epistasis_predictions.csv", epistasis)
    write_json(output / "rows.json", rows)
    write_json(output / "contrasts.json", contrasts)
    write_json(
        output / "audit.json",
        {
            "data": data,
            "features": manifest,
            "history": before,
            "roles": role_evidence,
            "test_pairing": config["test_position_pair"],
            "validation_pairing": config["validation_position_pair"],
            "whole_protein_generalization": False,
            "measured_epistasis_not_inferred_from_fitness_accuracy": True,
        },
    )
    if historical() != before:
        raise ValueError("Historical evidence changed")
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "sources": sources,
            "checkpoint_root": str(cache),
            "feature_root": str(feature_root(config)),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "scikit-learn", "biopython", "transformers")
            },
            "machine": platform.machine(),
            "python": platform.python_version(),
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
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("action", choices=["download", "prepare", "run"])
    p.add_argument("--config", default="configs/m11.json")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.action == "download":
        acquire(config)
    elif args.action == "prepare":
        print(prepare(config))
    else:
        run(config)


if __name__ == "__main__":
    main()
