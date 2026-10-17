"""Opt-in cold acquisition and CPU/MPS qualification; refuses existing outputs."""

import copy
import hashlib
import json
import os
import platform
import time
import tomllib
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, download, load_dataset, verified_reference, write_json
from proteinmod.embeddings import FrozenESM, cache_location, extract, load_features
from proteinmod.experiments import run_experiments
from proteinmod.models import fit_ridge, metrics
from proteinmod.portability import activation_comparison


def main():
    import torch
    from huggingface_hub import snapshot_download

    policy = json.loads(Path("configs/m161.json").read_text())
    root, output = Path(policy["cache"]), Path(policy["output"])
    if root.exists() or output.exists():
        raise ValueError("Preserve previous attempts: use a fresh configured cache/output")
    if not torch.backends.mps.is_available():
        raise ValueError("This paired qualification requires native MPS access")
    root.mkdir(parents=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    config = tomllib.loads(Path("configs/amylase_m1.toml").read_text())
    config["dataset"]["path"] = str(root / "by_mutation.csv.gz")
    config["dataset"]["sha256"] = "2e76acc9da3ef04a8ccecd8a3f631b84ee2e4e1f0cdea406f18153387b986d04"
    download(config["dataset"])
    pins = json.loads(Path("configs/m15.json").read_text())["checkpoint_files"]
    settings = config["embedding"]
    settings["model_cache"] = str(root / "models")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    snapshot = Path(
        snapshot_download(
            settings["model"],
            revision=settings["revision"],
            cache_dir=settings["model_cache"],
            allow_patterns=[Path(p).name for p in pins],
            token=False,
        )
    )
    for path, expected in pins.items():
        assert checksum(snapshot / Path(path).name) == expected
    write_json(
        output / "acquisition.json",
        {
            "dataset": checksum(config["dataset"]["path"]),
            "checkpoint": {Path(p).name: v for p, v in pins.items()},
            "cold_cache_initially_absent": True,
        },
    )
    reference = verified_reference(config["reference"])
    rows, qc = load_dataset(config["dataset"]["path"], reference)
    sequences = [r["sequence"] for r in rows]
    probe_indices = sorted(
        range(len(rows)), key=lambda i: hashlib.sha256(sequences[i].encode()).hexdigest()
    )[:64]
    probe = [reference] + [sequences[i] for i in probe_indices]
    layers, features, configs, repeats = {}, {}, {}, {}
    for device in policy["devices"]:
        cfg = copy.deepcopy(config)
        cfg["embedding"].update(device=device, cache=str(root / device))
        cfg["learning_curve"]["output"] = str(output / device)
        encoder = FrozenESM(cfg["embedding"], local_only=True)
        first, second = [], []
        for destination in (first, second):
            for start in range(0, len(probe), 8):
                destination.extend(encoder.encode(probe[start : start + 8]))
        repeats[device] = activation_comparison(np.stack(first), np.stack(second))
        captured = {}
        hooks = [
            layer.register_forward_hook(
                lambda module, inputs, result, index=i, captured=captured: captured.update(
                    {str(index): result[0].detach().cpu().numpy().copy()}
                )
            )
            for i, layer in enumerate(encoder.model.encoder.layer)
        ]
        encoder.encode(probe[:8])
        for hook in hooks:
            hook.remove()
        layers[device] = captured
        extract(cfg, sequences, reference, checksum(cfg["dataset"]["path"]), encoder=encoder)
        directory, identity = cache_location(
            cfg, sequences, reference, checksum(cfg["dataset"]["path"])
        )
        features[device], _ = load_features(directory, identity, len(rows), 480)
        configs[device] = cfg
        del encoder
        if device == "mps":
            torch.mps.empty_cache()
        run_experiments(cfg)
        write_json(
            output / "progress.json",
            {"completed": device, "seconds": time.perf_counter() - started},
        )
    result = {
        "policy": policy,
        "qc": qc,
        "probe_ids": [rows[i]["id"] for i in probe_indices],
        "same_device": repeats,
        "layers": {
            i: activation_comparison(layers["cpu"][i], layers["mps"][i]) for i in layers["cpu"]
        },
        "features": {},
        "functional": {},
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    y = np.array([r["target"] for r in rows])
    indices = {
        p: np.array([i for i, r in enumerate(rows) if r["partition"] == p])
        for p in ("train", "validation", "test")
    }
    tr, va, te = (indices[p] for p in ("train", "validation", "test"))
    scale = float(y[tr].std())
    selections = {
        d: json.loads((output / d / "metrics.json").read_text()) for d in policy["devices"]
    }
    for name in features["cpu"]:
        a, b = features["cpu"][name], features["mps"][name]
        result["features"][name] = activation_comparison(a, b)
        fitted, _ = fit_ridge(a[tr], y[tr], a[va], y[va], config["run"]["alphas"], 0)
        budget = 32 * 12 * np.finfo(np.float32).eps * (1 + np.maximum(abs(a[te]), abs(b[te])))
        bound = budget @ abs(fitted.coef_)
        fixed_delta = abs(fitted.predict(a[te]) - fitted.predict(b[te]))
        import csv

        predictions = {}
        for d in policy["devices"]:
            with (output / d / "test_predictions.csv").open() as handle:
                records = list(csv.DictReader(handle))
            assert [r["id"] for r in records] == [rows[i]["id"] for i in te]
            assert np.array_equal([float(r["measured"]) for r in records], y[te])
            predictions[d] = np.array([float(r[name]) for r in records])
        scores = {d: metrics(y[te], p) for d, p in predictions.items()}
        differences = {
            "maximum_prediction_change_over_train_sd": float(
                abs(predictions["cpu"] - predictions["mps"]).max() / scale
            ),
            "spearman_change": abs(scores["cpu"]["spearman_rho"] - scores["mps"]["spearman_rho"]),
            **{
                m + "_change_over_train_sd": abs(scores["cpu"][m] - scores["mps"][m]) / scale
                for m in ("rmse", "mae")
            },
        }
        same_selection = (
            selections["cpu"]["selection"][name] == selections["mps"]["selection"][name]
        )
        # Compare chosen alpha, not slightly different validation scores.
        same_alpha = (
            selections["cpu"]["selection"][name]["alpha"]
            == selections["mps"]["selection"][name]["alpha"]
        )
        result["functional"][name] = {
            "metrics": scores,
            "differences": differences,
            "same_alpha": same_alpha,
            "validation_records_exact": same_selection,
            "pass": same_alpha
            and all(v <= policy["functional_margin"] for v in differences.values()),
            "fixed_weights_maximum_difference": float(fixed_delta.max()),
            "linear_bound_pass": bool((fixed_delta <= bound + 1e-12).all()),
        }
    result["negative_control"] = activation_comparison(np.ones(100), np.ones(100) * 1.01)
    result["seconds"] = time.perf_counter() - started
    write_json(output / "portability.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
