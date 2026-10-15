"""Independent experimental ΔΔG fits; physical targets, frozen encoders, no multitask loss."""

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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from proteinmod.data import checksum, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.models import metrics, one_hot
from proteinmod.stability_data import acquire, history, output_directory, stability_folds
from proteinmod.structure_pipeline import write_csv


def validate_task(rows):
    if any(
        r["task_type"] != "ddg"
        or r["target_unit"] != "kcal/mol"
        or r["target_convention"] != "positive_destabilizing"
        or not r["target_mask"]
        or not np.isfinite(r["target"])
        for r in rows
    ):
        raise ValueError("Only finite canonical ΔΔG labels may enter stability fits")
    if len({(r["assay_id"], fingerprint(r["conditions"])) for r in rows}) != 1:
        raise ValueError(
            "Different assays/conditions must be evaluated as separate stability tasks"
        )


def scores(rows, indices, prediction):
    values = {}
    for p in sorted({rows[i]["protein_id"] for i in indices}):
        local = [j for j, i in enumerate(indices) if rows[i]["protein_id"] == p]
        values[p] = {
            "n": len(local),
            **metrics([rows[indices[j]]["target"] for j in local], np.asarray(prediction)[local]),
        }
    return {
        "per_protein": values,
        "macro": {
            key: float(np.mean([v[key] if v[key] is not None else 0 for v in values.values()]))
            for key in ("spearman_rho", "pearson_r", "mae", "rmse")
        },
        "pooled": metrics([rows[i]["target"] for i in indices], prediction),
        "unit": "kcal/mol",
        "undefined_correlation_macro_policy": "zero; raw per-protein undefined correlations retained",
    }


def ridge_fit(x, rows, roles, alphas):
    validate_task(rows)
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("Positive finite ridge alphas required")
    scaler = StandardScaler(with_mean=not hasattr(x, "tocsr")).fit(x[roles["train"]])
    x = scaler.transform(x)
    target = np.array([r["target"] for r in rows])
    best, candidates = None, []
    for alpha in sorted(alphas):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000).fit(
            x[roles["train"]], target[roles["train"]]
        )
        score = scores(rows, roles["validation"], model.predict(x[roles["validation"]]))["macro"][
            "rmse"
        ]
        candidates.append({"alpha": alpha, "validation_macro_rmse_kcal_mol": score})
        if best is None or score < best[0]:
            best = score, model, alpha
    return best[1].predict(x[roles["test"]]), {
        "alpha": best[2],
        "validation_macro_rmse_kcal_mol": best[0],
        "candidates": candidates,
        "train": scores(rows, roles["train"], best[1].predict(x[roles["train"]])),
    }


def train_head(features, rows, roles, settings, seed):
    import torch
    from torch import nn

    validate_task(rows)
    if (
        settings["device"] != "cpu"
        or min(
            settings[k]
            for k in ("cpu_threads", "dimension", "max_epochs", "patience", "batch_size")
        )
        < 1
        or settings["learning_rate"] <= 0
        or settings["weight_decay"] < 0
    ):
        raise ValueError("Positive deterministic CPU training settings required")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    train = roles["train"]
    mean = float(np.mean([rows[i]["target"] for i in train]))
    scale = max(float(np.std([rows[i]["target"] for i in train])), 1e-8)
    # Fixed frozen features are all that reaches this head; no label-dependent encoder update.
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.full((len(rows),), float("nan"))
    y[train] = torch.tensor(
        [(rows[i]["target"] - mean) / scale for i in train], dtype=torch.float32
    )
    model = nn.Sequential(
        nn.Linear(features.shape[1], settings["dimension"]),
        nn.ReLU(),
        nn.Linear(settings["dimension"], 1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(seed)

    def predict(indices):
        model.eval()
        with torch.no_grad():
            return model(x[indices]).squeeze(-1).numpy().astype(float) * scale + mean

    best, stale, trace = None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), settings["batch_size"]):
            ix = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[ix]).squeeze(-1) - y[ix]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite stability head loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        score = scores(rows, roles["validation"], predict(roles["validation"]))["macro"]["rmse"]
        trace.append({"epoch": epoch, "validation_macro_rmse_kcal_mol": score})
        if best is None or score < best[0]:
            best = score, epoch, copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    return (
        predict(roles["test"]),
        {
            "epoch": best[1],
            "validation_macro_rmse_kcal_mol": best[0],
            "trace": trace,
            "train": scores(rows, roles["train"], predict(roles["train"])),
            "train_only_target_mean": mean,
            "train_only_target_scale": scale,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        },
        {"stability_head": best[2], "target_mean": mean, "target_scale": scale, "seed": seed},
    )


def run(config):
    import torch

    from proteinmod.stability_features import model_inputs

    started = time.perf_counter()
    before = history()
    output = output_directory(config["output"])
    rows, features, audit = model_inputs(config)
    validate_task(rows)
    features["mutation_count_ridge"] = np.array([[len(r["mutations"])] for r in rows], dtype=float)
    records, predictions, roles_evidence = [], [], []
    for protocol, fold, roles in stability_folds(rows, config):
        roles_evidence.append(
            {
                "protocol": protocol,
                "fold": fold,
                "ids": {p: [rows[i]["id"] for i in ix] for p, ix in roles.items()},
                "proteins": {
                    p: sorted({rows[i]["protein_id"] for i in ix}) for p, ix in roles.items()
                },
            }
        )
        for name in ["training_mean", *features, "stability_head"]:
            for seed in config["seeds"] if name == "stability_head" else [0]:
                fit_started = time.perf_counter()
                if name == "training_mean":
                    pred = np.full(
                        len(roles["test"]), np.mean([rows[i]["target"] for i in roles["train"]])
                    )
                    selection = {"rule": "training labels only"}
                elif name == "stability_head":
                    pred, selection, state = train_head(
                        features["frozen_hybrid_ridge"], rows, roles, config["head"], seed
                    )
                    state["encoder_tensor_sha256"] = config["transfer"]["encoder_tensor_sha256"]
                    state["feature_schema"] = "m6-frozen-hybrid-v1"
                    path = Path("data/cache/m6/heads") / fingerprint(config)
                    path.mkdir(parents=True, exist_ok=True)
                    torch.save(state, path / f"{protocol}-{fold}-{seed}.pt")
                else:
                    pred, selection = ridge_fit(features[name], rows, roles, config["alphas"])
                result = {
                    "protocol": protocol,
                    "fold": fold,
                    "model": name,
                    "seed": seed,
                    "test": scores(rows, roles["test"], pred),
                    "selection": selection,
                    "elapsed_seconds": time.perf_counter() - fit_started,
                }
                records.append(result)
                predictions.extend(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "model": name,
                        "seed": seed,
                        "id": rows[i]["id"],
                        "protein_id": rows[i]["protein_id"],
                        "assay_id": rows[i]["assay_id"],
                        "measured_ddg_kcal_mol": rows[i]["target"],
                        "prediction_ddg_kcal_mol": float(v),
                        "positive_means": "destabilizing",
                        "mutations": ";".join(rows[i]["mutations"]),
                    }
                    for i, v in zip(roles["test"], pred, strict=True)
                )
                print(protocol, fold, name, seed, result["test"]["macro"], flush=True)
        if protocol == "S1":
            for p in sorted({r["protein_id"] for r in rows}):
                local = {
                    role: np.array([i for i in ix if rows[i]["protein_id"] == p])
                    for role, ix in roles.items()
                }
                # Different reference lengths/positions cannot be aligned across unrelated proteins.
                indices = np.flatnonzero([r["protein_id"] == p for r in rows])
                local_rows = [rows[i] for i in indices]
                lookup = {i: j for j, i in enumerate(indices)}
                remapped = {role: np.array([lookup[i] for i in ix]) for role, ix in local.items()}
                pred, selection = ridge_fit(
                    one_hot([r["sequence"] for r in local_rows]),
                    local_rows,
                    remapped,
                    config["alphas"],
                )
                records.append(
                    {
                        "protocol": protocol,
                        "fold": p,
                        "model": "within_position_one_hot",
                        "seed": 0,
                        "test": scores(rows, local["test"], pred),
                        "selection": selection,
                    }
                )
                predictions.extend(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "model": "within_position_one_hot",
                        "seed": 0,
                        "id": rows[i]["id"],
                        "protein_id": p,
                        "assay_id": rows[i]["assay_id"],
                        "measured_ddg_kcal_mol": rows[i]["target"],
                        "prediction_ddg_kcal_mol": float(v),
                        "positive_means": "destabilizing",
                        "mutations": ";".join(rows[i]["mutations"]),
                    }
                    for i, v in zip(local["test"], pred, strict=True)
                )
    if history() != before:
        raise ValueError("M0–M5 history changed")
    write_json(output / "runs.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_json(output / "rows.json", rows)
    write_json(
        output / "audit.json",
        {
            **audit,
            "roles": roles_evidence,
            "historical_sha256": before,
            "label_policy": "Native kcal/mol ΔΔG, positive destabilization; training-only mean/std for head; validation-only macro RMSE selection",
            "no_multitask_training": True,
            "homology_rule": "Block >=30% aligned identity with >=80% coverage of BOTH references; descriptive threshold, not a family certificate",
        },
    )
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "config_fingerprint": fingerprint(config),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "torch", "biopython")
            },
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "source_sha256": {p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")},
            "artifacts_sha256": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "prepare", "rebuild-transfer", "run"))
    parser.add_argument("--config", default="configs/m6.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    from proteinmod.stability_features import prepare, rebuild_transfer

    {"download": acquire, "prepare": prepare, "rebuild-transfer": rebuild_transfer, "run": run}[
        args.action
    ](config)


if __name__ == "__main__":
    main()
