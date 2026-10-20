"""M15 fresh last-block adapters, matched head capacity, fixed family-held-out roles."""

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

from proteinmod.data import checksum, verify_file, write_json
from proteinmod.edge_chemistry import annotate, mapped_atoms
from proteinmod.edge_experiments import fit as graph_fit
from proteinmod.edge_experiments import ridge_predict
from proteinmod.embeddings import FrozenESM
from proteinmod.family_data import audit, folds
from proteinmod.family_experiments import inputs
from proteinmod.lora_data import history, prefix_cache, subset
from proteinmod.lora_model import AdaptedESM, tensor_digest
from proteinmod.multiprotein_experiments import fit_targets, macro_score, per_protein
from proteinmod.structure_pipeline import write_csv
from proteinmod.uncertainty_models import ridge_ensemble


def batch_inputs(group, local):
    prefix = torch.tensor(np.array(group["prefix"][np.r_[local, len(group["indices"])]], copy=True))
    extended = torch.zeros((len(prefix), 1, 1, prefix.shape[1]), dtype=torch.float32)
    return {
        "prefix": prefix,
        "extended_mask": extended,
        "mutation_mask": torch.tensor(group["mutation_mask"][local]),
    }


def call(model, group, local):
    if model.mode == "lora":
        return model(**batch_inputs(group, local))
    return model(features=torch.tensor(group["frozen"][local], dtype=torch.float32))


def predict(model, groups, indices, batch_size):
    model.eval()
    found = {}
    with torch.no_grad():
        for g in groups.values():
            local = np.flatnonzero(np.isin(g["indices"], indices))
            for start in range(0, len(local), batch_size):
                ix = local[start : start + batch_size]
                values = call(model, g, ix).numpy().astype(float)
                found.update(zip(g["indices"][ix], values, strict=True))
    return np.array([found[i] for i in indices])


def fit(backbone, groups, rows, roles, config, seed, mode):
    settings = config["training"]
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    if any(len(batch_indices) == 0 for batch_indices in roles.values()) or any(
        set(roles[a]) & set(roles[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Training roles must be nonempty and disjoint")
    targets = torch.tensor(fit_targets(rows, roles["train"]), dtype=torch.float32)
    winner, candidates = None, []
    for rate in settings["learning_rates"]:
        model = AdaptedESM(backbone, config["lora"], settings["hidden"], mode, seed)
        frozen_before = model.frozen_digest()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in trainable if n.startswith("head.")], "lr": rate},
                {
                    "params": [p for n, p in trainable if not n.startswith("head.")],
                    "lr": rate * settings["adapter_lr_ratio"],
                },
            ],
            weight_decay=settings["weight_decay"],
        )
        random_generator = np.random.default_rng(seed)
        best, stale_epochs, trace = None, 0, []
        finite_gradients, nonzero_adapter_gradients = True, False
        for epoch in range(1, settings["max_epochs"] + 1):
            model.train()
            batches = []
            for protein_id, protein_group in groups.items():
                local_indices = random_generator.permutation(
                    np.flatnonzero(np.isin(protein_group["indices"], roles["train"]))
                )
                batches.extend(
                    (protein_id, local_indices[j : j + settings["batch_size"]])
                    for j in range(0, len(local_indices), settings["batch_size"])
                )
            random_generator.shuffle(batches)
            total_loss = 0.0
            for protein_id, batch_indices in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = (
                    (
                        call(model, groups[protein_id], batch_indices)
                        - targets[groups[protein_id]["indices"][batch_indices]]
                    )
                    .square()
                    .mean()
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite M15 loss")
                loss.backward()
                for name, p in trainable:
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        finite_gradients = False
                        raise ValueError("Missing/nonfinite gradient: " + name)
                    if not name.startswith("head.") and torch.count_nonzero(p.grad):
                        nonzero_adapter_gradients = True
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 5.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_indices)
            validation = macro_score(
                rows,
                roles["validation"],
                predict(model, groups, roles["validation"], settings["batch_size"]),
            )
            trace.append(
                {
                    "epoch": epoch,
                    "online_training_mse": total_loss / len(roles["train"]),
                    "validation_macro_spearman": validation,
                }
            )
            if best is None or validation > best[0]:
                best = validation, epoch, model.selected_state()
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= settings["patience"]:
                break
        if model.frozen_digest() != frozen_before:
            raise ValueError("Original PLM parameters changed")
        if mode != "frozen_head" and not nonzero_adapter_gradients:
            raise ValueError("Adapters received no nonzero gradients")
        candidates.append(
            {
                "learning_rate": rate,
                "epoch": best[1],
                "validation_score": best[0],
                "trace": trace,
                "frozen_before": frozen_before,
                "frozen_after": model.frozen_digest(),
                "finite_gradients": finite_gradients,
                "nonzero_adapter_gradients": nonzero_adapter_gradients,
            }
        )
        if winner is None or best[0] > winner[0]:
            winner = best[0], best[1], best[2], rate
    model = AdaptedESM(backbone, config["lora"], settings["hidden"], mode, seed)
    model.load_selected(winner[2])
    selection = {
        "learning_rate": winner[3],
        "epoch": winner[1],
        "validation_score": winner[0],
        "train_score": macro_score(
            rows, roles["train"], predict(model, groups, roles["train"], settings["batch_size"])
        ),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_parameters": sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("head.")
        ),
        "candidates": candidates,
        "state_tensor_sha256": tensor_digest(winner[2]),
    }
    return predict(model, groups, roles["test"], settings["batch_size"]), selection, winner[2]


def prepare(config, acquire=False):
    verify_file(config["m12_config"]["path"], config["m12_config"])
    m12 = json.loads(Path(config["m12_config"]["path"]).read_text())
    for path, digest in config["checkpoint_files"].items():
        if checksum(path) != digest:
            raise ValueError("ESM checkpoint/tokenizer hash changed")
    encoder = FrozenESM(config["embedding"], local_only=True)
    rows, matrices, mask, groups, evidence = inputs(m12, prepare=acquire)
    if not mask.all():
        raise ValueError("M15 cohort differs from original structure eligibility")
    frozen = np.zeros((len(rows), 961), np.float32)
    provenance = {
        "m12": evidence,
        "prefixes": {},
        "full_backbone_before": tensor_digest(dict(encoder.model.named_parameters())),
        "encoder": encoder.info,
        "checkpoint_files": config["checkpoint_files"],
    }
    zero = AdaptedESM(encoder.model, config["lora"], config["training"]["hidden"], "lora", 0).eval()
    for spec in m12["proteins"]:
        pid = spec["protein_id"]
        g = groups[pid]
        part = [rows[i] for i in g["indices"]]
        prefix, masks, manifest, directory = prefix_cache(
            encoder, part, spec["reference"]["sequence"], config, acquire
        )
        g.update(prefix=prefix, mutation_mask=masks)
        atoms, atom_provenance = mapped_atoms(spec["structure_artifacts"], g["graph"])
        g["edge_features"] = annotate(g["graph"], atoms)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(part), config["training"]["batch_size"]):
                ix = np.arange(start, min(start + config["training"]["batch_size"], len(part)))
                chunks.append(zero.features(**batch_inputs(g, ix)).numpy())
        features = np.concatenate(chunks)
        frozen[g["indices"]] = features
        g["frozen"] = features
        old = matrices["esm_sites"][g["indices"]]
        provenance["prefixes"][pid] = {
            "manifest": manifest,
            "directory": directory,
            "atom_provenance": atom_provenance,
            "historical_frozen_cache_max_abs_difference": float(abs(features - old).max()),
        }
    matrices["esm_sites"] = frozen
    return encoder, rows, matrices, groups, provenance, folds(rows, m12["family_order"])


def run(config, acquire=False):
    canonical = json.loads(Path("configs/m15.json").read_text())
    if config | {"output": canonical["output"]} != canonical:
        raise ValueError("Prespecified M15 config changed")
    output = Path(config["output"])
    if Path("runs").resolve() != output.resolve().parent:
        raise ValueError("M15 output must be a new direct child of runs/")
    cache = Path(config["cache"]) / "weights" / output.name
    if (output.exists() and any(output.iterdir())) or (cache.exists() and any(cache.iterdir())):
        raise ValueError("Existing experiment/checkpoint directory; use a unique output name")
    before = history()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder, rows, matrices, groups, provenance, partitions = prepare(config, acquire)
    records, predictions, role_records = [], [], []
    for fraction in config["fractions"]:
        for protocol, fold, original in partitions:
            roles = subset(rows, original, fraction, config["subset_salt"])
            audit(rows, roles, protocol)
            role_records.append(
                {
                    "fraction": fraction,
                    "protocol": protocol,
                    "fold": fold,
                    "original_roles": {
                        k: [rows[i]["id"] for i in ix] for k, ix in original.items()
                    },
                    "roles": {k: [rows[i]["id"] for i in ix] for k, ix in roles.items()},
                }
            )
            for mode in config["models"]:
                for seed in [0] if mode.endswith("_ridge") else config["seeds"]:
                    start = time.perf_counter()
                    key = f"{fraction}_{protocol}_{fold}_{mode}_{seed}"
                    if mode.endswith("_ridge"):
                        x = matrices["esm_sites" if mode == "frozen_ridge" else "structural"]
                        _, selection, state = ridge_ensemble(
                            x, rows, roles, config["alphas"], 0, seed
                        )
                        pred = ridge_predict(x, state)[0, roles["test"]]
                        path = cache / (key + ".npz")
                        np.savez_compressed(path, **state)
                    elif mode in ("original_gnn", "edge_full"):
                        pred, selection, state = graph_fit(
                            groups, rows, roles, config["structural_training"], seed, mode, 314159
                        )
                        path = cache / (key + ".pt")
                        torch.save(state, path)
                    else:
                        pred, selection, state = fit(
                            encoder.model, groups, rows, roles, config, seed, mode
                        )
                        path = cache / (key + ".pt")
                        torch.save(state, path)
                    record = {
                        "fraction": fraction,
                        "protocol": protocol,
                        "fold": fold,
                        "model": mode,
                        "seed": seed,
                        "seconds": time.perf_counter() - start,
                        "selection": selection,
                        "per_protein": per_protein(rows, roles["test"], pred),
                        "state_path": str(path),
                        "state_sha256": checksum(path),
                    }
                    records.append(record)
                    for i, p in zip(roles["test"], pred, strict=True):
                        r = rows[i]
                        predictions.append(
                            {
                                "fraction": fraction,
                                "protocol": protocol,
                                "fold": fold,
                                "model": mode,
                                "seed": seed,
                                "id": r["id"],
                                "protein_id": r["protein_id"],
                                "family": r["family"],
                                "n_mutations": len(r["mutations"]),
                                "target": r["target"],
                                "prediction": float(p),
                            }
                        )
                    print(f"{key}: {record['seconds']:.2f}s", flush=True)
                    write_json(output / "metrics.partial.json", records)
    provenance["full_backbone_after"] = tensor_digest(dict(encoder.model.named_parameters()))
    assert provenance["full_backbone_before"] == provenance["full_backbone_after"]
    assert before == history()
    write_json(output / "metrics.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_json(
        output / "audit.json",
        {
            "config": config,
            "rows": rows,
            "roles": role_records,
            "provenance": provenance,
            "historical_sha256": before,
            "seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "device": "cpu",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                n: importlib.metadata.version(n)
                for n in ("torch", "transformers", "numpy", "scipy", "scikit-learn")
            },
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "sources": {str(p): checksum(p) for p in Path("src/proteinmod").glob("*.py")},
        },
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m15.json")
    parser.add_argument("--output")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.output:
        config["output"] = args.output
    run(config, args.prepare)


if __name__ == "__main__":
    main()
