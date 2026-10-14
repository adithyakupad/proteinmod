"""M5: fixed M4 roles, training-only rank targets, validation-only epoch selection."""

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

from proteinmod.data import checksum, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.hybrid import MODES, HybridRegressor
from proteinmod.multiprotein_data import audit_roles, collection_rows, model_inputs, protocols
from proteinmod.multiprotein_experiments import (
    fit_targets,
    historical_hashes,
    macro_score,
    per_protein,
)
from proteinmod.residue_cache import prepare, residue_cache
from proteinmod.structure_pipeline import write_csv


def history():
    return historical_hashes() | {
        str(p): checksum(p) for p in Path("benchmarks/m4").rglob("*") if p.is_file()
    }


def output_path(config):
    output = Path(config["output"])
    for name in (
        "benchmarks/amylase",
        "benchmarks/amylase_m1",
        "benchmarks/structures",
        "benchmarks/amylase_m3",
        "benchmarks/m4",
    ):
        protected = Path(name).resolve()
        if (
            output.resolve() == protected
            or output.resolve() in protected.parents
            or protected in output.resolve().parents
        ):
            raise ValueError("M5 output would overwrite historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output


def inputs(config, m4):
    rows, _, mask, groups, audit = model_inputs(m4)
    provenance = {}
    for spec in m4["collection"]:
        p = spec["protein_id"]
        part, _ = collection_rows(spec, m4["experiment"]["cap_per_partition"])
        tokens, manifest, directory = residue_cache(
            spec, part, m4["embedding"], config["residue_cache"]
        )
        offset = next(i for i, r in enumerate(rows) if r["protein_id"] == p)
        group = groups[p]
        group["residues"] = tokens
        group["token_indices"] = group["indices"] - offset
        provenance[p] = {
            "manifest": manifest,
            "directory": str(directory),
            "manifest_sha256": checksum(directory / "manifest.json"),
        }
    return rows, mask, groups, audit, provenance


def fit(groups, rows, roles, settings, seed, mode):
    audit_roles(
        rows,
        roles,
        held_out=len({rows[i]["protein_id"] for i in roles["test"]}) == 1
        and not {rows[i]["protein_id"] for i in roles["test"]}
        & {rows[i]["protein_id"] for i in roles["train"]},
    )
    if settings["device"] != "cpu" or any(
        settings[k] < 1
        for k in ("dimension", "heads", "max_epochs", "batch_size", "patience", "cpu_threads")
    ):
        raise ValueError("Positive settings and deterministic CPU required")
    if (
        settings["layers"] not in (2, 3)
        or settings["learning_rate"] <= 0
        or settings["weight_decay"] < 0
    ):
        raise ValueError("Invalid training settings")
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    first = next(iter(groups.values()))["graph"]
    model = HybridRegressor(
        first,
        next(iter(groups.values()))["residues"].shape[-1],
        settings["dimension"],
        settings["heads"],
        settings["layers"],
        mode,
    )
    topologies, dynamics, positions = {}, {}, {}
    for p, group in groups.items():
        g = group["graph"]
        topologies[p] = (
            torch.as_tensor(g["static"]),
            torch.as_tensor(g["edge_index"][0]),
            torch.as_tensor(g["edge_index"][1]),
            torch.as_tensor(g["edge_features"]),
            torch.as_tensor(
                np.maximum(np.bincount(g["edge_index"][1], minlength=len(g["static"])), 1),
                dtype=torch.float32,
            )[None, :, None],
        )
        dynamics[p] = torch.as_tensor(group["dynamic"])
        positions[p] = torch.tensor([r["reference_position"] - 1 for r in g["nodes"]])
    y = torch.tensor(fit_targets(rows, roles["train"]), dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(seed)

    def batches(indices, shuffle=False):
        result = []
        for p, group in groups.items():
            local = np.flatnonzero(np.isin(group["indices"], indices))
            if shuffle:
                local = rng.permutation(local)
            result.extend(
                (p, local[j : j + settings["batch_size"]])
                for j in range(0, len(local), settings["batch_size"])
            )
        if shuffle:
            rng.shuffle(result)
        return result

    def call(p, local, diagnostic=False):
        group = groups[p]
        residues = torch.from_numpy(
            np.array(group["residues"][group["token_indices"][local]], copy=True)
        )
        return model(
            residues, dynamics[p][local], topologies[p], positions[p], diagnostics=diagnostic
        )

    def predict(indices):
        model.eval()
        found = {}
        with torch.no_grad():
            for p, local in batches(indices):
                found.update(zip(groups[p]["indices"][local], call(p, local).numpy(), strict=True))
        return np.array([found[i] for i in indices], dtype=float)

    best, stale, trace = None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        total = 0.0
        for p, local in batches(roles["train"], True):
            optimizer.zero_grad(set_to_none=True)
            loss = ((call(p, local) - y[groups[p]["indices"][local]]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite hybrid training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item() * len(local)
        score = macro_score(rows, roles["validation"], predict(roles["validation"]))
        trace.append(
            {
                "epoch": epoch,
                "train_mse_online": total / len(roles["train"]),
                "validation_macro_spearman": score,
            }
        )
        if best is None or score > best[0]:
            best = score, epoch, copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    predictions = predict(roles["test"])
    diagnostics = {}
    with torch.no_grad():
        for p, local in batches(roles["test"]):
            _, weights = call(p, local, True)
            for direction, w in weights.items():
                valid = w.sum(-1) > 0
                if valid.any():
                    v = w[valid]
                    entropy = -(v * v.clamp(min=1e-12).log()).sum(-1) / np.log(w.shape[-1])
                    record = diagnostics.setdefault(p, {}).setdefault(
                        direction, {"n_queries_heads": 0, "entropy_sum": 0.0, "peak_sum": 0.0}
                    )
                    record["n_queries_heads"] += len(v)
                    record["entropy_sum"] += float(entropy.sum())
                    record["peak_sum"] += float(v.max(-1).values.sum())
    selection = {
        "epoch": best[1],
        "validation_macro_spearman": best[0],
        "trace": trace,
        "train_macro_spearman": macro_score(rows, roles["train"], predict(roles["train"])),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": "cpu",
        "attention": diagnostics,
    }
    if mode == "weighted_fusion":
        selection["sequence_weight"] = float(model.weight.sigmoid().detach())
    return predictions, selection, best[2]


def run(config, m4):
    started = time.perf_counter()
    before = history()
    output = output_path(config)
    rows, mask, groups, audit, provenance = inputs(config, m4)
    roles_evidence, records, predictions = [], [], []
    archived = json.loads(Path("benchmarks/m4/audit.json").read_text())
    for protocol, fold, original in protocols(rows, m4["experiment"]["fold_order"]):
        roles = {role: ix[mask[ix]] for role, ix in original.items()}
        audit_roles(rows, roles, held_out=protocol == "G2")
        ids = {role: [rows[i]["id"] for i in ix] for role, ix in roles.items()}
        old = next(
            r
            for r in archived["roles"]
            if r["protocol"] == protocol
            and r["fold"] == fold
            and r["cohort"] == "structure_matched"
        )
        if ids != old["ids"]:
            raise ValueError("M5 roles differ from archived M4 roles")
        roles_evidence.append({"protocol": protocol, "fold": fold, "ids": ids})
        for mode in config["modes"]:
            if mode not in MODES:
                raise ValueError("Unknown model")
            for seed in config["seeds"]:
                fit_started = time.perf_counter()
                pred, selection, state = fit(groups, rows, roles, config["training"], seed, mode)
                cache = Path("data/cache/m5/checkpoints") / fingerprint(config)
                cache.mkdir(parents=True, exist_ok=True)
                torch.save(state, cache / f"{protocol}-{fold}-{mode}-{seed}.pt")
                record = {
                    "protocol": protocol,
                    "fold": fold,
                    "model": mode,
                    "seed": seed,
                    "test": per_protein(rows, roles["test"], pred),
                    "macro_spearman": macro_score(rows, roles["test"], pred),
                    "selection": selection,
                    "elapsed_seconds": time.perf_counter() - fit_started,
                }
                records.append(record)
                print(
                    protocol,
                    fold,
                    mode,
                    seed,
                    record["macro_spearman"],
                    record["elapsed_seconds"],
                    flush=True,
                )
                predictions.extend(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "model": mode,
                        "seed": seed,
                        "id": rows[i]["id"],
                        "protein_id": rows[i]["protein_id"],
                        "assay_id": rows[i]["assay_id"],
                        "measured": rows[i]["target"],
                        "prediction": float(v),
                        "mutation_count": len(rows[i]["mutations"]),
                    }
                    for i, v in zip(roles["test"], pred, strict=True)
                )
                write_json(output / "runs.json", records)
    if before != history():
        raise ValueError("M0-M4 historical artifacts changed")
    write_csv(output / "predictions.csv", predictions)
    write_json(
        output / "audit.json",
        {
            "roles": roles_evidence,
            "data": audit,
            "residue_caches": provenance,
            "historical_sha256": before,
            "matched_variants": int(mask.sum()),
            "excluded": int((~mask).sum()),
            "family_separation_verified": False,
        },
    )
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "m4_config_sha256": checksum(config["m4_config"]),
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
                for p in ("torch", "numpy", "scipy", "scikit-learn", "biopython")
            },
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
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", default="configs/m5.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    m4 = json.loads(Path(config["m4_config"]).read_text())
    (prepare if args.action == "prepare" else run)(config, m4)


if __name__ == "__main__":
    main()
