"""G1 within-protein and G2 protein-held-out regression with assay-local rank targets."""

import copy
import importlib.metadata
import platform
import resource
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from proteinmod.data import checksum, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.models import metrics, one_hot
from proteinmod.multiprotein_data import (
    audit_roles,
    collection_output,
    model_inputs,
    protocols,
    similarity_audit,
    train_percentiles,
)
from proteinmod.structure_pipeline import write_csv


def historical_hashes():
    return {
        str(p): checksum(p)
        for directory in (
            "benchmarks/amylase",
            "benchmarks/amylase_m1",
            "benchmarks/structures",
            "benchmarks/amylase_m3",
        )
        for p in sorted(Path(directory).rglob("*"))
        if p.is_file()
    }


def per_protein(rows, indices, predictions):
    grouped = defaultdict(list)
    for i, pred in zip(indices, predictions, strict=True):
        grouped[rows[i]["protein_id"]].append((rows[i]["target"], float(pred)))
    return {
        p: {
            "n": len(values),
            "spearman_rho": metrics(np.array(values)[:, 0], np.array(values)[:, 1])["spearman_rho"],
        }
        for p, values in grouped.items()
    }


def macro_score(rows, indices, predictions):
    scores = [r["spearman_rho"] for r in per_protein(rows, indices, predictions).values()]
    # Undefined constant-prediction correlations cannot win validation selection.
    return float(np.mean([s if s is not None else 0 for s in scores]))


def fit_targets(rows, train):
    target = np.full(len(rows), np.nan)
    for assay in {rows[i]["assay_id"] for i in train}:
        ix = [i for i in train if rows[i]["assay_id"] == assay]
        target[ix] = train_percentiles([rows[i]["target"] for i in ix])
    return target


def ridge_fit(matrix, rows, roles, alphas):
    target = fit_targets(rows, roles["train"])
    scaler = StandardScaler(with_mean=not hasattr(matrix, "tocsr")).fit(matrix[roles["train"]])
    x = scaler.transform(matrix)
    selected, candidates = None, []
    for alpha in sorted(alphas):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000).fit(
            x[roles["train"]], target[roles["train"]]
        )
        score = macro_score(rows, roles["validation"], model.predict(x[roles["validation"]]))
        candidates.append({"alpha": alpha, "validation_macro_spearman": score})
        if selected is None or score > selected[0]:
            selected = score, model, alpha
    model = selected[1]
    return model.predict(x[roles["test"]]), {
        "alpha": selected[2],
        "candidates": candidates,
        "train_macro_spearman": macro_score(rows, roles["train"], model.predict(x[roles["train"]])),
        "validation_macro_spearman": selected[0],
        "target_handling": "Empirical midrank percentile per training assay; no held-out label normalization",
    }


def graph_fit(groups, rows, roles, settings, seed, threads):
    import torch

    from proteinmod.structural_gnn import ResidueGNN

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if settings["device"] != "cpu":
        raise ValueError("Deterministic M4 GNN requires CPU; MPS index aggregation is unsupported")
    first = next(iter(groups.values()))["graph"]
    model = ResidueGNN(
        first["static"],
        first["edge_index"],
        first["edge_features"],
        settings["hidden"],
        settings["layers"],
    )
    inputs, topologies = {}, {}
    for p, group in groups.items():
        g = group["graph"]
        inputs[p] = torch.as_tensor(group["dynamic"], dtype=torch.float32)
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
    target = fit_targets(rows, roles["train"])
    y = torch.as_tensor(target, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(seed)
    batch_size = settings["batch_size"]

    def batches(indices, shuffle=False):
        batches = []
        for p, g in groups.items():
            local = np.flatnonzero(np.isin(g["indices"], indices))
            if shuffle:
                local = rng.permutation(local)
            batches.extend((p, local[j : j + batch_size]) for j in range(0, len(local), batch_size))
        if shuffle:
            rng.shuffle(batches)
        return batches

    def predict(indices):
        model.eval()
        found = {}
        with torch.no_grad():
            for p, local in batches(indices):
                values = model(inputs[p][local], topologies[p]).numpy()
                found.update(zip(groups[p]["indices"][local], values, strict=True))
        return np.array([found[i] for i in indices], dtype=float)

    best, stale, trace = None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        for p, local in batches(roles["train"], shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(
                (model(inputs[p][local], topologies[p]) - y[groups[p]["indices"][local]]) ** 2
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cross-protein training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        score = macro_score(rows, roles["validation"], predict(roles["validation"]))
        trace.append({"epoch": epoch, "validation_macro_spearman": score})
        if best is None or score > best[0]:
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
            "validation_macro_spearman": best[0],
            "trace": trace,
            "train_macro_spearman": macro_score(rows, roles["train"], predict(roles["train"])),
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": "cpu",
        },
        model.state_dict(),
    )


def run_collection(config):
    started = time.perf_counter()
    before = historical_hashes()
    output = collection_output(config)
    rows, features, mask, groups, data_audit = model_inputs(config)
    records, predictions, role_audits = [], [], []
    folds = protocols(rows, config["experiment"]["fold_order"])
    for protocol, fold, original in folds:
        for cohort in ("sequence_all", "structure_matched"):
            roles = {
                p: ix if cohort == "sequence_all" else ix[mask[ix]] for p, ix in original.items()
            }
            audit_roles(rows, roles, held_out=protocol == "G2")
            role_audits.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "cohort": cohort,
                    "ids": {p: [rows[i]["id"] for i in ix] for p, ix in roles.items()},
                    "proteins": {
                        p: sorted({rows[i]["protein_id"] for i in ix}) for p, ix in roles.items()
                    },
                }
            )
            models = ["mutation_count_ridge", "sequence_ridge", "esm_sites"] + (
                ["structural_ridge", "structural_gnn"] if cohort == "structure_matched" else []
            )
            for name in models:
                seeds = config["experiment"]["seeds"] if name == "structural_gnn" else [0]
                for seed in seeds:
                    fit_started = time.perf_counter()
                    if name == "structural_gnn":
                        pred, selection, state = graph_fit(
                            groups,
                            rows,
                            roles,
                            config["gnn"],
                            seed,
                            config["experiment"]["cpu_threads"],
                        )
                        import torch

                        cache = Path("data/cache/m4/checkpoints") / fingerprint(config)
                        cache.mkdir(parents=True, exist_ok=True)
                        torch.save(state, cache / f"{protocol}-{fold}-{seed}.pt")
                    else:
                        pred, selection = ridge_fit(
                            features[name], rows, roles, config["experiment"]["alphas"]
                        )
                    scores = per_protein(rows, roles["test"], pred)
                    record = {
                        "protocol": protocol,
                        "fold": fold,
                        "cohort": cohort,
                        "model": name,
                        "seed": seed,
                        "n_train": len(roles["train"]),
                        "n_validation": len(roles["validation"]),
                        "test": scores,
                        "macro_spearman": macro_score(rows, roles["test"], pred),
                        "selection": selection,
                        "elapsed_seconds": time.perf_counter() - fit_started,
                    }
                    records.append(record)
                    predictions.extend(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "cohort": cohort,
                            "model": name,
                            "seed": seed,
                            "id": rows[i]["id"],
                            "protein_id": rows[i]["protein_id"],
                            "assay_id": rows[i]["assay_id"],
                            "measured": rows[i]["target"],
                            "prediction": float(value),
                        }
                        for i, value in zip(roles["test"], pred, strict=True)
                    )
                    print(
                        f"{protocol} {fold} {cohort} {name} seed={seed} macro_rho={record['macro_spearman']:.4f}",
                        flush=True,
                    )
                    write_json(output / "runs.json", records)
            if protocol == "G1":
                for p in config["experiment"]["fold_order"]:
                    local = {
                        part: ix[[rows[i]["protein_id"] == p for i in ix]]
                        for part, ix in roles.items()
                    }
                    indices = np.concatenate(list(local.values()))
                    matrix = one_hot([rows[i]["sequence"] for i in indices])
                    remap = {int(i): j for j, i in enumerate(indices)}
                    local_rows = [rows[i] for i in indices]
                    local_roles = {
                        part: np.array([remap[int(i)] for i in ix]) for part, ix in local.items()
                    }
                    pred, selection = ridge_fit(
                        matrix, local_rows, local_roles, config["experiment"]["alphas"]
                    )
                    records.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "cohort": cohort,
                            "model": "within_one_hot",
                            "seed": 0,
                            "test": per_protein(rows, local["test"], pred),
                            "selection": selection,
                            "macro_spearman": macro_score(rows, local["test"], pred),
                            "n_train": len(local["train"]),
                            "n_validation": len(local["validation"]),
                        }
                    )
                    predictions.extend(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "cohort": cohort,
                            "model": "within_one_hot",
                            "seed": 0,
                            "id": rows[i]["id"],
                            "protein_id": p,
                            "assay_id": rows[i]["assay_id"],
                            "measured": rows[i]["target"],
                            "prediction": float(value),
                        }
                        for i, value in zip(local["test"], pred, strict=True)
                    )
    if before != historical_hashes():
        raise ValueError("M0-M3 history changed")
    write_json(output / "runs.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_json(
        output / "audit.json",
        {
            "data": data_audit,
            "roles": role_audits,
            "sequence_similarity": similarity_audit(config["collection"]),
            "historical_sha256": before,
            "family_generalization_verified": False,
            "target_policy": "Training-assay midrank percentiles only; selection and evaluation per-protein Spearman; no cross-assay raw RMSE",
            "structural_mask": [
                {
                    "id": r["id"],
                    "valid": r["structural_valid"],
                    "reasons": r["structural_exclusions"],
                }
                for r in rows
            ],
        },
    )
    versions = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "scikit-learn", "torch", "biopython")
    }
    try:
        versions["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        versions["transformers"] = "not installed; using cached frozen features"
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": versions,
            "source_sha256": {p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")},
            "historical_files_unchanged": len(before),
            "artifacts_sha256": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    return records
