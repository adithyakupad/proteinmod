"""Fixed M4/M6 folds; task-balanced progressive training, validation-only selection."""

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
from proteinmod.hybrid_experiments import inputs as function_inputs
from proteinmod.multiprotein_data import audit_roles, protocols, similarity_audit
from proteinmod.multiprotein_experiments import macro_score, per_protein
from proteinmod.multitask import TASKS, MultitaskModel, calibrate, masked_loss
from proteinmod.residue_cache import residue_cache
from proteinmod.stability_data import load_stability, stability_folds
from proteinmod.stability_experiments import scores
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import write_csv


def history():
    return {
        str(p): checksum(p)
        for p in sorted(Path("benchmarks").rglob("*"))
        if p.is_file() and "m7" not in p.parts
    }


def output_path(path):
    output = Path(path).resolve()
    protected = [p.resolve() for p in Path("benchmarks").iterdir() if p.name != "m7"]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("M7 output would overwrite historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output


def inputs(config):
    m5 = json.loads(Path(config["function_config"]).read_text())
    m4 = json.loads(Path(m5["m4_config"]).read_text())
    m6 = json.loads(Path(config["stability_config"]).read_text())
    fr, valid, fg, fa, fp = function_inputs(m5, m4)
    sr, sa = load_stability(m6)
    sg, sp = {}, {}
    for spec in m6["proteins"]:
        pid = spec["protein_id"]
        ix = np.array([i for i, r in enumerate(sr) if r["protein_id"] == pid])
        part = [sr[i] for i in ix]
        graph = load_graph(spec["structure_artifacts"], spec["reference"]["sequence"])
        tokens, manifest, directory = residue_cache(
            {"reference": spec["reference"], "dataset": m6["sources"]["measurements"]},
            part,
            m6["embedding"],
            m6["residue_cache"],
        )
        dynamic = []
        for r in part:
            d, _, reasons = variant_inputs(r["mutations"], spec["reference"]["sequence"], graph)
            if reasons:
                raise ValueError("M6 mapped cohort changed")
            dynamic.append(d)
        sg[pid] = {
            "indices": ix,
            "graph": graph,
            "dynamic": np.array(dynamic),
            "residues": tokens,
            "token_indices": np.arange(len(ix)),
        }
        sp[pid] = {
            "manifest_sha256": checksum(directory / "manifest.json"),
            "encoder": manifest["encoder"],
            "graph": graph["audit"],
        }
    ff = {
        fold: {p: ix[valid[ix]] for p, ix in roles.items()}
        for _, fold, roles in protocols(fr, m4["experiment"]["fold_order"])
    }
    sf = {fold: roles for _, fold, roles in stability_folds(sr, m6)}
    archived_f = json.loads(Path("benchmarks/m5/audit.json").read_text())["roles"]
    archived_s = json.loads(Path("benchmarks/m6/audit.json").read_text())["roles"]
    evidence = []
    for f, s in config["fold_pairs"]:
        roles = {"function": ff[f], "stability": sf[s]}
        for task, rows, archived, fold in (
            ("function", fr, archived_f, f),
            ("stability", sr, archived_s, s),
        ):
            ids = {p: [rows[i]["id"] for i in ix] for p, ix in roles[task].items()}
            old = next(r for r in archived if r["fold"] == fold)
            if ids != old["ids"]:
                raise ValueError("Historical evaluation IDs changed")
        audit_combined({"function": fr, "stability": sr}, roles, f != "within")
        evidence.append(
            {
                "function_fold": f,
                "stability_fold": s,
                "ids": {
                    t: {p: [rows[i]["id"] for i in ix] for p, ix in roles[t].items()}
                    for t, rows in (("function", fr), ("stability", sr))
                },
            }
        )
    similarity = similarity_audit(m4["collection"] + m6["proteins"])
    # Reject close cross-task homologs, rather than silently leak labels through an auxiliary task.
    functional = {s["protein_id"] for s in m4["collection"]}
    for p in similarity:
        if (
            (p["left"] in functional) != (p["right"] in functional)
            and p["aligned_identity"] >= 0.3
            and min(p["left_coverage"], p["right_coverage"]) >= 0.8
        ):
            raise ValueError("Cross-task near-homolog requires grouped partitioning")
    return (
        {"function": fr, "stability": sr},
        {"function": fg, "stability": sg},
        [(f, s, {"function": ff[f], "stability": sf[s]}) for f, s in config["fold_pairs"]],
        {
            "function": fa,
            "stability": sa,
            "function_tokens": fp,
            "stability_tokens": sp,
            "roles": evidence,
            "similarity": similarity,
            "function_exclusions": [r["id"] for i, r in enumerate(fr) if not valid[i]],
            "paired_examples": 0,
            "initialization": config["initialization"],
        },
    )


def audit_combined(rows, roles, held_out):
    for task in TASKS:
        audit_roles(rows[task], roles[task], held_out=held_out)
    for key in ("protein_id", "sequence"):
        sets = {
            p: {r[key] for t in TASKS for i in roles[t][p] for r in [rows[t][i]]}
            for p in ("train", "validation", "test")
        }
        if (key == "sequence" or held_out) and any(
            sets[a] & sets[b]
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError("Cross-task protein/sequence leakage")


def fit(rows, groups, roles, settings, seed, mode, weight=1.0):
    if settings["device"] != "cpu" or mode not in (*TASKS, "frozen", "joint"):
        raise ValueError("Deterministic CPU and known mode required")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = MultitaskModel(
        next(iter(groups["function"].values()))["graph"],
        embedding_dimension=next(iter(groups["function"].values()))["residues"].shape[-1],
        dimension=settings["dimension"],
        heads=settings["heads"],
        layers=settings["layers"],
    )
    active = (mode,) if mode in TASKS else TASKS
    for task in TASKS:
        model.heads[task].requires_grad_(task in active)
    target, calibration = {}, {}
    for task in TASKS:
        y, calibration[task] = calibrate(rows[task], roles[task]["train"], task)
        target[task] = torch.tensor(y, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = {t: np.random.default_rng(seed) for t in TASKS}
    prepared = {}
    for task in TASKS:
        for pid, group in groups[task].items():
            g = group["graph"]
            edge = g["edge_index"]
            topology = (
                torch.as_tensor(g["static"]),
                torch.as_tensor(edge[0]),
                torch.as_tensor(edge[1]),
                torch.as_tensor(g["edge_features"]),
                torch.tensor(
                    np.maximum(np.bincount(edge[1], minlength=len(g["static"])), 1),
                    dtype=torch.float32,
                )[None, :, None],
            )
            mutation = np.zeros(group["residues"].shape[:2], dtype=bool)
            for local, index in enumerate(group["indices"]):
                for token in rows[task][index]["mutations"]:
                    mutation[group["token_indices"][local], int(token[1:-1]) - 1] = True
            prepared[task, pid] = (
                torch.as_tensor(group["dynamic"]),
                topology,
                torch.tensor([n["reference_position"] - 1 for n in g["nodes"]]),
                mutation,
            )

    def batches(task, indices, shuffle=False):
        out = []
        for pid, group in groups[task].items():
            local = np.flatnonzero(np.isin(group["indices"], indices))
            if shuffle:
                local = rng[task].permutation(local)
            out.extend(
                (pid, local[i : i + settings["batch_size"]])
                for i in range(0, len(local), settings["batch_size"])
            )
        if shuffle:
            rng[task].shuffle(out)
        return out

    def call(task, pid, local):
        group = groups[task][pid]
        dynamic, topology, positions, mutation = prepared[task, pid]
        ix = group["token_indices"][local]
        return model(
            torch.from_numpy(np.array(group["residues"][ix], copy=True)),
            dynamic[local],
            topology,
            positions,
            torch.tensor(mutation[ix]),
        )

    def predict(task, indices):
        model.eval()
        found = {}
        with torch.no_grad():
            for pid, local in batches(task, indices):
                values = call(task, pid, local)[:, TASKS.index(task)].double().numpy()
                values = values * calibration[task]["scale"] + calibration[task]["mean"]
                found.update(zip(groups[task][pid]["indices"][local], values, strict=True))
        return np.array([found[i] for i in indices])

    def validation():
        result = {}
        for task in active:
            ix = roles[task]["validation"]
            pred = predict(task, ix)
            result[task] = (
                (1 - macro_score(rows[task], ix, pred)) / 2
                if task == "function"
                else scores(rows[task], ix, pred)["macro"]["rmse"] / calibration[task]["scale"]
            )
        return float(np.mean(list(result.values()))), result

    steps = max(len(batches(t, roles[t]["train"])) for t in TASKS)
    best, trace, gradient = None, [], None
    exposures = {t: 0 for t in active}
    started = time.perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        train_encoder = mode != "frozen" and epoch > settings["warmup_epochs"]
        model.encoder.backbone.requires_grad_(train_encoder)
        model.train()
        task_batches = {t: batches(t, roles[t]["train"], True) for t in active}
        total = 0.0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for task in active:
                pid, local = task_batches[task][step % len(task_batches[task])]
                output = call(task, pid, local)
                y = torch.full_like(output, float("nan"))
                mask = torch.zeros_like(output, dtype=torch.bool)
                j = TASKS.index(task)
                y[:, j] = target[task][groups[task][pid]["indices"][local]]
                mask[:, j] = True
                losses.append(masked_loss(output, y, mask))
                exposures[task] += len(local)
            if train_encoder and len(active) == 2 and gradient is None:
                common = list(model.encoder.backbone.parameters())
                vectors = [
                    torch.cat(
                        [g.flatten() for g in torch.autograd.grad(loss, common, retain_graph=True)]
                    )
                    for loss in losses
                ]
                gradient = {
                    "epoch": epoch,
                    "norms": [float(v.norm()) for v in vectors],
                    "cosine": float(
                        torch.nn.functional.cosine_similarity(vectors[0][None], vectors[1][None])
                    ),
                }
            weights = [weight if t == "stability" else 1.0 for t in active]
            loss = sum(w * loss for w, loss in zip(weights, losses, strict=True)) / sum(weights)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        score, task_scores = validation()
        trace.append(
            {
                "epoch": epoch,
                "encoder_trainable": train_encoder,
                "online_normalized_loss": total / steps,
                "validation": task_scores,
                "selection_score": score,
                "last_gradient_norm": float(norm),
            }
        )
        if best is None or score < best[0]:
            best = score, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best[2])
    predictions = {t: predict(t, roles[t]["test"]) for t in active}
    train_metrics = {
        t: (
            per_protein(rows[t], roles[t]["train"], predict(t, roles[t]["train"]))
            if t == "function"
            else scores(rows[t], roles[t]["train"], predict(t, roles[t]["train"]))
        )
        for t in active
    }
    return (
        predictions,
        {
            "epoch": best[1],
            "score": best[0],
            "trace": trace,
            "calibration": calibration,
            "examples_processed": exposures,
            "steps_per_epoch": steps,
            "gradient": gradient,
            "train": train_metrics,
            "parameters": sum(p.numel() for p in model.parameters()),
            "encoder_parameters": sum(p.numel() for p in model.encoder.backbone.parameters()),
            "elapsed_seconds": time.perf_counter() - started,
        },
        best[2],
    )


def run(config):
    before = history()
    output = output_path(config["output"])
    started = time.perf_counter()
    rows, groups, folds, audit = inputs(config)
    records, predictions = [], []
    cache_identity = {
        "config": config,
        "historical_data": before,
        "sources": {str(p): checksum(p) for p in Path("src/proteinmod").glob("*.py")},
    }
    cache = Path("data/cache/m7/checkpoints") / fingerprint(cache_identity)
    cache.mkdir(parents=True, exist_ok=True)
    for f, s, roles in folds:
        for seed in config["seeds"]:
            candidates = []
            for mode, weight in [("function", 1), ("stability", 1), ("frozen", 1)] + [
                ("joint", w) for w in config["stability_weights"]
            ]:
                name = f"{f}-{s}-{mode}-{weight}-{seed}"
                pred, selection, state = fit(
                    rows, groups, roles, config["training"], seed, mode, weight
                )
                torch.save(
                    {
                        "schema": config["schema"],
                        "state": state,
                        "calibration": selection["calibration"],
                        "config": config,
                        "selection": selection,
                        "task_order": TASKS,
                    },
                    cache / (name + ".pt"),
                )
                record = {
                    "function_fold": f,
                    "stability_fold": s,
                    "seed": seed,
                    "mode": mode,
                    "stability_weight": weight,
                    "selection": selection,
                    "checkpoint_sha256": checksum(cache / (name + ".pt")),
                    "test": {
                        t: (
                            per_protein(rows[t], roles[t]["test"], p)
                            if t == "function"
                            else scores(rows[t], roles[t]["test"], p)
                        )
                        for t, p in pred.items()
                    },
                }
                records.append(record)
                if mode == "joint":
                    candidates.append((selection["score"], weight, pred, record))
                label = "independent" if mode in TASKS else mode
                for task, values in pred.items():
                    predictions.extend(
                        prediction_rows(
                            rows[task], roles[task]["test"], values, f, s, seed, label, weight, task
                        )
                    )
                print(name, "validation", round(selection["score"], 5), flush=True)
            _, weight, chosen, record = min(candidates, key=lambda x: (x[0], x[1]))
            records.append({**record, "mode": "selected_joint"})
            for task, values in chosen.items():
                predictions.extend(
                    prediction_rows(
                        rows[task],
                        roles[task]["test"],
                        values,
                        f,
                        s,
                        seed,
                        "selected_joint",
                        weight,
                        task,
                    )
                )
            write_json(output / "runs.json", records)
    write_csv(output / "predictions.csv", predictions)
    if history() != before:
        raise ValueError("Historical artifacts changed")
    write_json(output / "audit.json", {**audit, "history": before})
    write_json(output / "rows.json", rows)
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "config_sha256": fingerprint(config),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "device": "cpu",
            "python": platform.python_version(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "scikit-learn", "biopython")
            },
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_hashes": {str(p): checksum(p) for p in Path("src/proteinmod").glob("*.py")},
            "files": {str(p.name): checksum(p) for p in output.iterdir() if p.is_file()},
        },
    )


def prediction_rows(rows, ix, values, f, s, seed, model, weight, task):
    return [
        {
            "function_fold": f,
            "stability_fold": s,
            "task": task,
            "model": model,
            "stability_weight": weight,
            "seed": seed,
            "id": rows[i]["id"],
            "protein_id": rows[i]["protein_id"],
            "assay_id": rows[i]["assay_id"],
            "measured": rows[i]["target"],
            "prediction": float(p),
            "unit": "kcal/mol"
            if task == "stability"
            else "assay-native measured; prediction train-rank",
            "sign": "positive_destabilizing" if task == "stability" else "assay-defined",
        }
        for i, p in zip(ix, values, strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/m7.json")
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
