"""M14 fixed-cohort fresh fits; target-independent graphs, validation-only selection."""

import argparse
import copy
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import checksum, verify_file, write_json
from proteinmod.edge_chemistry import EDGE_NAMES, annotate, controlled_graph, mapped_atoms
from proteinmod.edge_gnn import EdgeGNN
from proteinmod.family_data import audit, folds
from proteinmod.family_experiments import inputs
from proteinmod.multiprotein_experiments import fit_targets, macro_score, per_protein
from proteinmod.pretrained_structure import FrozenProteinMPNN, site_probe
from proteinmod.stability_data import load_stability, stability_folds
from proteinmod.stability_experiments import scores, validate_task
from proteinmod.structural_gnn import ResidueGNN
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import write_csv
from proteinmod.uncertainty_models import ridge_ensemble, ridge_predict


def historical():
    return {
        str(p): checksum(p)
        for root in ("benchmarks", "configs")
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
        and not any(
            part.startswith("m")
            and part[1:].split(".")[0].isdigit()
            and int(part[1:].split(".")[0]) >= 14
            for part in p.parts
        )
    }


def topology(group, mode, seed):
    g = group["graph"]
    edge, feature = (
        (g["edge_index"], g["edge_features"])
        if mode == "original_gnn"
        else controlled_graph(g, group["edge_features"], mode, seed)
    )
    return (
        torch.tensor(g["static"], dtype=torch.float32),
        torch.tensor(edge[0]),
        torch.tensor(edge[1]),
        torch.tensor(feature, dtype=torch.float32),
        torch.tensor(
            np.maximum(np.bincount(edge[1], minlength=len(g["static"])), 1), dtype=torch.float32
        )[None, :, None],
    )


def make_model(group, mode, settings):
    g = group["graph"]
    if mode == "original_gnn":
        return ResidueGNN(
            g["static"], g["edge_index"], g["edge_features"], settings["hidden"], settings["layers"]
        )
    return EdgeGNN(
        g["static"].shape[1],
        len(EDGE_NAMES),
        settings["hidden"],
        settings["layers"],
        mode == "zero_message",
    )


def fit(groups, rows, roles, settings, seed, mode, controls_seed, native=False):
    if settings["device"] != "cpu" or any(
        settings[k] < 1 for k in ("hidden", "max_epochs", "patience", "batch_size", "cpu_threads")
    ):
        raise ValueError("Positive training settings and deterministic CPU required")
    if (
        settings["layers"] not in (2, 3)
        or settings["learning_rate"] <= 0
        or settings["weight_decay"] < 0
    ):
        raise ValueError("Invalid architecture/optimizer")
    if any(len(ix) == 0 for ix in roles.values()) or any(
        set(roles[a]) & set(roles[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Nonempty disjoint partitions required")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    first = next(iter(groups.values()))
    model = make_model(first, mode, settings)
    graphs = {p: topology(g, mode, controls_seed) for p, g in groups.items()}
    dynamics = {p: torch.tensor(g["dynamic"], dtype=torch.float32) for p, g in groups.items()}
    if native:
        validate_task(rows)
    targets = np.array([r["target"] for r in rows]) if native else fit_targets(rows, roles["train"])
    mean = float(targets[roles["train"]].mean()) if native else 0.0
    scale = max(float(targets[roles["train"]].std()), 1e-8) if native else 1.0
    y = torch.tensor((targets - mean) / scale, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

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

    def predict(indices):
        model.eval()
        values = {}
        with torch.no_grad():
            for p, ix in batches(indices):
                pred = model(dynamics[p][ix], graphs[p]).numpy().astype(float) * scale + mean
                values.update(zip(groups[p]["indices"][ix], pred, strict=True))
        return np.array([values[i] for i in indices])

    def score(indices):
        pred = predict(indices)
        return (
            -scores(rows, indices, pred)["macro"]["rmse"]
            if native
            else macro_score(rows, indices, pred)
        )

    best, stale, trace = None, 0, []
    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        total = 0.0
        for p, ix in batches(roles["train"], True):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(dynamics[p][ix], graphs[p]) - y[groups[p]["indices"][ix]]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(ix)
        validation = score(roles["validation"])
        trace.append(
            {
                "epoch": epoch,
                "online_training_mse": total / len(roles["train"]),
                "validation_score": validation,
            }
        )
        if best is None or validation > best[0]:
            best = validation, epoch, copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    selection = {
        "epoch": best[1],
        "validation_score": best[0],
        "train_score": score(roles["train"]),
        "score_type": "negative_native_rmse" if native else "macro_spearman",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trace": trace,
        "target_mean": mean,
        "target_scale": scale,
    }
    prediction = predict(roles["test"])
    # Reload the exact selected state and check actual held-out inference.
    model.load_state_dict(best[2])
    if not np.array_equal(prediction, predict(roles["test"])):
        raise ValueError("Selected-state replay changed predictions")
    return prediction, selection, best[2]


def prepare(config, acquire=False):
    specs = {}
    for name in ("m12", "m6"):
        spec = config[name + "_config"]
        verify_file(spec["path"], spec)
        specs[name] = json.loads(Path(spec["path"]).read_text())
    m12, m6 = specs["m12"], specs["m6"]
    rows, matrices, mask, groups, evidence = inputs(m12, prepare=acquire)
    if not mask.all():
        raise ValueError("M14 must keep complete original M12 cohort")
    native_rows, native_data = load_stability(m6)
    native_groups = {}
    structural = np.zeros((len(native_rows), 272))
    for spec in m6["proteins"]:
        pid, ref = spec["protein_id"], spec["reference"]["sequence"]
        graph = load_graph(spec["structure_artifacts"], ref)
        ix = np.array([i for i, r in enumerate(native_rows) if r["protein_id"] == pid])
        dynamic = []
        for i in ix:
            d, f, reasons = variant_inputs(native_rows[i]["mutations"], ref, graph)
            if reasons:
                raise ValueError("M14 native cohort mapping changed")
            dynamic.append(d)
            structural[i] = np.r_[f, 1.0, 1.0]
        native_groups[pid] = {"graph": graph, "dynamic": np.array(dynamic), "indices": ix}
    cohorts = {
        "family": (rows, matrices["structural"], groups, folds(rows, m12["family_order"])),
        "native": (native_rows, structural, native_groups, stability_folds(native_rows, m6)),
    }
    encoder = FrozenProteinMPNN(config["pretrained"], acquire)
    provenance = {
        "family": evidence,
        "native": native_data,
        "graphs": {},
        "pretrained": config["pretrained"]
        | {
            "total_parameters": encoder.parameters,
            "encoder_parameters": encoder.encoder_parameters,
        },
        "exclusions": [],
    }
    for cohort, (rr, _, gg, _) in cohorts.items():
        probe = np.zeros((len(rr), 585))
        for pid, group in gg.items():
            spec = next(
                s
                for s in specs["m12" if cohort == "family" else "m6"]["proteins"]
                if s["protein_id"] == pid
            )
            atoms, prov = mapped_atoms(spec["structure_artifacts"], group["graph"])
            group["edge_features"] = annotate(group["graph"], atoms)
            started = time.perf_counter()
            positions = [n["polymer_position"] - 1 for n in group["graph"]["nodes"]]
            embeddings = encoder.encode(atoms, positions)
            frozen = site_probe(embeddings, group["dynamic"])
            if frozen.shape[1] != probe.shape[1]:
                raise ValueError("Unexpected frozen encoder probe dimensions")
            probe[group["indices"]] = frozen
            prov.update(
                {
                    "encoder_seconds": time.perf_counter() - started,
                    "nodes": len(atoms),
                    "directed_edges": len(group["edge_features"]),
                    "edge_feature_names": EDGE_NAMES,
                    "feature_mean": group["edge_features"].mean(0).tolist(),
                    "feature_std": group["edge_features"].std(0).tolist(),
                    "candidate_directed_counts": group["edge_features"][:, 19:22].sum(0).tolist(),
                    "invalid_directed_counts": (1 - group["edge_features"][:, 22:25])
                    .sum(0)
                    .tolist(),
                    "complete_backbone": True,
                    "connected_components": group["graph"]["audit"].get("connected_components"),
                }
            )
            provenance["graphs"][cohort + ":" + pid] = prov
        cohorts[cohort] = cohorts[cohort] + (probe,)
    return cohorts, provenance


def run(config, acquire=False):
    canonical = json.loads(Path("configs/m14.json").read_text())
    if config | {"output": canonical["output"]} != canonical:
        raise ValueError("M14 configuration differs from prespecified design")
    before = historical()
    output = Path(config["output"])
    if output.resolve() == Path.cwd() or any(
        output.resolve() == p.resolve()
        or output.resolve() in p.resolve().parents
        or p.resolve() in output.resolve().parents
        for p in (Path("benchmarks"), Path("configs"))
    ):
        raise ValueError("Use an isolated runs output directory; preserve historical trees")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise ValueError("Existing experiment; use a new output directory")
    started = time.perf_counter()
    cohorts, provenance = prepare(config, acquire)
    records, predictions, role_records = [], [], []
    cache = Path(config["cache"]) / "weights" / output.name
    cache.mkdir(parents=True, exist_ok=True)
    for cohort, (rows, structural, groups, partitions, probe) in cohorts.items():
        native = cohort == "native"
        for protocol, fold, roles in partitions:
            if not native:
                audit(rows, roles, protocol)
            ids = {role: [rows[i]["id"] for i in ix] for role, ix in roles.items()}
            role_records.append(
                {"cohort": cohort, "protocol": protocol, "fold": fold, "roles": ids}
            )
            for mode in config["models"]:
                for seed in config["seeds"] if not mode.endswith("_ridge") else [0]:
                    start = time.perf_counter()
                    key = f"{cohort}_{protocol}_{fold}_{mode}_{seed}"
                    if mode.endswith("_ridge"):
                        x = structural if mode == "structural_ridge" else probe
                        pred, selection, state = ridge_ensemble(
                            x, rows, roles, config["alphas"], 0, seed, native
                        )
                        prediction = pred[0, roles["test"]]
                        np.savez_compressed(cache / (key + ".npz"), **state)
                        restored = dict(np.load(cache / (key + ".npz")))
                        assert np.array_equal(pred, ridge_predict(x, restored))
                    else:
                        prediction, selection, state = fit(
                            groups,
                            rows,
                            roles,
                            config["training"],
                            seed,
                            mode,
                            config["controls_seed"],
                            native,
                        )
                        torch.save(state, cache / (key + ".pt"))
                        restored = torch.load(cache / (key + ".pt"), weights_only=True)
                        assert all(torch.equal(state[k], restored[k]) for k in state)
                    values = (
                        scores(rows, roles["test"], prediction)["per_protein"]
                        if native
                        else per_protein(rows, roles["test"], prediction)
                    )
                    record = {
                        "cohort": cohort,
                        "protocol": protocol,
                        "fold": fold,
                        "model": mode,
                        "seed": seed,
                        "seconds": time.perf_counter() - start,
                        "selection": selection,
                        "per_protein": values,
                        "state_path": str(
                            cache / (key + (".npz" if mode.endswith("_ridge") else ".pt"))
                        ),
                    }
                    record["state_sha256"] = checksum(record["state_path"])
                    records.append(record)
                    for i, pred in zip(roles["test"], prediction, strict=True):
                        r = rows[i]
                        predictions.append(
                            {
                                "cohort": cohort,
                                "protocol": protocol,
                                "fold": fold,
                                "model": mode,
                                "seed": seed,
                                "id": r["id"],
                                "protein_id": r["protein_id"],
                                "family": r.get("family", "unassigned"),
                                "n_mutations": len(r["mutations"]),
                                "target": r["target"],
                                "prediction": float(pred),
                            }
                        )
                    print(f"{key}: {record['seconds']:.2f}s", flush=True)
                    write_json(output / "metrics.partial.json", records)
    write_json(output / "metrics.json", records)
    write_csv(output / "predictions.csv", predictions)
    write_json(
        output / "audit.json",
        {
            "roles": role_records,
            "provenance": provenance,
            "history_before": before,
            "historical_unchanged": historical() == before,
            "config": config,
            "seconds": time.perf_counter() - started,
            "device": "cpu",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scipy", "scikit-learn", "biopython")
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "sources": {str(p): checksum(p) for p in Path("src/proteinmod").glob("*.py")},
        },
    )
    if historical() != before:
        raise ValueError("Historical artifact changed")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m14.json")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.output:
        config["output"] = args.output
    run(config, acquire=args.prepare)


if __name__ == "__main__":
    main()
