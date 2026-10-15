"""M10 frozen confidence-input and learned-gate experiments on immutable M9 roles."""

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

from proteinmod.assay_data import collection, folds
from proteinmod.assay_features import encoder_state
from proteinmod.assay_models import evaluate
from proteinmod.confidence_fusion import MODES, ReliabilityFusion, frozen_representation
from proteinmod.data import checksum, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.multiprotein_experiments import fit_targets
from proteinmod.residue_cache import residue_cache
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import write_csv
from proteinmod.structure_reliability import FIELDS, confidence_graph, reliability_features


def history():
    return {
        str(p): checksum(p)
        for root in (Path("benchmarks"), Path("configs"))
        for p in sorted(root.rglob("*"))
        if p.is_file() and "m10" not in p.parts and p.name != "m10.json"
    }


def output_path(path):
    output, root = Path(path).resolve(), Path("benchmarks").resolve()
    allowed = root / "m10"
    if (output == root or root in output.parents or output in root.parents) and not (
        output == allowed or allowed in output.parents
    ):
        raise ValueError("M10 would overwrite historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output


def inputs(config):
    m9 = json.loads(Path(config["m9_config"]).read_text())
    rows, audit = collection(m9)
    if json.loads(json.dumps(rows)) != json.loads(Path("benchmarks/m9/rows.json").read_text()):
        raise ValueError("M9 observations changed")
    roles = folds(rows, m9["protein_order"])
    old = json.loads(Path("benchmarks/m9/audit.json").read_text())["roles"]
    for (protocol, fold, role), expected in zip(roles, old, strict=True):
        if (protocol, fold) != (expected["protocol"], expected["fold"]) or any(
            [rows[i]["id"] for i in ix] != expected["ids"][key] for key, ix in role.items()
        ):
            raise ValueError("M9 protocol roles changed")
    return m9, rows, audit, roles


def feature_path(config):
    return (
        Path(config["cache"])
        / "features"
        / fingerprint(
            {
                "m9": checksum(Path(config["m9_config"])),
                "seeds": config["seeds"],
                "source": {
                    n: checksum(Path("src/proteinmod") / n)
                    for n in (
                        "reliability_experiments.py",
                        "structure_reliability.py",
                        "confidence_fusion.py",
                    )
                },
            }
        )
    )


def prepare(config):
    start = time.perf_counter()
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    m9, rows, data, _ = inputs(config)
    state = encoder_state(m9)
    profiles = ["neutral", "real"] + [f"shuffled{s}" for s in config["seeds"]]
    values = {"h_" + p: np.zeros((len(rows), 100), np.float32) for p in profiles + ["missing"]}
    values.update(
        {
            "q_" + p: np.zeros((len(rows), len(FIELDS)), np.float32)
            for p in profiles
            if p != "neutral"
        }
    )
    graphs = {}
    for spec in m9["proteins"]:
        pid, ref = spec["protein_id"], spec["reference"]["sequence"]
        graph = load_graph(spec["structure_artifacts"], ref)
        unique = {}
        for r in rows:
            if r["protein_id"] == pid:
                unique.setdefault(
                    r["genotype_id"], {k: r[k] for k in ("id", "sequence", "mutations")}
                )
        # M9 residue cache uses genotype IDs, not assay observation IDs.
        for genotype, r in unique.items():
            r["id"] = genotype
        part = list(unique.values())
        source = Path(m9["cache"]) / "variants" / (pid + ".json")
        tokens, manifest, directory = residue_cache(
            {"reference": spec["reference"], "dataset": {"path": str(source)}},
            part + [{"id": pid + ":WT", "sequence": ref}],
            m9["embedding"],
            Path(m9["cache"]) / "residues",
        )
        dynamic, masks = [], []
        for r in part:
            d, _, reasons = variant_inputs(r["mutations"], ref, graph)
            if reasons:
                raise ValueError("Matched M9 structural mapping changed")
            dynamic.append(d)
            mask = np.zeros(len(ref), bool)
            mask[[int(t[1:-1]) - 1 for t in r["mutations"]]] = True
            masks.append(mask)
        dynamic, masks = np.array(dynamic), np.array(masks)
        index = {key: j for j, key in enumerate(unique)}
        for profile in profiles + ["missing"]:
            shuffled = profile.startswith("shuffled")
            current = (
                None
                if profile == "missing"
                else confidence_graph(
                    graph,
                    "shuffled" if shuffled else profile,
                    int(profile.removeprefix("shuffled")) if shuffled else 0,
                )
            )
            representations = []
            for start_batch in range(0, len(part), 32):
                ix = slice(start_batch, min(start_batch + 32, len(part)))
                representations.extend(
                    frozen_representation(
                        state, current, np.array(tokens[ix], copy=True), dynamic[ix], masks[ix]
                    )
                )
            quality = (
                None
                if profile in ("neutral", "missing")
                else [reliability_features(r["mutations"], ref, current) for r in part]
            )
            for i, r in enumerate(rows):
                if r["protein_id"] == pid:
                    j = index[r["genotype_id"]]
                    values["h_" + profile][i] = representations[j]
                    if quality is not None:
                        values["q_" + profile][i] = quality[j]
        graphs[pid] = {
            "graph": graph["audit"],
            "token_manifest_sha256": checksum(directory / "manifest.json"),
            "encoder": manifest["encoder"],
            "source": spec["structure"],
            "mapping_exclusions": 0,
        }
    for x in values.values():
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite frozen confidence features")
    # The real-input profile must reproduce M9's frozen multimodal array exactly.
    from proteinmod.assay_features import load

    historical, _ = load(m9)
    np.testing.assert_array_equal(values["h_real"], historical["multimodal"])
    root = feature_path(config)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "features.npz", **values)
    write_json(
        root / "manifest.json",
        {
            "config": config,
            "fields": FIELDS,
            "graphs": graphs,
            "data": data,
            "observation_ids": [r["id"] for r in rows],
            "feature_sha256": checksum(root / "features.npz"),
            "m9_real_input_encoder_replay_exact": True,
            "elapsed_seconds": time.perf_counter() - start,
            "no_stability_predictions": True,
            "confidence_is_predictive_uncertainty": False,
        },
    )
    return root


def load_features(config):
    root = feature_path(config)
    manifest = json.loads((root / "manifest.json").read_text())
    if checksum(root / "features.npz") != manifest["feature_sha256"]:
        raise ValueError("M10 feature checksum mismatch")
    with np.load(root / "features.npz", allow_pickle=False) as source:
        features = {k: source[k] for k in source.files}
    return features, manifest


def mode_inputs(features, mode, seed):
    if mode not in MODES:
        raise ValueError("Unknown M10 ablation")
    profile = (
        f"shuffled{seed}"
        if "shuffled" in mode
        else "real"
        if mode in ("confidence_input", "combined")
        else "neutral"
    )
    quality = features["q_" + (f"shuffled{seed}" if "shuffled" in mode else "real")]
    return features["h_" + profile], quality


def fit(h, quality, missing, rows, roles, settings, seed, mode):
    if settings["device"] != "cpu" or mode not in MODES:
        raise ValueError("Deterministic CPU and declared mode required")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = ReliabilityFusion(hidden=settings["hidden"])
    x, q = torch.tensor(h), torch.tensor(quality)
    y = torch.tensor(fit_targets(rows, roles["train"]), dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    rng = np.random.default_rng(seed)
    best, stale, trace = None, 0, []

    def prediction(representation=x, context=q):
        model.eval()
        with torch.no_grad():
            return tuple(
                v.double().numpy() for v in model(representation, context, mode, diagnostics=True)
            )

    for epoch in range(1, settings["max_epochs"] + 1):
        model.train()
        order, total = rng.permutation(roles["train"]), 0.0
        for start in range(0, len(order), settings["batch_size"]):
            ix = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[ix], q[ix], mode) - y[ix]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite M10 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        pred = prediction()[0]
        score = evaluate(rows, roles["validation"], pred[roles["validation"]])["macro"]
        trace.append(
            {"epoch": epoch, "training_mse_online": total / len(order), "validation_macro": score}
        )
        if best is None or score > best[0]:
            best, stale = (score, epoch, copy.deepcopy(model.state_dict())), 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    model.load_state_dict(best[2])
    prediction_values = prediction()
    fallback = prediction(torch.tensor(missing), torch.zeros_like(q))
    selection = {
        "epoch": best[1],
        "validation_macro": best[0],
        "trace": trace,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_macro": evaluate(rows, roles["train"], prediction_values[0][roles["train"]])[
            "macro"
        ],
    }
    state = {
        "schema": "m10-reliability-fusion-v1",
        "state": best[2],
        "mode": mode,
        "hidden": settings["hidden"],
        "fields": FIELDS,
        "normalization": "fixed physical units; no per-protein fitted feature scales",
        "seed": seed,
    }
    return prediction_values, fallback, selection, state


def run(config):
    before, start = history(), time.perf_counter()
    output = output_path(config["output"])
    _, rows, _, roles = inputs(config)
    features, feature_manifest = load_features(config)
    if feature_manifest["observation_ids"] != [r["id"] for r in rows]:
        raise ValueError("M10 observation order differs")
    records, predictions = [], []
    sources = {p.name: checksum(p) for p in Path("src/proteinmod").glob("*reliab*.py")}
    sources["confidence_fusion.py"] = checksum(Path("src/proteinmod/confidence_fusion.py"))
    cache = (
        Path(config["cache"])
        / "heads"
        / fingerprint(
            {"config": config, "features": feature_manifest["feature_sha256"], "source": sources}
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    for protocol, fold, role in roles:
        for seed in config["seeds"]:
            for mode in config["modes"]:
                begun = time.perf_counter()
                h, quality = mode_inputs(features, mode, seed)
                pred, fallback, selection, state = fit(
                    h, quality, features["h_missing"], rows, role, config["training"], seed, mode
                )
                path = cache / f"{protocol}-{fold}-{mode}-{seed}.pt"
                torch.save({**state, "feature_sha256": feature_manifest["feature_sha256"]}, path)
                records.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "mode": mode,
                        "seed": seed,
                        "selection": selection,
                        "test": evaluate(rows, role["test"], pred[0][role["test"]]),
                        "missing_structure_test": evaluate(
                            rows, role["test"], fallback[0][role["test"]]
                        ),
                        "elapsed_seconds": time.perf_counter() - begun,
                        "checkpoint_sha256": checksum(path),
                    }
                )
                for i in role["test"]:
                    predictions.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "mode": mode,
                            "seed": seed,
                            "id": rows[i]["id"],
                            "protein_id": rows[i]["protein_id"],
                            "assay_id": rows[i]["assay_id"],
                            "family": rows[i]["family"],
                            "measured_native_score": rows[i]["target"],
                            "prediction": pred[0][i],
                            "structural_gate": pred[1][i],
                            "sequence_query_expert": pred[2][i],
                            "structure_query_expert": pred[3][i],
                            "missing_structure_prediction": fallback[0][i],
                            "missing_structure_gate": fallback[1][i],
                            "site_plddt": features["q_real"][i, 3] * 100,
                            "site_plddt_available": bool(features["q_real"][i, 7]),
                            "mutation_count": len(rows[i]["mutations"]),
                            "source": "alphafold_plddt",
                        }
                    )
                assert (fallback[1] == 0).all()
            print(protocol, fold, seed, "complete", flush=True)
        write_json(output / "runs.json", records)
    if len({r["selection"]["parameters"] for r in records}) != 1:
        raise ValueError("Registered model budgets differ")
    write_csv(output / "predictions.csv", predictions)
    write_json(
        output / "audit.json",
        {
            "features": feature_manifest,
            "history": before,
            "roles": [
                {
                    "protocol": p,
                    "fold": f,
                    "ids": {k: [rows[i]["id"] for i in ix] for k, ix in role.items()},
                }
                for p, f, role in roles
            ],
            "source_identifiability": "Primary M9 cohort is all AlphaFold; experimental-vs-AF performance not identifiable",
            "reference_geometry_only": True,
            "predictive_uncertainty_calibrated": False,
            "stability_scalar_inputs": False,
        },
    )
    if history() != before:
        raise ValueError("Historical artifacts/configurations changed")
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_hashes": sources,
            "checkpoint_root": str(cache),
            "feature_root": str(feature_path(config)),
            "device": "cpu",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "numpy", "scipy", "scikit-learn", "biopython")
            },
            "elapsed_seconds": time.perf_counter() - start,
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
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--config", default="configs/m10.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(prepare(config)) if args.action == "prepare" else run(config)


if __name__ == "__main__":
    main()
