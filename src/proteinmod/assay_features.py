"""Frozen, metadata-independent M4/M5 features; no M8 stability scalars."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from proteinmod.assay_data import collection, folds
from proteinmod.conditioning_features import array_hash
from proteinmod.data import checksum, write_json
from proteinmod.embeddings import FrozenESM, fingerprint, site_features
from proteinmod.residue_cache import residue_cache
from proteinmod.stability_model import StabilityModel, encoder_digest
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import run_structure


def cache_directory(config):
    contract = {
        k: config[k]
        for k in (
            "schema",
            "proteins",
            "assays",
            "cap_genotypes_per_protein",
            "split_salt",
            "split_fractions",
            "embedding",
            "pretrained",
        )
    }
    return (
        Path(config["cache"])
        / "features"
        / fingerprint({"contract": contract, "source": checksum(Path(__file__))})
    )


def encoder_state(config):
    path = Path(config["pretrained"]["path"])
    if not path.exists():
        from proteinmod.multitask_experiments import fit, inputs

        m7 = json.loads(Path("configs/m7.json").read_text())
        rows, groups, roles, _ = inputs(m7)
        spec = config["pretrained"]
        role = next(
            r
            for f, s, r in roles
            if f == spec["m7_function_fold"] and s == spec["m7_stability_fold"]
        )
        _, selection, state = fit(rows, groups, role, m7["training"], spec["seed"], "function")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": state, "calibration": selection["calibration"]}, path)
    state = torch.load(path, weights_only=True, map_location="cpu")["state"]
    selected = {
        k.removeprefix("encoder.backbone."): v
        for k, v in state.items()
        if k.startswith("encoder.backbone.")
    }
    if encoder_digest(selected) != config["pretrained"]["encoder_tensor_sha256"]:
        raise ValueError("Pretrained encoder tensor identity mismatch")
    return selected


def prepare(config):
    started = time.perf_counter()
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    rows, data = collection(config)
    state = encoder_state(config)
    values = {
        "esm_sites": np.zeros((len(rows), 961), np.float32),
        "structural": np.zeros((len(rows), 270), np.float32),
        "multimodal": np.zeros((len(rows), 100), np.float32),
    }
    evidence, encoder = {}, None
    for spec in config["proteins"]:
        pid, ref = spec["protein_id"], spec["reference"]["sequence"]
        settings = {
            "structure": {**spec["structure"], "name": pid + " AlphaFold DB v6"},
            "reference": spec["reference"],
            "mapping": {
                "chain": "A",
                "model_index": 0,
                "min_coverage": 1.0,
                "min_identity": 1.0,
                "allowed_mismatches": [],
            },
            "geometry": {"contact_cutoff_a": 8.0, "sasa_points": 960, "dssp": "off"},
            "structure_run": {"output": spec["structure_artifacts"]},
        }
        manifest_path = Path(spec["structure_artifacts"]) / "manifest.json"
        if not manifest_path.exists():
            run_structure(settings, render=False)
        elif json.loads(manifest_path.read_text())["config"] != settings:
            raise ValueError("Cached M9 graph configuration differs")
        graph = load_graph(spec["structure_artifacts"], ref)
        unique = {}
        for r in rows:
            if r["protein_id"] == pid:
                unique.setdefault(
                    r["genotype_id"],
                    {
                        "id": r["genotype_id"],
                        "sequence": r["sequence"],
                        "mutations": r["mutations"],
                    },
                )
        part = list(unique.values())
        label_free = Path(config["cache"]) / "variants" / (pid + ".json")
        label_free.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            label_free,
            {
                "schema": config["schema"],
                "rows": part,
                "sources": [
                    a["dataset"]["sha256"] for a in config["assays"] if a["protein_id"] == pid
                ],
            },
        )
        cache_spec = {"reference": spec["reference"], "dataset": {"path": str(label_free)}}
        token_rows = part + [{"id": pid + ":WT", "sequence": ref}]
        try:
            tokens, manifest, directory = residue_cache(
                cache_spec, token_rows, config["embedding"], Path(config["cache"]) / "residues"
            )
        except ValueError as error:
            if "Missing M5 residue cache" not in str(error):
                raise
            encoder = encoder or FrozenESM(config["embedding"], local_only=True)
            tokens, manifest, directory = residue_cache(
                cache_spec,
                token_rows,
                config["embedding"],
                Path(config["cache"]) / "residues",
                True,
                encoder,
            )
        model = StabilityModel(graph, encoder_state=state).requires_grad_(False).eval()
        positions = torch.tensor([n["reference_position"] - 1 for n in graph["nodes"]])
        dynamic, mutation_masks, structural, sites = [], [], [], []
        for j, r in enumerate(part):
            d, features, reasons = variant_inputs(r["mutations"], ref, graph)
            if reasons:
                raise ValueError(
                    "Selected genotype lacks compatible structure; cohort must be re-declared"
                )
            dynamic.append(d)
            structural.append(features)
            mask = np.zeros(len(ref), bool)
            mask[sorted(int(t[1:-1]) - 1 for t in r["mutations"])] = True
            mutation_masks.append(mask)
            sites.append(site_features(tokens[j], tokens[-1], np.flatnonzero(mask)))
        h = []
        for start in range(0, len(part), 32):
            ix = slice(start, min(start + 32, len(part)))
            with torch.no_grad():
                h.extend(
                    model.represent(
                        torch.tensor(np.array(tokens[ix], copy=True)),
                        torch.tensor(np.array(dynamic[ix])),
                        None,
                        positions,
                        torch.tensor(np.array(mutation_masks[ix])),
                    ).numpy()
                )
        mapping = {r["id"]: j for j, r in enumerate(part)}
        for i, r in enumerate(rows):
            if r["protein_id"] == pid:
                j = mapping[r["genotype_id"]]
                for key, array in [
                    ("esm_sites", sites),
                    ("structural", structural),
                    ("multimodal", h),
                ]:
                    values[key][i] = array[j]
        confidence = [n["plddt"] for n in graph["nodes"] if n["plddt"] is not None]
        evidence[pid] = {
            "graph": graph["audit"],
            "token_manifest_sha256": checksum(directory / "manifest.json"),
            "encoder": manifest["encoder"],
            "unique_genotypes": len(part),
            "mapping_exclusions": 0,
            "plddt_below_50_fraction": float(np.mean(np.array(confidence) < 50)),
        }
    if any(not np.isfinite(x).all() for x in values.values()):
        raise ValueError("Incomplete feature collection")
    output = cache_directory(config)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "features.npz", **values)
    role_evidence = [
        {"protocol": p, "fold": f, "ids": {k: [rows[i]["id"] for i in v] for k, v in role.items()}}
        for p, f, role in folds(rows, config["protein_order"])
    ]
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "data": data,
            "proteins": evidence,
            "roles": role_evidence,
            "feature_sha256": checksum(output / "features.npz"),
            "array_sha256": {k: array_hash(v) for k, v in values.items()},
            "pretrained": config["pretrained"],
            "no_stability_prediction_inputs": True,
            "observation_ids": [r["id"] for r in rows],
            "elapsed_seconds": time.perf_counter() - started,
            "timing_scope": "graph/embedding cache hits included; not a cold-start measurement",
        },
    )
    return output


def load(config):
    directory = cache_directory(config)
    manifest = json.loads((directory / "manifest.json").read_text())
    if checksum(directory / "features.npz") != manifest["feature_sha256"]:
        raise ValueError("M9 feature cache checksum mismatch")
    rows, _ = collection(config)
    if manifest["observation_ids"] != [r["id"] for r in rows]:
        raise ValueError("M9 feature observation order differs")
    with np.load(directory / "features.npz", allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    return values, manifest
