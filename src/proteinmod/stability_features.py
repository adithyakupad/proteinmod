"""Recompute label-free M2/M5 features; source PCA/label-derived columns are forbidden."""

import json
import time
import zipfile
from pathlib import Path

import numpy as np

from proteinmod.data import AMINO_ACIDS, checksum, write_json
from proteinmod.embeddings import FrozenESM, site_features
from proteinmod.multiprotein_data import sequence_features, similarity_audit
from proteinmod.residue_cache import residue_cache
from proteinmod.stability_data import load_stability, output_directory
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import run_structure


def prepare(config):
    output = output_directory(config["output"])
    rows, audit = load_stability(config)
    evidence = {}
    encoder = None
    for spec in config["proteins"]:
        p = spec["protein_id"]
        source = Path("data/cache/m6/structures") / f"{p}.pdb"
        source.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(config["sources"]["structures"]["path"]) as z:
            raw = z.read(spec["structure_member"])
        import hashlib

        if hashlib.sha256(raw).hexdigest() != spec["structure_sha256"]:
            raise ValueError("Author structure checksum mismatch")
        source.write_bytes(raw)
        settings = {
            "structure": {
                "name": p + " author AlphaFold assay construct",
                "kind": "local",
                "format": "pdb",
                "path": str(source),
                "sha256": spec["structure_sha256"],
                "source": config["sources"]["structures"]["url"],
                "license": config["license"],
                "confidence": "alphafold_plddt",
            },
            "reference": spec["reference"],
            "mapping": {
                "chain": "A",
                "model_index": 0,
                "min_coverage": 0.95,
                "min_identity": 1.0,
                "allowed_mismatches": [],
            },
            "geometry": {"contact_cutoff_a": 8.0, "sasa_points": 960, "dssp": "off"},
            "structure_run": {"output": spec["structure_artifacts"]},
        }
        output_directory(spec["structure_artifacts"])
        manifest = Path(spec["structure_artifacts"]) / "manifest.json"
        if not manifest.exists():
            run_structure(settings, render=False)
        elif json.loads(manifest.read_text())["config"] != settings:
            raise ValueError("M6 graph cache configuration mismatch; cached artifacts preserved")
        graph = load_graph(spec["structure_artifacts"], spec["reference"]["sequence"])
        part = [r for r in rows if r["protein_id"] == p]
        # No labels enter the residue cache identity or encoder inputs.
        cache_spec = {"reference": spec["reference"], "dataset": config["sources"]["measurements"]}
        try:
            _, cache_manifest, _ = residue_cache(
                cache_spec, part, config["embedding"], config["residue_cache"]
            )
        except ValueError as error:
            if "Missing M5 residue cache" not in str(error):
                raise
            encoder = encoder or FrozenESM(config["embedding"], local_only=True)
            _, cache_manifest, _ = residue_cache(
                cache_spec, part, config["embedding"], config["residue_cache"], True, encoder
            )
        wt_rows = [{"id": p + ":WT", "sequence": spec["reference"]["sequence"]}]
        try:
            residue_cache(cache_spec, wt_rows, config["embedding"], config["residue_cache"])
        except ValueError as error:
            if "Missing M5 residue cache" not in str(error):
                raise
            encoder = encoder or FrozenESM(config["embedding"], local_only=True)
            residue_cache(
                cache_spec, wt_rows, config["embedding"], config["residue_cache"], True, encoder
            )
        evidence[p] = {"graph": graph["audit"], "residue_cache": cache_manifest}
    write_json(output / "preparation.json", {"data": audit, "proteins": evidence})
    return evidence


def transfer_state(config):
    import torch

    from proteinmod.stability_model import encoder_digest

    path = Path(config["transfer"]["checkpoint"])
    if not path.exists():
        raise ValueError("Missing frozen M5 checkpoint; run rebuild-transfer (see M6 protocol)")
    state = torch.load(path, weights_only=True, map_location="cpu")
    if encoder_digest(state) != config["transfer"]["encoder_tensor_sha256"]:
        raise ValueError("Frozen M5 encoder tensor checksum mismatch")
    return state, {
        "path": str(path),
        "serialization_sha256": checksum(path),
        "encoder_tensor_sha256": config["transfer"]["encoder_tensor_sha256"],
        "transfer": config["transfer"],
    }


def rebuild_transfer(config):
    checkpoint = Path(config["transfer"]["checkpoint"])
    if checkpoint.exists():
        transfer_state(config)
        return checksum(checkpoint)
    import torch

    from proteinmod import hybrid_experiments as m5
    from proteinmod.multiprotein_data import protocols
    from proteinmod.stability_model import encoder_digest

    settings = json.loads(Path(config["transfer"]["m5_config"]).read_text())
    m4 = json.loads(Path(settings["m4_config"]).read_text())
    rows, mask, groups, _, _ = m5.inputs(settings, m4)
    spec = config["transfer"]
    roles = next(
        ix
        for protocol, fold, ix in protocols(rows, m4["experiment"]["fold_order"])
        if protocol == spec["protocol"] and fold == spec["fold"]
    )
    roles = {p: ix[mask[ix]] for p, ix in roles.items()}
    _, _, state = m5.fit(groups, rows, roles, settings["training"], spec["seed"], spec["mode"])
    if encoder_digest(state) != spec["encoder_tensor_sha256"]:
        raise ValueError(
            "Rebuilt encoder differs from the preregistered M5 encoder; preserve and investigate environment"
        )
    path = Path(spec["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return checksum(path)


def model_inputs(config):
    import torch

    from proteinmod.stability_model import StabilityModel

    started = time.perf_counter()
    rows, audit = load_stability(config)
    state, transfer = transfer_state(config)
    m4 = json.loads(Path("configs/m4.json").read_text())
    similarity = similarity_audit(config["proteins"] + m4["collection"])
    selected = {s["protein_id"] for s in config["proteins"]}
    for pair in similarity:
        if (
            (pair["left"] in selected) != (pair["right"] in selected)
            and pair["aligned_identity"] >= 0.3
            and min(pair["left_coverage"], pair["right_coverage"]) >= 0.8
        ):
            raise ValueError("Stability protein overlaps supervised function-pretraining cohort")
    sequence, structural, onehot, sites, frozen = [], [], [], [], []
    graph_audits, encoders, missing = {}, {}, []
    torch.set_num_threads(config["head"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    reference_encoder = None
    for spec in config["proteins"]:
        protein = spec["protein_id"]
        part = [r for r in rows if r["protein_id"] == protein]
        reference = spec["reference"]["sequence"]
        graph = load_graph(spec["structure_artifacts"], reference)
        token_values, manifest, directory = residue_cache(
            {"reference": spec["reference"], "dataset": config["sources"]["measurements"]},
            part,
            config["embedding"],
            config["residue_cache"],
        )
        # Reuse the same cached extraction for WT; labels never enter this call.
        wt_rows = [{"id": protein + ":WT", "sequence": reference}]
        cache_spec = {"reference": spec["reference"], "dataset": config["sources"]["measurements"]}
        try:
            wt_tokens, _, _ = residue_cache(
                cache_spec, wt_rows, config["embedding"], config["residue_cache"]
            )
        except ValueError as error:
            if "Missing M5 residue cache" not in str(error):
                raise
            reference_encoder = reference_encoder or FrozenESM(config["embedding"], local_only=True)
            wt_tokens, _, _ = residue_cache(
                cache_spec,
                wt_rows,
                config["embedding"],
                config["residue_cache"],
                True,
                reference_encoder,
            )
        model = StabilityModel(
            graph,
            config["embedding"]["dimension"],
            config["head"]["dimension"],
            config["head"]["heads"],
            config["head"]["layers"],
            state,
        )
        model.eval()
        topology = (
            model.backbone.structure.static,
            model.backbone.structure.source,
            model.backbone.structure.target,
            model.backbone.structure.edge_features,
            model.backbone.structure.degree,
        )
        positions = torch.tensor([n["reference_position"] - 1 for n in graph["nodes"]])
        dynamics, mutation_masks = [], []
        for row in part:
            mapped = tuple(t for t in row["mutations"] if int(t[1:-1]) in graph["lookup"])
            dynamic, descriptors, reasons = variant_inputs(mapped, reference, graph)
            if reasons:
                raise ValueError("Unexpected graph input rejection")
            available = len(mapped) / max(1, len(row["mutations"]))
            missing.append(
                {
                    "id": row["id"],
                    "sequence_available": True,
                    "structure_available": True,
                    "mutation_mapping_fraction": available,
                }
            )
            structural.append(np.r_[descriptors, available, 1.0])
            dynamics.append(dynamic)
            mask = np.array([a != b for a, b in zip(reference, row["sequence"], strict=True)])
            mutation_masks.append(mask)
            sequence.append(sequence_features(row, reference))
            pair = np.zeros(401, dtype=np.float32)
            for t in row["mutations"]:
                pair[AMINO_ACIDS.index(t[0]) * 20 + AMINO_ACIDS.index(t[-1])] += 1
            pair[-1] = len(row["mutations"])
            onehot.append(pair)
        mutation_masks = np.stack(mutation_masks)
        size = config["head"]["batch_size"]
        for start in range(0, len(part), size):
            ix = slice(start, start + size)
            tokens = np.array(token_values[ix], copy=True)
            features = model.represent(
                torch.from_numpy(tokens),
                torch.tensor(np.array(dynamics[ix])),
                topology,
                positions,
                torch.tensor(mutation_masks[ix]),
            ).numpy()
            frozen.extend(features)
            sites.extend(
                site_features(values, wt_tokens[0], np.flatnonzero(mask))
                for values, mask in zip(tokens, mutation_masks[ix], strict=True)
            )
        graph_audits[protein] = graph["audit"]
        encoders[protein] = {
            "manifest_sha256": checksum(directory / "manifest.json"),
            "encoder": manifest["encoder"],
        }
    features = {
        "physicochemical_ridge": np.asarray(sequence),
        "substitution_one_hot_ridge": np.asarray(onehot),
        "esm_sites_ridge": np.asarray(sites),
        "structural_ridge": np.asarray(structural),
        "frozen_hybrid_ridge": np.asarray(frozen),
    }
    if any(len(x) != len(rows) or not np.isfinite(x).all() for x in features.values()):
        raise ValueError("Invalid or incomplete stability feature arrays")
    evidence = {
        "data": audit,
        "graphs": graph_audits,
        "frozen_embeddings": encoders,
        "transfer": transfer,
        "sequence_similarity": similarity,
        "missingness": missing,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_columns": {k: v.shape[1] for k, v in features.items()},
        "label_free": True,
    }
    return rows, features, evidence
