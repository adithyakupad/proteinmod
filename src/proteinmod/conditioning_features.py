"""M8 label-free M6 feature reconstruction and immutable M7 function-only replay."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import checksum, write_json
from proteinmod.embeddings import FrozenESM, fingerprint, site_features
from proteinmod.multitask import MultitaskModel
from proteinmod.residue_cache import residue_cache
from proteinmod.stability_model import StabilityModel
from proteinmod.structural_graph import variant_inputs


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape, str(x.dtype))).encode() + x.tobytes()).hexdigest()


def mutation_features(rows, groups, specs, embedding, cache_root, source=None):
    """Same 961/272 features as M6, with independently cached frozen WT tokens."""
    sites = np.full((len(rows), 961), np.nan, dtype=np.float32)
    structural = np.full((len(rows), 272), np.nan, dtype=np.float32)
    evidence, encoder = {}, None
    for spec in specs:
        pid = spec["protein_id"]
        group = groups[pid]
        reference = spec["reference"]["sequence"]
        cache_spec = {"reference": spec["reference"], "dataset": source or spec["dataset"]}
        wt = [{"id": pid + ":WT", "sequence": reference}]
        try:
            tokens, manifest, directory = residue_cache(cache_spec, wt, embedding, cache_root)
        except ValueError as error:
            if "Missing M5 residue cache" not in str(error):
                raise
            encoder = encoder or FrozenESM(embedding, local_only=True)
            tokens, manifest, directory = residue_cache(
                cache_spec, wt, embedding, cache_root, True, encoder
            )
        for local, index in enumerate(group["indices"]):
            mutation = rows[index]["mutations"]
            _, descriptors, reasons = variant_inputs(mutation, reference, group["graph"])
            if reasons:
                raise ValueError("Mapped cohort changed")
            values = group["residues"][group["token_indices"][local]]
            positions = sorted(int(t[1:-1]) - 1 for t in mutation)
            sites[index] = site_features(values, tokens[0], positions)
            structural[index] = np.r_[descriptors, float(bool(mutation)), 1.0]
        evidence[pid] = {
            "wt_manifest_sha256": checksum(directory / "manifest.json"),
            "encoder": manifest["encoder"],
            "graph": group["graph"]["audit"],
        }
    return {"esm_sites_ridge": sites, "structural_ridge": structural}, evidence


def baseline(rows, groups, paired_rows, paired_groups, f, s, roles, m7, seed, cache_root):
    archived = Path("benchmarks/m7")
    manifest = json.loads((archived / "manifest.json").read_text())
    audit = json.loads((archived / "audit.json").read_text())
    old_identity = {
        "config": m7,
        "historical_data": audit["history"],
        "sources": manifest["source_hashes"],
    }
    original = (
        Path("data/cache/m7/checkpoints")
        / fingerprint(old_identity)
        / f"{f}-{s}-function-1-{seed}.pt"
    )
    restored = Path(cache_root) / "recovered_m7" / original.name
    path = original if original.exists() else restored
    if not path.exists():
        from proteinmod.multitask_experiments import fit

        _, selection, state = fit(
            paired_rows, paired_groups, roles, m7["training"], seed, "function"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"schema": m7["schema"], "state": state, "calibration": selection["calibration"]}, path
        )
    state = torch.load(path, weights_only=True, map_location="cpu")
    if state["schema"] != m7["schema"]:
        raise ValueError("Wrong M7 checkpoint schema")
    identity = {
        "schema": "m8-frozen-function-representation-v1",
        "checkpoint_sha256": checksum(path),
        "roles": {p: [rows[i]["id"] for i in ix] for p, ix in roles["function"].items()},
        "inputs": [(r["id"], r["sequence"], sorted(r["mutations"])) for r in rows],
        "source": checksum(Path(__file__)),
        "tokens": audit["function_tokens"],
    }
    cache = Path(cache_root) / "function_representations" / fingerprint(identity)
    cache.mkdir(parents=True, exist_ok=True)
    cache_file, cache_meta = cache / "features.npz", cache / "manifest.json"
    if cache_file.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        if (
            meta["identity"] != json.loads(json.dumps(identity))
            or checksum(cache_file) != meta["sha256"]
        ):
            raise ValueError("Function feature cache provenance mismatch")
        with np.load(cache_file, allow_pickle=False) as data:
            return data["representation"], data["prediction"], meta
    torch.set_num_threads(m7["training"]["cpu_threads"])
    model = MultitaskModel(next(iter(groups.values()))["graph"])
    model.load_state_dict(state["state"])
    model.requires_grad_(False).eval()
    calibration = state["calibration"]["function"]
    h = np.full((len(rows), 100), np.nan, dtype=np.float32)
    prediction = np.full(len(rows), np.nan)
    for group in groups.values():
        g = group["graph"]
        edge = g["edge_index"]
        topology = (
            torch.as_tensor(g["static"]),
            torch.as_tensor(edge[0]),
            torch.as_tensor(edge[1]),
            torch.as_tensor(g["edge_features"]),
            torch.tensor(
                np.maximum(np.bincount(edge[1], minlength=len(g["static"])), 1), dtype=torch.float32
            )[None, :, None],
        )
        positions = torch.tensor([n["reference_position"] - 1 for n in g["nodes"]])
        for indices in roles["function"].values():
            local = np.flatnonzero(np.isin(group["indices"], indices))
            for start in range(0, len(local), 32):
                ix = local[start : start + 32]
                global_ix = group["indices"][ix]
                residues = torch.from_numpy(
                    np.array(group["residues"][group["token_indices"][ix]], copy=True)
                )
                mutation_mask = torch.zeros(residues.shape[:2], dtype=torch.bool)
                for j, i in enumerate(global_ix):
                    for token in rows[i]["mutations"]:
                        mutation_mask[j, int(token[1:-1]) - 1] = True
                with torch.no_grad():
                    features = StabilityModel.represent(
                        model.encoder,
                        residues,
                        torch.tensor(group["dynamic"][ix]),
                        topology,
                        positions,
                        mutation_mask,
                    )
                    h[global_ix] = features.numpy()
                    prediction[global_ix] = (
                        model.heads["function"](features).squeeze(-1).double().numpy()
                        * calibration["scale"]
                        + calibration["mean"]
                    )
    eligible = np.unique(np.concatenate(list(roles["function"].values())))
    if not np.isfinite(h[eligible]).all() or not np.isfinite(prediction[eligible]).all():
        raise ValueError("Incomplete baseline features")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Function pathway was not frozen")
    np.savez_compressed(cache_file, representation=h, prediction=prediction)
    meta = {
        "identity": identity,
        "sha256": checksum(cache_file),
        "path": str(cache_file),
        "checkpoint": str(path),
        "representation_sha256": array_hash(h),
        "all_parameters_frozen": True,
        "supervision": "function training roles only; epoch selected on validation",
    }
    write_json(cache_meta, meta)
    return h, prediction, meta
