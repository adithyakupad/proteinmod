"""Controlled M17 development fits. No external outcomes are loaded."""

import copy
import csv
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from proteinmod.data import checksum, write_json
from proteinmod.edge_chemistry import annotate, mapped_atoms
from proteinmod.edge_experiments import fit as graph_fit
from proteinmod.family_experiments import inputs
from proteinmod.mechanism import degree_null
from proteinmod.multiprotein_experiments import fit_targets, macro_score, per_protein
from proteinmod.pretrained_structure import FrozenProteinMPNN, site_probe


def prepare():
    config = json.loads(Path("configs/m12.json").read_text())
    rows, matrices, mask, groups, evidence = inputs(config)
    if not mask.all():
        raise ValueError("Matched cohort changed")
    stored = json.loads(Path("benchmarks/m12/rows.json").read_text())
    assert json.loads(json.dumps(rows)) == stored
    for name, width in [
        ("esm_global", 480),
        ("esm_wt", 480),
        ("mpnn_local", 585),
        ("mpnn_global", 201),
    ]:
        matrices[name] = np.zeros((len(rows), width), dtype=np.float32)
    frozen = FrozenProteinMPNN(json.loads(Path("configs/m14.json").read_text())["pretrained"])
    audits = {}
    null_groups = copy.deepcopy(groups)
    for pid, g in groups.items():
        spec = next(p for p in config["proteins"] if p["protein_id"] == pid)
        atoms, provenance = mapped_atoms(spec["structure_artifacts"], g["graph"])
        g["edge_features"] = annotate(g["graph"], atoms)
        embeddings = frozen.encode(atoms, [n["polymer_position"] - 1 for n in g["graph"]["nodes"]])
        matrices["mpnn_local"][g["indices"]] = site_probe(embeddings, g["dynamic"])
        for local, i in enumerate(g["indices"]):
            tokens = g["residues"][g["token_indices"][local]]
            matrices["esm_global"][i] = tokens.mean(0)
            matrices["esm_wt"][i] = g["residues"][-1].mean(0)
            d = g["dynamic"][local]
            changed = d[:, 0] > 0
            # Same substitution pooling width as local probe, without local backbone identity.
            chemistry = d[changed, 1:]
            matrices["mpnn_global"][i] = np.r_[
                embeddings.mean(0),
                chemistry.mean(0),
                chemistry.sum(0),
                chemistry.max(0),
                changed.sum(),
            ]
        ng = null_groups[pid]
        ng["graph"]["edge_index"], swaps = degree_null(g["graph"]["edge_index"], 314159)
        ng["edge_features"] = g["edge_features"].copy()
        # Null edges deliberately break endpoint-feature geometry, retaining feature marginals.
        ng["graph"]["edge_features"] = g["graph"]["edge_features"].copy()
        coords = np.array([n["ca"] for n in g["graph"]["nodes"]])
        distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        expected = {
            (a, b)
            for a in range(len(coords))
            for b in range(a + 1, len(coords))
            if distance[a, b] <= 8
            or abs(
                g["graph"]["nodes"][a]["polymer_position"]
                - g["graph"]["nodes"][b]["polymer_position"]
            )
            == 1
        }
        found = {tuple(sorted(e)) for e in g["graph"]["edge_index"].T}
        missing = g["graph"]["audit"]["removed_positions"]
        mutation_distances = []
        for i in g["indices"]:
            positions = [int(t[1:-1]) for t in rows[i]["mutations"]]
            mutation_distances.append(
                min((abs(p - m) for p in positions for m in missing), default=None)
            )
        audits[pid] = {
            "physical_contact_pairs": len(found),
            "expected_pairs": len(expected),
            "missing_expected_pairs": len(expected - found),
            "extra_pairs": len(found - expected),
            "graph_audit": g["graph"]["audit"],
            "provenance": provenance,
            "backbone_complete_nodes": sum(
                all(k in a for k in ("N", "CA", "C", "O")) for a in atoms
            ),
            "sidechain_atom_counts": [len(a) for a in atoms],
            "mutation_sequence_distance_to_missing": mutation_distances,
            "degree_preserving_swaps": swaps,
            "null_changed_undirected_pairs": len(
                found.symmetric_difference({tuple(sorted(e)) for e in ng["graph"]["edge_index"].T})
            ),
            "contact_caveat": "Mapped induced graph; backbone gap/chain-break exclusions retained; chemistry proximity is not a confirmed interaction",
        }
    return rows, matrices, groups, null_groups, audits, evidence


def probe_fit(matrix, rows, roles, alphas, seed, scaled=True, nonlinear=False):
    target = fit_targets(rows, roles["train"])
    scaler = StandardScaler().fit(matrix[roles["train"]]) if scaled else None
    x = scaler.transform(matrix).astype(float) if scaler else matrix.astype(float)
    best = None
    candidates = []
    for alpha in alphas:
        model = (
            MLPRegressor(
                hidden_layer_sizes=(16,),
                alpha=alpha,
                max_iter=200,
                random_state=seed,
                early_stopping=False,
            )
            if nonlinear
            else Ridge(alpha=alpha, solver="lsqr", tol=1e-8, max_iter=10000)
        )
        with warnings.catch_warnings(record=True) as caught:
            model.fit(x[roles["train"]], target[roles["train"]])
        score = macro_score(rows, roles["validation"], model.predict(x[roles["validation"]]))
        candidates.append(
            {"alpha": alpha, "validation_rho": score, "warnings": [str(w.message) for w in caught]}
        )
        if best is None or score > best[0]:
            best = score, model, alpha
    model = best[1]
    if nonlinear:
        state = {
            **{f"weight_{i}": w for i, w in enumerate(model.coefs_)},
            **{f"bias_{i}": w for i, w in enumerate(model.intercepts_)},
        }
        parameters = sum(w.size for w in model.coefs_ + model.intercepts_)
    else:
        state = {"coefficient": model.coef_, "intercept": np.asarray(model.intercept_)}
        parameters = model.coef_.size + 1
    if scaler:
        state.update(mean=scaler.mean_, scale=scaler.scale_)
    prediction = model.predict(x[roles["test"]])
    assert np.array_equal(prediction, model.predict(x[roles["test"]]))
    return (
        prediction,
        {
            "alpha": best[2],
            "validation_score": best[0],
            "train_score": macro_score(rows, roles["train"], model.predict(x[roles["train"]])),
            "parameters": parameters,
            "candidates": candidates,
            "scaled": scaled,
            "width": matrix.shape[1],
        },
        state,
    )


def run():
    import torch

    config = json.loads(Path("configs/m17.json").read_text())
    for p, digest in json.loads(Path("benchmarks/m17/preregistration.json").read_text()).items():
        assert checksum(p) == digest, p
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    weights = Path(config["cache"]) / "weights"
    weights.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, matrices, groups, nulls, graph_audit, evidence = prepare()
    write_json(output / "graph-audit.json", graph_audit)
    write_json(output / "feature-provenance.json", evidence)
    write_json(output / "feature-widths.json", {k: v.shape[1] for k, v in matrices.items()})
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    roles_manifest = json.loads(Path("benchmarks/m17/split-manifest.json").read_text())["folds"]
    records = []
    predictions = []
    probes = {
        "structural_ridge": ("structural", True, False),
        "biochemical_ridge": ("sequence", True, False),
        "esm_local_ridge": ("esm_sites", True, False),
        "esm_global_ridge": ("esm_global", True, False),
        "esm_unscaled_ridge": ("esm_sites", False, False),
        "wt_only_ridge": ("esm_wt", True, False),
        "mpnn_local_ridge": ("mpnn_local", True, False),
        "mpnn_global_ridge": ("mpnn_global", True, False),
        "structural_mlp": ("structural", True, True),
        "esm_mlp": ("esm_sites", True, True),
    }
    for case_index, case in enumerate(roles_manifest):
        roles = {k: np.array([lookup[x] for x in ids]) for k, ids in case["roles"].items()}
        modes = list(probes) if case["study"] == "H1" else ["structural_ridge", "esm_local_ridge"]
        if case["study"] == "H1" and case["fraction"] in config["graph_fractions"]:
            modes += [
                "original_gnn",
                "edge_full",
                "zero_message",
                "degree_null",
                "edge_shuffled_chemistry",
            ]
        for mode in modes:
            t = time.perf_counter()
            key = f"{case_index:03d}_{mode}"
            if mode in probes:
                name, scaled, nonlinear = probes[mode]
                pred, selection, state = probe_fit(
                    matrices[name], rows, roles, config["alphas"], case["seed"], scaled, nonlinear
                )
                state_path = weights / (key + ".npz")
                np.savez_compressed(state_path, **state)
            else:
                gg = nulls if mode == "degree_null" else groups
                pred, selection, state = graph_fit(
                    gg,
                    rows,
                    roles,
                    config["training"],
                    case["seed"],
                    "edge_full" if mode == "degree_null" else mode,
                    config["controls_seed"],
                )
                state_path = weights / (key + ".pt")
                torch.save(state, state_path)
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite held-out prediction")
            metadata = {k: case[k] for k in ("study", "fold", "protocol", "seed", "fraction")}
            record = {
                **metadata,
                "model": mode,
                "n_train": len(roles["train"]),
                "selection": selection,
                "per_protein": per_protein(rows, roles["test"], pred),
                "seconds": time.perf_counter() - t,
                "state_path": str(state_path),
                "state_sha256": checksum(state_path),
            }
            records.append(record)
            predictions.extend(
                {
                    **metadata,
                    "model": mode,
                    "id": rows[i]["id"],
                    "protein_id": rows[i]["protein_id"],
                    "family": rows[i]["family"],
                    "target": rows[i]["target"],
                    "mutation_count": len(rows[i]["mutations"]),
                    "prediction": float(v),
                }
                for i, v in zip(roles["test"], pred, strict=True)
            )
        write_json(output / "runs.json", records)
        print(
            case_index,
            case["study"],
            case["fold"],
            case["fraction"],
            len(records),
            round(time.perf_counter() - started, 1),
            flush=True,
        )
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    # Independent fresh selected-model replay on one prespecified full-label fold.
    case = next(
        c for c in roles_manifest if c["study"] == "H1" and c["fraction"] == 1 and c["seed"] == 0
    )
    roles = {k: np.array([lookup[x] for x in ids]) for k, ids in case["roles"].items()}
    pred, _, _ = graph_fit(
        groups, rows, roles, config["training"], 0, "edge_full", config["controls_seed"]
    )
    original = [
        p["prediction"]
        for p in predictions
        if p["study"] == "H1"
        and p["fold"] == case["fold"]
        and p["seed"] == 0
        and p["fraction"] == 1
        and p["model"] == "edge_full"
    ]
    assert np.array_equal(pred, original)
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "preregistration": json.loads(Path("benchmarks/m17/preregistration.json").read_text()),
            "seconds": time.perf_counter() - started,
            "device": "cpu",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "biopython")
            },
            "fits": len(records),
            "held_out_predictions": len(predictions),
            "fresh_edge_refit_exact": True,
            "external_outcomes_used": False,
            "source_sha256": {
                str(p): checksum(p) for p in Path("src/proteinmod").glob("mechanism*.py")
            },
        },
    )


if __name__ == "__main__":
    run()
