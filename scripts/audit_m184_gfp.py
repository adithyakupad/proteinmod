"""Outcome-blind comparison of exact-assay-WT prediction with deposited GFP geometry."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, write_json
from proteinmod.edge_chemistry import annotate, mapped_atoms
from proteinmod.masked_structure import extract
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_mapping import map_reference, parse_structure
from proteinmod.structure_pipeline import run_structure


def align_coordinates(mobile, target):
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Matched N-by-3 coordinates required")
    if len(mobile) < 3 or not np.isfinite([mobile, target]).all():
        raise ValueError("At least three finite matched coordinates required")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    correction = np.diag([1, 1, np.linalg.det(left @ right)])
    rotation = left @ correction @ right
    return rotation, mobile_center, target_center


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "rms": float(np.sqrt(np.mean(values**2))),
    }


def audit(config, output):
    if config.get("dataset_variants") or config["structure"]["confidence"] != "unknown":
        raise ValueError("Genotype-only diagnostic; API confidence scale remains unverified")
    result = run_structure(config, render=False)
    reference = config["reference"]["sequence"]
    predicted = load_graph(config["structure_run"]["output"], reference)
    atoms, _ = mapped_atoms(config["structure_run"]["output"], predicted)
    edge_features = annotate(predicted, atoms)
    assert predicted["static"].shape == (237, 34)
    assert edge_features.shape == (2220, 29) and np.isfinite(edge_features).all()
    historical = json.loads(Path("benchmarks/m182/evidence/structure-mask-audits.json").read_text())
    digest = next(row["coordinate_sha256"] for row in historical if row["pdb"] == "1GFL")
    eligible = [p for p in range(1, 238) if p not in {63, 79, *range(230, 238)}]
    original, _, _, original_rows = extract(
        Path("data/cache/m182/source/1GFL.cif"), reference, eligible, digest
    )
    parsed = parse_structure(config["structure"]["path"])
    predicted_rows, mapping = map_reference(parsed, reference, config["mapping"])
    assert mapping["accepted"] and not mapping["differences"]
    mobile = predicted["xyz"][[predicted["lookup"][p] for p in eligible]]
    target = original["xyz"]
    rotation, center, target_center = align_coordinates(mobile, target)

    def displacements(positions, atom_names):
        differences = []
        for position in positions:
            old_atoms = original_rows[position - 1]["record"]["residue"]
            new_atoms = predicted_rows[position - 1]["record"]["residue"]
            for name in atom_names:
                if name in old_atoms and name in new_atoms:
                    aligned = (new_atoms[name].coord - center) @ rotation + target_center
                    differences.append(np.linalg.norm(aligned - old_atoms[name].coord))
        return differences

    old_distances = np.linalg.norm(target[:, None] - target[None, :], axis=2)
    new_distances = np.linalg.norm(mobile[:, None] - mobile[None, :], axis=2)
    old_contacts = set(map(tuple, np.argwhere(np.triu(old_distances <= 8, k=1))))
    new_contacts = set(map(tuple, np.argwhere(np.triu(new_distances <= 8, k=1))))
    neighbors = [1, 15, 17, 59, 60, 61, 62, 64, 65, 66, 67, 76, 77, 78, 80, 81, 120, 122, 193, 195]
    mutations = json.loads(Path("benchmarks/m182/GFP-genotype-audit.json").read_text())
    assert config["variants"] == [
        {"id": r["id"], "mutations": r["mutations"]} for r in mutations["selected_candidates"]
    ]
    pairs = []
    for row in config["variants"]:
        dynamic, features, reasons = variant_inputs(row["mutations"], reference, predicted)
        reversed_dynamic, reversed_features, reversed_reasons = variant_inputs(
            row["mutations"][::-1], reference, predicted
        )
        assert not reasons and not reversed_reasons
        assert dynamic.shape == (237, 25) and features.shape == (270,)
        assert np.array_equal(dynamic, reversed_dynamic)
        assert np.array_equal(features, reversed_features)
        positions = [int(token[1:-1]) for token in row["mutations"]]
        assert set(positions) <= set(eligible)
        distances = []
        for first, second in itertools.combinations(sorted(positions), 2):
            i, j = eligible.index(first), eligible.index(second)
            distances.append(
                {
                    "positions": [first, second],
                    "1GFL_a": old_distances[i, j],
                    "ESMFold_a": new_distances[i, j],
                }
            )
        pairs.append({"id": row["id"], "pairs": distances})
    asa_differences = [
        predicted["nodes"][predicted["lookup"][p]]["rsa"] - original["nodes"][i]["rsa"]
        for i, p in enumerate(eligible)
    ]
    raw = [r["raw_ca_b_iso_or_equiv"] for r in predicted["nodes"]]
    assert all(r["plddt"] is None and r["bfactor_a2"] is None for r in predicted["nodes"])
    evidence = {
        "scope": "Diagnostic only; no source substitution or admission decision based on post-hoc geometry cutoffs",
        "config_sha256": checksum("configs/m184-gfp-diagnostic.json"),
        "coordinate_sha256": checksum(config["structure"]["path"]),
        "strict_M2": result,
        "common_reference_positions": eligible,
        "common_CA_displacement_a": summary(displacements(eligible, ["CA"])),
        "common_backbone_displacement_a": summary(displacements(eligible, ["N", "CA", "C", "O"])),
        "excluded_mismatches_backbone_displacement_a": summary(
            displacements([63, 79], ["N", "CA", "C", "O"])
        ),
        "prior_20_neighbors_backbone_displacement_a": summary(
            displacements(neighbors, ["N", "CA", "C", "O"])
        ),
        "common_identity_sidechain_displacement_a": summary(
            displacements(
                eligible,
                [
                    "CB",
                    "CG",
                    "CG1",
                    "CG2",
                    "CD",
                    "CD1",
                    "CD2",
                    "CE",
                    "CE1",
                    "CE2",
                    "CZ",
                    "OG",
                    "OG1",
                    "SG",
                    "NZ",
                    "OE1",
                    "OE2",
                    "OD1",
                    "OD2",
                    "NE",
                    "NE1",
                    "NE2",
                    "NH1",
                    "NH2",
                    "ND1",
                    "ND2",
                    "OH",
                ],
            )
        ),
        "common_CA_contacts_8a": {
            "1GFL": len(old_contacts),
            "ESMFold": len(new_contacts),
            "intersection": len(old_contacts & new_contacts),
            "jaccard": len(old_contacts & new_contacts) / len(old_contacts | new_contacts),
        },
        "absolute_RSA_difference": summary(np.abs(asa_differences)),
        "RSA_context_caveat": "1GFL includes deposited other chains/ligands; prediction is a monomer. Not an isolated background substitution effect.",
        "prediction_raw_CA_field": {
            "min": min(raw),
            "max": max(raw),
            "semantics": "unknown API normalization; neither experimental B-factor nor verified 0-100 pLDDT",
        },
        "graph_audit": predicted["audit"],
        "validated_genotypes": len(pairs),
        "model_input_dimensions": {"static": 34, "dynamic": 25, "ridge": 270, "edge": 29},
        "directed_chemical_edges_finite": len(edge_features),
        "mutation_order_exact_invariance": True,
        "mutation_pair_distances": pairs,
        "candidate_target_cells_accessed": False,
        "models_fit": 0,
        "confirmation_predictions": 0,
        "maturation_caveat": "Exact canonical residue identity is not validation of mature chromophore chemistry or experimental mutant rearrangements.",
    }
    write_json(output, evidence)
    print(
        json.dumps(
            {
                k: evidence[k]
                for k in (
                    "strict_M2",
                    "common_CA_displacement_a",
                    "common_CA_contacts_8a",
                    "absolute_RSA_difference",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/m184/GFP-geometry-audit.json")
    )
    args = parser.parse_args()
    audit(json.loads(Path("configs/m184-gfp-diagnostic.json").read_text()), args.output)
