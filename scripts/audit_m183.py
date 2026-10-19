"""Real geometry and common-cohort audit. Reads only archived genotype projections."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import Bio
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from proteinmod.data import write_json
from proteinmod.edge_chemistry import CARBONS, CHARGED, EDGE_NAMES, annotate
from proteinmod.masked_structure import extract
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_features import contact_graph, residue_features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("benchmarks/m183"))
out = parser.parse_args().output
out.mkdir(parents=True, exist_ok=True)
refs = {
    r["id"]: r["sequence"]
    for r in json.loads(Path("benchmarks/m182/reference-panel.json").read_text())[
        "external_candidates"
    ]
}
old = json.loads(Path("benchmarks/m182/evidence/structure-mask-audits.json").read_text())
old_index = json.loads(Path("benchmarks/m18/evidence/blind-index.json").read_text())["selected"]
cohorts = {p: [r for r in old_index if r["protein_id"] == p] for p in ("P0A9X9", "P01053")}
for p, name in [("TEM1", "TEM1-common-mask-audit.json"), ("GFP", "GFP-genotype-audit.json")]:
    cohorts[p] = json.loads((Path("benchmarks/m182") / name).read_text())["selected_candidates"]
results = []
start = time.perf_counter()
for pid, pdb, eligible in [
    ("P0A9X9", "1MJC", list(range(1, 66))),
    ("P01053", "3CI2", list(range(1, 64))),
    ("TEM1", "1XPB", list(range(24, 287))),
    ("GFP", "1GFL", [p for p in range(1, 238) if p not in {63, 79, *range(230, 238)}]),
]:
    root = "m18" if pid.startswith("P") else "m182"
    path = Path(f"data/cache/{root}/source/{pdb}.cif")
    if root == "m18":
        m = json.loads(Path(f"data/cache/m18/graphs/{pid}/manifest.json").read_text())
        expected = m["config"]["structure"]["sha256"]
    else:
        expected = next(r["coordinate_sha256"] for r in old if r["pdb"] == pdb)
    graph, atoms, parsed, rows = extract(path, refs[pid], eligible, expected)
    chem = annotate(graph, atoms)
    raw = MMCIF2Dict(path)
    raw_atoms = {}
    for i, name in enumerate(raw["_atom_site.label_atom_id"]):
        if raw["_atom_site.pdbx_PDB_model_num"][i] != "1":
            continue
        key = (
            raw["_atom_site.auth_asym_id"][i],
            int(raw["_atom_site.auth_seq_id"][i]),
            raw["_atom_site.pdbx_PDB_ins_code"][i].replace("?", "").replace(".", ""),
            name,
        )
        if float(raw["_atom_site.occupancy"][i]) > 0:
            raw_atoms.setdefault(key, []).append(
                [
                    float(raw[k][i])
                    for k in ["_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"]
                ]
            )
    for node, atom_row in zip(graph["nodes"], atoms, strict=True):
        for name, coordinate in atom_row.items():
            key = (node["chain"], node["author_number"], node["insertion_code"], name)
            assert any(np.allclose(coordinate, xyz, atol=1e-4, rtol=0) for xyz in raw_atoms[key])
    edges = graph["edge_index"]
    # Independent all-pair CA distances, and symmetry/atom correspondence checks.
    xyz = np.array([a["CA"] for a in atoms])
    assert np.array_equal(xyz, graph["xyz"])
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    actual = {tuple(e) for e in edges.T}
    assert all((j, i) in actual and i != j for i, j in actual)
    expected_spatial = {
        (i, j) for i in range(len(xyz)) for j in range(len(xyz)) if i != j and distances[i, j] <= 8
    }
    spatial = {tuple(e) for e, a in zip(edges.T, graph["edge_features"], strict=True) if a[1]}
    assert spatial == expected_spatial
    expected_backbone = set()
    for i, left in enumerate(eligible):
        a = rows[left - 1]["record"]
        for j in range(i + 1, len(eligible)):
            b = rows[eligible[j] - 1]["record"]
            if (
                b["polymer_index"] - a["polymer_index"] == 1
                and a["segment"] == b["segment"]
                and "C" in atoms[i]
                and "N" in atoms[j]
                and np.linalg.norm(atoms[i]["C"] - atoms[j]["N"]) <= 1.8
            ):
                expected_backbone.update([(i, j), (j, i)])
    assert actual == expected_spatial | expected_backbone
    assert {
        tuple(e) for e, a in zip(edges.T, graph["edge_features"], strict=True) if a[2]
    } == expected_backbone
    assert np.allclose(graph["edge_features"][:, 0] * 8, distances[edges[0], edges[1]], atol=1e-6)
    assert graph["static"].shape == (len(eligible), 34)
    assert graph["neighborhood"].shape == (len(eligible), 30)
    assert np.isfinite(chem).all() and chem.shape[1] == 29
    selected = cohorts[pid]
    assert len(selected) == 96 and len({r["id"] for r in selected}) == 96
    for r in selected:
        reconstructed = list(refs[pid])
        for token in r["mutations"]:
            reconstructed[int(token[1:-1]) - 1] = token[-1]
        assert "".join(reconstructed) == r["sequence"] and len(reconstructed) <= 1022
        d, f, reasons = variant_inputs(r["mutations"], refs[pid], graph)
        rd, rf, rr = variant_inputs(list(reversed(r["mutations"])), refs[pid], graph)
        assert not reasons and not rr and d.shape == (len(eligible), 25) and f.shape == (270,)
        assert np.array_equal(d, rd) and np.array_equal(f, rf)
        assert not any(k in r for k in ("target", "fitness", "brightness"))
    tensor_hashes = {
        k: hashlib.sha256(graph[k].tobytes()).hexdigest()
        for k in ["static", "neighborhood", "edge_index", "edge_features", "xyz"]
    }
    if pid.startswith("P"):
        original = load_graph(f"data/cache/m18/graphs/{pid}", refs[pid])
        # Full masks reduce exactly to the original M2/M3 model inputs.
        assert all(np.array_equal(graph[k], original[k]) for k in tensor_hashes)
    audit = {
        **graph["audit"],
        "protein_id": pid,
        "pdb": pdb,
        "coordinate_path": str(path),
        "tensor_sha256": tensor_hashes,
        "atoms": sum(map(len, atoms)),
        "chemistry_shape": list(chem.shape),
        "raw_atom_site_coordinates_verified": True,
        "per_residue_missing_chemical_atoms": [
            {
                "position": node["reference_position"],
                "missing": sorted(
                    set(
                        ("N", "CA", "C", "O")
                        + CHARGED.get(node["reference_aa"], ())
                        + CARBONS.get(node["reference_aa"], ())
                    )
                    - set(atom_row)
                ),
            }
            for node, atom_row in zip(graph["nodes"], atoms, strict=True)
        ],
        "chemistry_invalid_directed_edges": {
            n: int(sum(chem[:, EDGE_NAMES.index(n)] == 0))
            for n in ["salt_valid", "hydrophobic_valid", "backbone_polar_valid"]
        },
        "ASA_range_a2": [
            min(r["asa_a2"] for r in graph["nodes"]),
            max(r["asa_a2"] for r in graph["nodes"]),
        ],
        "RSA_range": [min(r["rsa"] for r in graph["nodes"]), max(r["rsa"] for r in graph["nodes"])],
        "model_input_dimensions": {"static": 34, "dynamic": 25, "ridge": 270, "edge": 29},
        "common_candidate_ids": [r["id"] for r in selected],
        "matched_candidates": len(selected),
        "cohort_exclusions_after_mask": [],
        "mutation_order_invariance": True,
        "independent_spatial_edges_verified": True,
        "independent_backbone_edges_verified": True,
    }
    if pid == "GFP":
        fullfeatures, _ = residue_features(
            parsed, rows, {"sasa_points": 960, "dssp": "off"}, path, "experimental_bfactor"
        )
        fulledges, _ = contact_graph(rows, fullfeatures, 8)
        excluded = {63, 79}
        touching = [
            e
            for e in fulledges
            if (e["source"] in excluded) != (e["target"] in excluded) and e["spatial"]
        ]
        neighbors = sorted(
            {e["source"] if e["target"] in excluded else e["target"] for e in touching}
            & set(eligible)
        )
        for p in excluded:
            residue = rows[p - 1]["record"]["residue"]
            residue.parent.detach_child(residue.id)
        removed, _ = residue_features(
            parsed,
            [r for r in rows if r["reference_position"] not in excluded],
            {"sasa_points": 960, "dssp": "off"},
            path,
            "experimental_bfactor",
        )
        bypos = {r["reference_position"]: r for r in removed}
        changes = [
            {
                "position": p,
                "delta_ASA_a2": bypos[p]["asa_a2"] - fullfeatures[p - 1]["asa_a2"],
                "delta_RSA": bypos[p]["rsa"] - fullfeatures[p - 1]["rsa"],
            }
            for p in eligible
        ]
        audit["background_sensitivity"] = {
            "mismatches": ["L63F", "Q79R"],
            "excluded_to_eligible_contacts": len(touching),
            "eligible_neighbors": neighbors,
            "selected_candidates_mutating_contact_neighbors": [
                r["id"] for r in selected if {int(t[1:-1]) for t in r["mutations"]} & set(neighbors)
            ],
            "ASA_affected_sites": [r for r in changes if abs(r["delta_ASA_a2"]) > 1e-8],
            "max_delta_ASA_a2": max(r["delta_ASA_a2"] for r in changes),
            "interpretation": "Deleting the two chain-A mismatched residues is an occlusion sensitivity test, not WT reconstruction; chain B is unchanged. Conformational propagation remains unknown; background gate unresolved.",
        }
    results.append(audit)
    print(pid, len(eligible), audit["undirected_edges"], audit["matched_candidates"], flush=True)
write_json(
    out / "structural-cohort-audit.json",
    {
        "scope": "masked-reference-v1",
        "families": results,
        "common_candidates": 384,
        "candidate_target_cells_accessed": False,
        "models_fit": 0,
        "predictions_generated": 0,
        "runtime_seconds": time.perf_counter() - start,
        "device": "CPU",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "biopython": Bio.__version__,
        "randomness": "none; deterministic geometry/genotype projections",
    },
)
