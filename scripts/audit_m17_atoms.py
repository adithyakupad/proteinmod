"""Physical atom/missingness audit independent of model scores; no new fits."""

import json
from pathlib import Path

from Bio.PDB.ic_data import residue_atom_bond_state

from proteinmod.data import write_json
from proteinmod.edge_chemistry import annotate, mapped_atoms
from proteinmod.structural_graph import load_graph


def audit():
    config = json.loads(Path("configs/m12.json").read_text())
    records = {}
    for spec in config["proteins"]:
        graph = load_graph(spec["structure_artifacts"], spec["reference"]["sequence"])
        atoms, provenance = mapped_atoms(spec["structure_artifacts"], graph)
        assert len(atoms) == len(graph["nodes"])
        absent = []
        for node, observed in zip(graph["nodes"], atoms, strict=True):
            aa = node["reference_aa"]
            expected = (
                set(residue_atom_bond_state["X"]) | set(residue_atom_bond_state.get(aa, {}))
            ) - {"H", "OXT"}
            if aa == "G":
                expected.discard("CB")
            missing = sorted(expected - set(observed))
            if missing:
                absent.append(
                    {"position": node["reference_position"], "aa": aa, "missing_atoms": missing}
                )
        features = annotate(graph, atoms)
        records[spec["protein_id"]] = {
            "nodes": len(atoms),
            "missing_heavy_atoms": absent,
            "candidate_directed_edges": features[:, 19:22].sum(0).astype(int).tolist(),
            "unknown_interaction_directed_edges": (1 - features[:, 22:25])
            .sum(0)
            .astype(int)
            .tolist(),
            "semantics": "Canonical heavy atoms, excluding terminal OXT/hydrogens; absent atoms are unknown interactions, not no interaction",
            "coordinate_sha256": provenance["coordinate_sha256"],
        }
    write_json("runs/m17/atom-audit.json", records)


if __name__ == "__main__":
    audit()
