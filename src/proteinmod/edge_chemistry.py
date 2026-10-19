"""M14 heavy-atom candidates on unchanged, validated M2 topology (Å)."""

import json
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, verify_file
from proteinmod.structure_mapping import map_reference, parse_structure

CHARGED = {"K": ("NZ",), "R": ("NE", "NH1", "NH2"), "D": ("OD1", "OD2"), "E": ("OE1", "OE2")}
CARBONS = {
    "A": ("CB",),
    "V": ("CB", "CG1", "CG2"),
    "I": ("CB", "CG1", "CG2", "CD1"),
    "L": ("CB", "CG", "CD1", "CD2"),
    "M": ("CB", "CG", "CE"),
    "F": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "W": ("CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}
EDGE_NAMES = (
    [f"rbf_{i}" for i in range(16)]
    + ["distance_over_8", "spatial", "backbone"]
    + [
        "salt_candidate",
        "hydrophobic_candidate",
        "backbone_polar_candidate",
        "salt_valid",
        "hydrophobic_valid",
        "backbone_polar_valid",
        "both_charged",
        "opposite_charge",
        "both_hydrophobic",
        "both_aromatic",
    ]
)


def minimum(a, b, left, right):
    return min(np.linalg.norm(a[x] - b[y]) for x in left for y in right)


def interaction(a, b, aa, bb, sequential=False):
    """Three symmetric candidates and explicit validity, never confirmed hydrogen bonds."""
    value = np.zeros(3, np.float32)
    valid = np.ones(3, np.float32)
    opposite = (aa in "KR" and bb in "DE") or (bb in "KR" and aa in "DE")
    if opposite:
        left, right = CHARGED[aa], CHARGED[bb]
        valid[0] = all(x in a for x in left) and all(x in b for x in right)
        if valid[0]:
            value[0] = minimum(a, b, left, right) <= 4.0
    if aa in CARBONS and bb in CARBONS:
        left, right = CARBONS[aa], CARBONS[bb]
        valid[1] = all(x in a for x in left) and all(x in b for x in right)
        if valid[1]:
            value[1] = minimum(a, b, left, right) <= 4.5
    if not sequential:
        # Hydrogens are absent from M2. Restrict to carbonyl geometry-qualified proximity.
        candidates = [(a, b, bb), (b, a, aa)]
        for acceptor, donor, donor_aa in candidates:
            if donor_aa == "P":
                continue
            if not ("C" in acceptor and "O" in acceptor and "N" in donor):
                valid[2] = 0
                continue
            u, v = acceptor["C"] - acceptor["O"], donor["N"] - acceptor["O"]
            denominator = np.linalg.norm(u) * np.linalg.norm(v)
            if denominator < 1e-8:
                valid[2] = 0
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1)))
            value[2] = max(value[2], 2.5 <= np.linalg.norm(v) <= 3.5 and angle >= 120)
    value *= valid
    categories = [
        aa in "KRDE" and bb in "KRDE",
        opposite,
        aa in CARBONS and bb in CARBONS,
        aa in "FWY" and bb in "FWY",
    ]
    return np.r_[value, valid, categories].astype(np.float32)


def annotate(graph, atoms):
    if len(atoms) != len(graph["nodes"]):
        raise ValueError("Atom rows must match graph nodes")
    distances = graph["edge_features"][:, 0] * 8
    rbf = np.exp(-(((distances[:, None] - np.linspace(0, 12, 16)) / 0.8) ** 2))
    chemistry = []
    for source_index, target_index in graph["edge_index"].T:
        left, right = graph["nodes"][source_index], graph["nodes"][target_index]
        chemistry.append(
            interaction(
                atoms[source_index],
                atoms[target_index],
                left["reference_aa"],
                right["reference_aa"],
                abs(left["polymer_position"] - right["polymer_position"]) == 1,
            )
        )
    values = np.c_[rbf, graph["edge_features"], np.asarray(chemistry)].astype(np.float32)
    if values.shape != (len(distances), len(EDGE_NAMES)) or not np.isfinite(values).all():
        raise ValueError("Invalid edge chemistry dimensions/values")
    return values


def mapped_atoms(directory, graph):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    config = manifest["config"]
    verify_file(config["structure"]["path"], config["structure"])
    parsed = parse_structure(config["structure"]["path"], config["mapping"].get("model_index", 0))
    rows, audit = map_reference(parsed, config["reference"]["sequence"], config["mapping"])
    if not audit["accepted"]:
        raise ValueError("Full atom mapping rejected")
    atoms = []
    for node in graph["nodes"]:
        row = rows[node["reference_position"] - 1]
        for key in (
            "reference_aa",
            "author_number",
            "insertion_code",
            "polymer_position",
            "status",
        ):
            if row[key] != node[key]:
                raise ValueError(f"M2/full-atom mapping differs: {key}")
        atoms.append({a.name: a.coord.astype(float) for a in row["record"]["residue"]})
    return atoms, {
        "coordinate_sha256": checksum(config["structure"]["path"]),
        "mapping": audit,
        "confidence": graph["audit"]["confidence_semantics"],
        "manifest_sha256": checksum(Path(directory) / "manifest.json"),
    }


def controlled_graph(graph, features, mode, seed):
    edges, values = graph["edge_index"].copy(), features.copy()
    if mode == "edge_distance":
        values[:, 17:] = 0
    elif mode == "edge_no_chemistry":
        values[:, 19:] = 0
    elif mode == "edge_shuffled_chemistry":
        keys = sorted({tuple(sorted(e)) for e in edges.T})
        lookup = {key: i for i, key in enumerate(keys)}
        chemistry = {tuple(sorted(e)): v[19:].copy() for e, v in zip(edges.T, values, strict=True)}
        permutation = np.random.default_rng(seed).permutation(len(keys))
        for i, edge in enumerate(edges.T):
            values[i, 19:] = chemistry[keys[permutation[lookup[tuple(sorted(edge))]]]]
    elif mode == "edge_shuffled_topology":
        permutation = np.random.default_rng(seed).permutation(len(graph["nodes"]))
        edges = permutation[edges]
    elif mode not in ("edge_full", "zero_message"):
        raise ValueError("Unknown edge control")
    return edges, values
