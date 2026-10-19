"""Outcome-blind masked-reference-v1 extraction; never an accepted M2 artifact."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from proteinmod.admission_masks import compatible_positions
from proteinmod.data import checksum
from proteinmod.structural_graph import AA, DESCRIPTORS, IDENTITY, PROPERTIES
from proteinmod.structure_features import contact_graph, residue_features
from proteinmod.structure_mapping import map_reference, parse_structure

VERSION = "masked-reference-v1"


def validate_mask(rows, reference, eligible):
    if (
        not eligible
        or any(type(p) is not int for p in eligible)
        or eligible != sorted(set(eligible))
        or eligible[0] < 1
        or eligible[-1] > len(reference)
    ):
        raise ValueError("Mask must be nonempty, sorted, unique integer reference positions")
    if len(rows) != len(reference) or any(
        r["reference_position"] != p or r["reference_aa"] != reference[p - 1]
        for p, r in enumerate(rows, 1)
    ):
        raise ValueError("Unique complete reference mapping required")
    if set(eligible) - set(compatible_positions(rows)):
        raise ValueError("Every eligible site requires exact canonical identity and observed CA")


def extract(path, reference, eligible, expected_sha256, chain="A"):
    if checksum(path) != expected_sha256:
        raise ValueError("Coordinate checksum mismatch")
    parsed = parse_structure(path, 0)
    rows, full_audit = map_reference(
        parsed,
        reference,
        {"chain": chain, "min_coverage": 1.0, "min_identity": 1.0, "allowed_mismatches": []},
    )
    validate_mask(rows, reference, eligible)
    features, sasa = residue_features(
        parsed, rows, {"sasa_points": 960, "dssp": "off"}, path, "experimental_bfactor"
    )
    # Preserve full indices: contact_graph uses row indices as reference positions.
    masked_rows = [
        {**r, "has_ca": r["has_ca"] and r["reference_position"] in eligible} for r in rows
    ]
    edges, breaks = contact_graph(masked_rows, features, 8.0)
    for r in features:
        if r["reference_position"] not in eligible:
            r.update(
                ca=None, asa_a2=None, rsa=None, contact_degree=None, nonlocal_contact_degree=None
            )
    nodes = [features[p - 1] for p in eligible]
    lookup = {p: i for i, p in enumerate(eligible)}
    pairs, attrs = [], []
    for e in edges:
        a, b = lookup[e["source"]], lookup[e["target"]]
        pairs.extend([(a, b), (b, a)])
        attrs.extend([[e["distance_a"] / 8, e["spatial"], e["backbone"]]] * 2)
    edge_index = np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
    edge_features = np.asarray(attrs, dtype=np.float32).reshape(-1, 3)
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = 1
    degree = adjacency.sum(axis=1)
    values = np.array([[r[k] if r[k] is not None else 0 for k in DESCRIPTORS] for r in nodes])
    missing = np.array([[r[k] is None for k in DESCRIPTORS] for r in nodes])
    desc = np.c_[values / [1, 20, 20, 100, 100], missing].astype(np.float32)
    identity = IDENTITY[[AA[r["reference_aa"]] for r in nodes]]
    properties = PROPERTIES[[AA[r["reference_aa"]] for r in nodes]]
    atoms = [
        {a.name: a.coord.astype(float) for a in rows[p - 1]["record"]["residue"]} for p in eligible
    ]
    audit = {
        "scope": VERSION,
        "masked_accepted": True,
        "full_reference_M2_audit": full_audit,
        "reference": reference,
        "coordinate_sha256": expected_sha256,
        "eligible_positions": eligible,
        "eligible_identity": 1.0,
        "eligible_CA_coverage": 1.0,
        "excluded_positions": [p for p in range(1, len(reference) + 1) if p not in lookup],
        "nodes": len(nodes),
        "undirected_edges": len(edges),
        "connected_components": int(connected_components(csr_matrix(adjacency), directed=False)[0]),
        "isolated_nodes": int(sum(degree == 0)),
        "descriptor_missing_nodes": {
            k: int(missing[:, i].sum()) for i, k in enumerate(DESCRIPTORS)
        },
        "confidence_semantics": "experimental_bfactor",
        "ASA": sasa,
        "contacts": "eligible induced graph; no invented WT atoms; deposited context ASA",
        "backbone_breaks": breaks,
        "reference_construct_sequence_compatible": not full_audit["differences"],
    }
    graph = {
        "nodes": nodes,
        "lookup": lookup,
        "residues": features,
        "static": np.c_[identity, properties, desc],
        "neighborhood": (adjacency / np.maximum(degree[:, None], 1)) @ np.c_[identity, desc],
        "edge_index": edge_index,
        "edge_features": edge_features,
        "xyz": np.array([r["ca"] for r in nodes]),
        "audit": audit,
    }
    return graph, atoms, parsed, rows
