"""Label-free M3 inputs from checksum-verified M2 residue graphs."""

import csv
import json
from pathlib import Path

import numpy as np
from Bio.Data.IUPACData import protein_weights
from Bio.SeqUtils.ProtParamData import kd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from proteinmod.data import AMINO_ACIDS, checksum
from proteinmod.structure_features import parse_mutation_set

AA = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDENTITY = np.eye(20, dtype=np.float32)
PROPERTIES = np.array(
    [
        [kd[a] / 4.5, protein_weights[a] / 200, (a in "KR") - (a in "DE"), a in "FWY"]
        for a in AMINO_ACIDS
    ],
    dtype=np.float32,
)
DESCRIPTORS = ["rsa", "contact_degree", "nonlocal_contact_degree", "bfactor_a2", "plddt"]


def load_graph(directory, reference):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("residues.json", "contacts.csv", "audit.json"):
        if checksum(directory / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"M2 artifact checksum mismatch: {name}")
    audit = json.loads((directory / "audit.json").read_text())
    residues = json.loads((directory / "residues.json").read_text())
    if not audit["accepted"] or audit["pipeline_status"] != "complete":
        raise ValueError("M2 extraction was not accepted and complete")
    if (
        manifest["config"]["reference"]["sequence"] != reference
        or "".join(residue["reference_aa"] for residue in residues) != reference
    ):
        raise ValueError("M2 reference differs from dataset reference")
    if [residue["reference_position"] for residue in residues] != list(
        range(1, len(reference) + 1)
    ):
        raise ValueError("M2 reference positions must be complete and ordered")
    # Exclude incompatible nodes as well as mutations; never invent WT geometry at those sites.
    nodes = [
        residue
        for residue in residues
        if residue["status"] == "mapped" and residue["ca"] is not None
    ]
    lookup = {residue["reference_position"]: i for i, residue in enumerate(nodes)}
    with (directory / "contacts.csv").open() as handle:
        original = list(csv.DictReader(handle))
    edges, edge_features = [], []
    for contact in original:
        if int(contact["source"]) not in lookup or int(contact["target"]) not in lookup:
            continue
        source_index, target_index = lookup[int(contact["source"])], lookup[int(contact["target"])]
        distance = float(contact["distance_a"])
        if source_index == target_index or not np.isfinite(distance) or distance <= 0:
            raise ValueError("Invalid M2 contact edge")
        edges.extend([(source_index, target_index), (target_index, source_index)])
        edge_features.extend(
            [[distance / 8, contact["spatial"] == "True", contact["backbone"] == "True"]] * 2
        )
    edge_index = np.array(edges, dtype=np.int64).T.reshape(2, -1)
    edge_attributes = np.array(edge_features, dtype=np.float32).reshape(-1, 3)
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    adjacency[edge_index[0], edge_index[1]] = 1
    degree = adjacency.sum(axis=1)
    neighbor_mean = adjacency / np.maximum(degree[:, None], 1)
    # Physical units are fixed; no test labels or fitted dataset statistics enter graph inputs.
    values = np.array(
        [
            [
                residue[descriptor_name] if residue[descriptor_name] is not None else 0
                for descriptor_name in DESCRIPTORS
            ]
            for residue in nodes
        ]
    )
    missing = np.array(
        [[residue[descriptor_name] is None for descriptor_name in DESCRIPTORS] for residue in nodes]
    )
    values = values / np.array([1, 20, 20, 100, 100])
    descriptors = np.concatenate([values, missing], axis=1).astype(np.float32)
    identity = IDENTITY[[AA[residue["reference_aa"]] for residue in nodes]]
    properties = PROPERTIES[[AA[residue["reference_aa"]] for residue in nodes]]
    static = np.concatenate([identity, properties, descriptors], axis=1)
    neighborhood = neighbor_mean @ np.concatenate([identity, descriptors], axis=1)
    components, _ = connected_components(csr_matrix(adjacency), directed=False)
    graph_audit = {
        "mapping": audit,
        "confidence_semantics": manifest["confidence_semantics"],
        "nodes": len(nodes),
        "undirected_edges": len(edges) // 2,
        "connected_components": int(components),
        "isolated_nodes": int(sum(degree == 0)),
        "removed_positions": [
            residue["reference_position"] for residue in residues if residue not in nodes
        ],
        "descriptor_missing_nodes": {
            descriptor_name: int(missing[:, i].sum())
            for i, descriptor_name in enumerate(DESCRIPTORS)
        },
        "m2_manifest_sha256": checksum(directory / "manifest.json"),
        "geometry": "reference construct only; not mutant rearrangements",
        "neighborhood": "identity-compatible induced M2 graph; M2 degree/RSA descriptors retain deposited context",
    }
    return {
        "nodes": nodes,
        "lookup": lookup,
        "residues": residues,
        "static": static,
        "neighborhood": neighborhood,
        "edge_index": edge_index,
        "edge_features": edge_attributes,
        "xyz": np.array([residue["ca"] for residue in nodes]),
        "audit": graph_audit,
    }


def variant_inputs(tokens, reference, graph):
    tokens = parse_mutation_set(tokens, reference)
    positions = [int(t[1:-1]) for t in tokens]
    invalid = [p for p in positions if p not in graph["lookup"]]
    if invalid:
        return (
            None,
            None,
            [{"position": p, "reason": graph["residues"][p - 1]["status"]} for p in invalid],
        )
    node_count = len(graph["nodes"])
    dynamic = np.zeros((node_count, 25), dtype=np.float32)
    sites = []
    for token, position in zip(tokens, positions, strict=True):
        node_index = graph["lookup"][position]
        reference_index, mutant_index = AA[token[0]], AA[token[-1]]
        dynamic[node_index, 0] = 1
        dynamic[node_index, 1:21] = IDENTITY[mutant_index]
        dynamic[node_index, 21:] = PROPERTIES[mutant_index] - PROPERTIES[reference_index]
        sites.append(
            np.concatenate(
                [
                    graph["static"][node_index],
                    dynamic[node_index, 1:],
                    graph["neighborhood"][node_index],
                ]
            )
        )
    width = graph["static"].shape[1] + 24 + graph["neighborhood"].shape[1]
    aggregated = (
        np.concatenate([np.mean(sites, axis=0), np.sum(sites, axis=0), np.max(sites, axis=0)])
        if sites
        else np.zeros(width * 3)
    )
    site_coordinates = graph["xyz"][[graph["lookup"][p] for p in positions]]
    distances = [
        np.linalg.norm(site_coordinates[node_index] - site_coordinates[j]) / 30
        for node_index in range(len(site_coordinates))
        for j in range(node_index)
    ]
    pair = [np.mean(distances), min(distances), max(distances)] if distances else [0, 0, 0]
    features = np.concatenate(
        [aggregated, [len(tokens), len(distances), bool(distances)], pair]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite structural features")
    return dynamic, features, []


def prepare_inputs(rows, reference, graph):
    dynamic, features, valid, exclusions = [], [], [], []
    for i, row in enumerate(rows):
        mutation, feature, reasons = variant_inputs(row["mutations"], reference, graph)
        if reasons:
            exclusions.append(
                {
                    "id": row["id"],
                    "partition": row["partition"],
                    "mutations": list(row["mutations"]),
                    "reasons": reasons,
                }
            )
        else:
            valid.append(i)
            dynamic.append(mutation)
            features.append(feature)
    if not valid:
        raise ValueError("No structurally valid variants")
    return np.array(valid), np.stack(dynamic), np.stack(features), exclusions
