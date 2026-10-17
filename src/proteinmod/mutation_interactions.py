"""Symmetric, background-independent pair features; no outcomes enter construction."""

import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from proteinmod.data import AMINO_ACIDS
from proteinmod.structural_graph import AA, IDENTITY, PROPERTIES
from proteinmod.structure_features import parse_mutation_set

SEQUENCE_WIDTH = 345
GEOMETRY_WIDTH = 15
PAIR_WIDTH = 420
GEOMETRY_NAMES = (
    "distance_over_30_a",
    "inverse_distance_a",
    "spatial_contact",
    "shared_neighbor_jaccard",
    "same_component",
    "path_over_10",
    "mapped_pair",
    "path_observed",
    "plddt_mean_over_100",
    "plddt_observed",
    "bfactor_mean_over_100_a2",
    "bfactor_observed",
    "source_af",
    "source_experimental",
    "source_unknown",
)


def topology(graph):
    n = len(graph["nodes"])
    adj = np.zeros((n, n), bool)
    adj[tuple(graph["edge_index"])] = True
    _, components = connected_components(csr_matrix(adj), directed=False)
    return adj, components, shortest_path(csr_matrix(adj), directed=False, unweighted=True)


def geometry(a, b, graph, precomputed):
    out = np.zeros(GEOMETRY_WIDTH, np.float32)
    semantics = graph["audit"]["confidence_semantics"]
    out[{"alphafold_plddt": 12, "experimental_bfactor": 13, "unknown": 14}[semantics]] = 1
    if a not in graph["lookup"] or b not in graph["lookup"]:
        return out
    i, j = graph["lookup"][a], graph["lookup"][b]
    adj, components, paths = precomputed
    distance = np.linalg.norm(graph["xyz"][i] - graph["xyz"][j])
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Invalid mapped pair distance")
    neighbors_a, neighbors_b = set(np.flatnonzero(adj[i])), set(np.flatnonzero(adj[j]))
    out[:8] = [
        distance / 30,
        1 / (1 + distance),
        distance <= 8,
        len(neighbors_a & neighbors_b) / max(len(neighbors_a | neighbors_b), 1),
        components[i] == components[j],
        paths[i, j] / 10 if np.isfinite(paths[i, j]) else 0,
        1,
        np.isfinite(paths[i, j]),
    ]
    for key, offset, expected in [
        ("plddt", 8, "alphafold_plddt"),
        ("bfactor_a2", 10, "experimental_bfactor"),
    ]:
        values = [graph["nodes"][k][key] for k in (i, j) if graph["nodes"][k][key] is not None]
        if values and (
            semantics != expected
            or any(not np.isfinite(v) or v < 0 or (key == "plddt" and v > 100) for v in values)
        ):
            raise ValueError("Invalid typed pair confidence")
        out[offset] = np.mean(values) / 100 if values else 0
        out[offset + 1] = len(values) / 2
    return out


def mutation_vector(token):
    return np.r_[
        IDENTITY[AA[token[0]]],
        IDENTITY[AA[token[-1]]],
        PROPERTIES[AA[token[-1]]] - PROPERTIES[AA[token[0]]],
    ]


def pair_features(tokens, reference, graph, positions, control="real", seed=0, context=None):
    tokens = parse_mutation_set(tokens, reference)
    if control not in ("real", "zero3d", "shuffled_distance", "shuffled_biochemical"):
        raise ValueError("Unknown interaction control")
    physical = topology(graph)
    pairs = list(itertools.combinations(sorted(positions), 2))
    geom = {p: geometry(*p, graph, physical) for p in pairs}
    rng = np.random.default_rng(seed)
    if control == "shuffled_distance":
        order = rng.permutation(len(pairs))
        geom = {p: geom[pairs[j]].copy() for p, j in zip(pairs, order, strict=True)}
    bio = {}
    for p in sorted(positions):
        alleles = [a for a in AMINO_ACIDS if a != reference[p - 1]]
        shuffled = rng.permutation(alleles)
        bio.update(
            {
                f"{reference[p - 1]}{p}{a}": f"{reference[p - 1]}{p}{b}"
                for a, b in zip(alleles, shuffled, strict=True)
            }
        )
    output = []
    for left, right in itertools.combinations(tokens, 2):
        a, b = int(left[1:-1]), int(right[1:-1])
        key = tuple(sorted((a, b)))
        if key not in geom:
            raise ValueError("Mutation position outside declared landscape")
        l, r = (bio[left], bio[right]) if control == "shuffled_biochemical" else (left, right)
        x, y = mutation_vector(l), mutation_vector(r)
        outer = np.outer(IDENTITY[AA[l[-1]]], IDENTITY[AA[r[-1]]])
        outer = outer + outer.T - np.diag(np.diag(outer))
        seq = np.r_[
            1,
            x + y,
            np.abs(x - y),
            x * y,
            outer[np.triu_indices(20)],
            abs(a - b) / max(len(reference) - 1, 1),
            np.exp(-abs(a - b) / 10),
        ]
        geo = np.zeros(GEOMETRY_WIDTH) if control == "zero3d" else geom[key]
        row = np.r_[seq, geo, np.outer(x[-4:] * y[-4:], geo).ravel()]
        if context is not None:
            h, k = context[left], context[right]
            row = np.r_[row, h + k, np.abs(h - k), h * k]
        if not np.isfinite(row).all():
            raise ValueError("Nonfinite interaction feature")
        output.append(row)
    width = PAIR_WIDTH + (3 * len(next(iter(context.values()))) if context else 0)
    return np.array(output, dtype=np.float32).reshape(-1, width)
