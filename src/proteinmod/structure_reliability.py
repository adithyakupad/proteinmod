"""Typed reference-geometry reliability, never mutation-effect uncertainty."""

import copy
from dataclasses import dataclass

import numpy as np

from proteinmod.structure_features import parse_mutation_set

FIELDS = (
    "source_alphafold",
    "source_experimental",
    "source_unknown",
    "site_plddt_over_100",
    "neighbor_plddt_over_100",
    "site_bfactor_over_100_a2",
    "neighbor_bfactor_over_100_a2",
    "site_plddt_observed",
    "neighbor_plddt_observed",
    "site_bfactor_observed",
    "neighbor_bfactor_observed",
    "coordinate_coverage",
    "missing_coordinate_fraction",
    "mutated_site_coverage",
    "alignment_mismatch_fraction",
    "resolution_over_5_a",
    "resolution_observed",
    "structure_available",
)
CONFIDENCE_COLUMNS = (27, 28, 32, 33)  # M3: B/100, pLDDT/100, and their missing bits.


@dataclass(frozen=True)
class StructureReliability:
    source: str
    coordinate_coverage: float
    alignment_identity: float
    resolution_a: float | None
    provenance: str

    def __post_init__(self):
        if self.source not in ("alphafold_plddt", "experimental_bfactor", "unknown"):
            raise ValueError("Unknown confidence source semantics")
        if any(
            not np.isfinite(v) or not 0 <= v <= 1
            for v in (self.coordinate_coverage, self.alignment_identity)
        ):
            raise ValueError("Invalid mapping reliability")
        if self.resolution_a is not None and (
            self.source != "experimental_bfactor"
            or not np.isfinite(self.resolution_a)
            or self.resolution_a <= 0
        ):
            raise ValueError("Resolution is an experimental Angstrom measurement")
        if not self.provenance:
            raise ValueError("Confidence provenance required")

    @classmethod
    def from_graph(cls, graph):
        audit = graph["audit"]
        mapping = audit["mapping"]
        source = audit["confidence_semantics"]
        resolution = mapping.get("deposition", {}).get("resolution")
        result = cls(
            source,
            mapping["coordinate_coverage"],
            mapping["aligned_identity"],
            resolution if source == "experimental_bfactor" else None,
            audit["m2_manifest_sha256"],
        )
        for node in graph["nodes"]:
            for key, expected in (
                ("plddt", "alphafold_plddt"),
                ("bfactor_a2", "experimental_bfactor"),
            ):
                value = node[key]
                if value is not None and (
                    source != expected
                    or not np.isfinite(value)
                    or value < 0
                    or (key == "plddt" and value > 100)
                ):
                    raise ValueError("B-factor and pLDDT semantics cannot be interchanged")
        return result


def confidence_graph(graph, profile="real", seed=0):
    """Mask legacy confidence too; shuffled values/masks move together within reference."""
    if profile not in ("real", "neutral", "shuffled"):
        raise ValueError("Unknown confidence profile")
    StructureReliability.from_graph(graph)
    result = {**graph, "static": graph["static"].copy(), "nodes": copy.deepcopy(graph["nodes"])}
    if result["static"].shape[1] != 34:
        raise ValueError("Expected M3 static descriptor schema")
    if profile == "neutral":
        result["static"][:, CONFIDENCE_COLUMNS] = [0, 0, 1, 1]
    elif profile == "shuffled":
        permutation = np.random.default_rng(seed).permutation(len(graph["nodes"]))
        result["static"][:, CONFIDENCE_COLUMNS] = graph["static"][permutation][
            :, CONFIDENCE_COLUMNS
        ]
        for j, i in enumerate(permutation):
            for key in ("plddt", "bfactor_a2"):
                result["nodes"][j][key] = graph["nodes"][i][key]
    return result


def reliability_features(tokens, reference, graph):
    mutations = parse_mutation_set(tokens, reference)
    out = np.zeros(len(FIELDS), dtype=np.float32)
    if graph is None:
        out[2], out[12] = 1, 1
        return out
    info = StructureReliability.from_graph(graph)
    out[("alphafold_plddt", "experimental_bfactor", "unknown").index(info.source)] = 1
    sites = [graph["lookup"][int(t[1:-1])] for t in mutations if int(t[1:-1]) in graph["lookup"]]
    neighbors = sorted(
        {
            j
            for i in sites
            for j, xyz in enumerate(graph["xyz"])
            if np.linalg.norm(graph["xyz"][i] - xyz) <= 8
        }
    )
    for key, value_offset, mask_offset in (("plddt", 3, 7), ("bfactor_a2", 5, 9)):
        for offset, region in enumerate((sites, neighbors)):
            observed = [
                graph["nodes"][i][key] for i in region if graph["nodes"][i][key] is not None
            ]
            out[value_offset + offset] = np.mean(observed) / 100 if observed else 0
            out[mask_offset + offset] = len(observed) / max(len(region), 1)
    out[11:15] = [
        info.coordinate_coverage,
        1 - info.coordinate_coverage,
        len(sites) / len(mutations) if mutations else 0,
        1 - info.alignment_identity,
    ]
    out[15:18] = [
        info.resolution_a / 5 if info.resolution_a else 0,
        info.resolution_a is not None,
        True,
    ]
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite reliability descriptor")
    return out
