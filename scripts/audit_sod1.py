"""Outcome-independent 2C9V geometry, canonical genotypes and historical similarity."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from proteinmod.blinded_homology import alignment_diagnostics
from proteinmod.data import checksum, write_json
from proteinmod.edge_chemistry import annotate
from proteinmod.external_gate import similarity_review
from proteinmod.masked_structure import extract
from proteinmod.structural_graph import variant_inputs
from proteinmod.structure_mapping import map_reference, parse_structure


def decode_genotype(hgvs, reference):
    if hgvs == "p.=":
        return None, "synonymous_or_WT"
    match = re.fullmatch(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|\*)", hgvs)
    if not match:
        return None, "unsupported_not_single_substitution"
    old, position, new = match.groups()
    position = int(position)
    old = protein_letters_3to1.get(old)
    if not old or not 1 <= position <= len(reference) or reference[position - 1] != old:
        raise ValueError(f"Assay-frame reference mismatch: {hgvs}")
    if new in {"Ter", "*"}:
        return None, "stop"
    new = protein_letters_3to1.get(new) if new != "=" else old
    if new is None:
        return None, "noncanonical_substitution"
    if new == old:
        return None, "synonymous_or_WT"
    if position == 1:
        return None, "unobserved_initiator_Met"
    return f"{old}{position}{new}", None


def references():
    panel = json.loads(Path("benchmarks/m182/reference-panel.json").read_text())
    return panel["historical"] + panel["external_candidates"]


def audit(output):
    if (output / "structural-genotype-audit.json").exists():
        raise FileExistsError("Choose a new audit output; preserve frozen qualification evidence")
    output.mkdir(parents=True, exist_ok=True)
    source = Path("data/cache/sod1/source/2C9V.cif")
    expected = json.loads(Path("configs/m185.json").read_text())
    if checksum(source) != expected["coordinate_sha256"]:
        raise ValueError("Pinned coordinate source differs")
    reference = expected["reference"]
    metadata = json.loads(Path("benchmarks/m185/evidence/abundance-metadata.json").read_text())
    if metadata["targetGenes"][0]["targetSequence"]["sequence"] != reference:
        raise ValueError("Assay metadata and pinned reference differ")
    parsed = parse_structure(source)
    strict = []
    accepted = []
    eligible = expected["eligible_positions"]
    assert eligible == list(range(2, 155))
    assert (
        expected["primary_score_set"] == metadata["urn"] and not expected["target_read_permission"]
    )
    for chain in sorted(parsed["chains"]):
        _, mapping = map_reference(
            parsed,
            reference,
            {"chain": chain, "min_coverage": 1.0, "min_identity": 1.0, "allowed_mismatches": []},
        )
        strict.append({"chain": chain, "audit": mapping})
        if (
            mapping["missing_positions"] == [1]
            and not mapping["differences"]
            and mapping["exact_compatible_positions"] == 153
        ):
            accepted.append(chain)
    if not accepted:
        raise ValueError("No chain satisfies preregistered initiator-only omission")
    chain = accepted[0]
    graph, atoms, _, rows = extract(
        source, reference, eligible, expected["coordinate_sha256"], chain
    )
    chemistry = annotate(graph, atoms)
    distances = np.linalg.norm(graph["xyz"][:, None] - graph["xyz"][None, :], axis=2)
    physical = {
        tuple(e) for e, a in zip(graph["edge_index"].T, graph["edge_features"], strict=True) if a[1]
    }
    independent = {
        (i, j) for i in range(153) for j in range(153) if i != j and distances[i, j] <= 8
    }
    assert physical == independent
    assert all((j, i) in physical for i, j in physical)
    assert graph["static"].shape == (153, 34) and np.isfinite(graph["static"]).all()
    assert chemistry.shape == (graph["edge_index"].shape[1], 29)
    variants = json.loads(Path("benchmarks/m185/evidence/genotypes.json").read_text())
    assert all(
        r["mavedb.post_mapped_hgvs_p"].split(":", 1)[-1] == r["hgvs_pro"] for r in variants["rows"]
    )
    counts = Counter()
    candidates = []
    exclusions = []
    seen = set()
    for variant in variants["rows"]:
        token, reason = decode_genotype(variant["hgvs_pro"], reference)
        if reason:
            counts[reason] += 1
            exclusions.append(
                {"id": variant["accession"], "hgvs_pro": variant["hgvs_pro"], "reason": reason}
            )
            continue
        position = int(token[1:-1])
        sequence = reference[: position - 1] + token[-1] + reference[position:]
        if sequence in seen:
            raise ValueError("Duplicate canonical mutant sequence")
        seen.add(sequence)
        dynamic, ridge, errors = variant_inputs([token], reference, graph)
        assert not errors and dynamic.shape == (153, 25) and ridge.shape == (270,)
        rank = hashlib.sha256(("m18:external:SOD1" + sequence).encode()).hexdigest()
        candidates.append(
            {
                "id": variant["accession"],
                "protein_id": "SOD1",
                "sequence": sequence,
                "mutations": [token],
                "rank_sha256": rank,
                "candidate_role": "test_only_if_admitted",
            }
        )
    candidates.sort(key=lambda row: row["rank_sha256"])
    selected = candidates[:96]
    # Exercise the existing multiple-mutation interface synthetically, without experimental labels.
    synthetic = [
        f"{reference[1]}2{'V' if reference[1] != 'V' else 'A'}",
        f"{reference[2]}3{'A' if reference[2] != 'A' else 'V'}",
    ]
    forward = variant_inputs(synthetic, reference, graph)
    reverse = variant_inputs(synthetic[::-1], reference, graph)
    assert not forward[2] and not reverse[2]
    assert np.array_equal(forward[0], reverse[0]) and np.array_equal(forward[1], reverse[1])
    cif = MMCIF2Dict(source)
    metals = []
    for i, name in enumerate(cif["_atom_site.label_comp_id"]):
        if name in {"CU", "ZN"}:
            metals.append(
                {
                    key: cif[field][i]
                    for key, field in {
                        "component": "_atom_site.label_comp_id",
                        "chain": "_atom_site.auth_asym_id",
                        "author_number": "_atom_site.auth_seq_id",
                        "altloc": "_atom_site.label_alt_id",
                        "occupancy": "_atom_site.occupancy",
                    }.items()
                }
            )
    sg1 = rows[57]["record"]["residue"]["SG"].coord
    sg2 = rows[146]["record"]["residue"]["SG"].coord
    write_json(
        output / "structural-genotype-audit.json",
        {
            "coordinate_sha256": checksum(source),
            "reference": reference,
            "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            "strict_full_reference": strict,
            "selected_chain": chain,
            "masked": graph["audit"],
            "independent_8a_spatial_edges_verified": True,
            "model_input_dimensions": {"static": 34, "dynamic": 25, "ridge": 270, "edge": 29},
            "finite_chemical_edge_rows": len(chemistry),
            "synthetic_multi_mutation_order_invariance": True,
            "genotypes": len(variants["rows"]),
            "assay_and_RefSeq_HGVS_agreement": len(variants["rows"]),
            "canonical_structurally_eligible": len(candidates),
            "exclusion_counts": dict(counts),
            "exclusions": exclusions,
            "selected_candidates": selected,
            "metals_deposited": metals,
            "disulfide": {
                "assay_positions": [58, 147],
                "author_positions": [57, 146],
                "SG_distance_a": float(np.linalg.norm(sg1 - sg2)),
            },
            "context": "Experimental metallated, disulfide-bonded deposition; deposited other-chain/ligand context ASA. Not mutant rearrangements or proof of identical assay maturation/metallation.",
            "variant_outcome_values_accessed": False,
            "models_fit": 0,
            "predictions_generated": 0,
        },
    )
    settings = json.loads(Path("configs/m12.json").read_text())["similarity"]
    comparisons = []
    for old in references():
        diagnostics = alignment_diagnostics(reference, old["sequence"])
        comparisons.append(
            {
                "a": "SOD1",
                "b": old["id"],
                **diagnostics,
                "original_flag": similarity_review(
                    diagnostics["global"], diagnostics["local"], settings
                ),
            }
        )
    write_json(
        output / "similarity-audit.json",
        {
            "settings": settings,
            "pairs": comparisons,
            "scope": "All18 historical reference-panel proteins plus4 old external candidates; unchanged thresholds, not an exhaustive all-sequence homology proof",
        },
    )
    print(
        "2C9V",
        chain,
        "nodes",
        len(graph["nodes"]),
        "edges",
        graph["audit"]["undirected_edges"],
        "eligible variants",
        len(candidates),
        "selected",
        len(selected),
        "exclusions",
        dict(counts),
    )
    print("Similarity flags", [r["b"] for r in comparisons if r["original_flag"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/m185"))
    audit(parser.parse_args().output)
