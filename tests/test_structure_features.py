import math
from pathlib import Path

import numpy as np
import pytest

from proteinmod.structure_features import (
    confidence_features,
    contact_graph,
    parse_mutation_set,
    residue_features,
    secondary_structure,
    variant_context,
)
from proteinmod.structure_mapping import map_reference, parse_structure

FIXTURE = Path(__file__).parent / "fixtures" / "structure_tiny.pdb"


@pytest.fixture
def extracted():
    parsed = parse_structure(FIXTURE)
    rows, _ = map_reference(parsed, "ACDEFGHM", {"chain": "A", "min_coverage": 0.5})
    features, info = residue_features(
        parsed, rows, {"sasa_points": 60, "dssp": "off"}, FIXTURE, "experimental_bfactor"
    )
    edges, breaks = contact_graph(rows, features, 8)
    return rows, features, edges, breaks, info


def test_spatial_vs_sequence_and_chain_breaks(extracted):
    _, _, edges, breaks, _ = extracted
    edge = next(e for e in edges if (e["source"], e["target"]) == (1, 6))
    assert edge["distance_a"] == pytest.approx(math.sqrt(9.25))
    assert edge["nonlocal"] and edge["spatial"] and not edge["sequence_neighbor"]
    sequential = next(e for e in edges if (e["source"], e["target"]) == (4, 5))
    assert sequential["sequence_neighbor"] and not sequential["backbone"]  # explicit TER
    assert not any(e["backbone"] and (e["source"], e["target"]) == (2, 4) for e in edges)
    assert any(e["backbone"] and (e["source"], e["target"]) == (1, 2) for e in edges)
    assert any(b["left"] == 4 and b["right"] == 5 for b in breaks)


def test_multi_mutation_symmetry_distances_and_missing(extracted):
    _, features, edges, _, _ = extracted
    first = variant_context("x", ["G6A", "A1C"], "ACDEFGHM", features, edges)
    second = variant_context("x", ["A1C", "G6A"], "ACDEFGHM", features, edges)
    assert first == second
    np.testing.assert_allclose(
        first["mutation_distance_matrix_a"], [[0, math.sqrt(9.25)], [math.sqrt(9.25), 0]]
    )
    assert first["summary"]["n_mapped"] == 2
    missing = variant_context("missing", ["D3A", "H7A", "A1C"], "ACDEFGHM", features, edges)
    assert missing["summary"]["n_unmapped"] == 2
    assert missing["mutation_distance_matrix_a"][0][1] is None
    assert missing["sites"][1]["mapping_status"] == "missing_coordinates"
    assert (
        variant_context("WT", [], "ACDEFGHM", features, edges)["mutation_distance_matrix_a"] == []
    )


@pytest.mark.parametrize("tokens", [["A1A"], ["C1A"], ["A1C", "A1D"], ["A99C"], ["A0C"], ["A1X"]])
def test_invalid_mutation_sets(tokens):
    with pytest.raises(ValueError):
        parse_mutation_set(tokens, "ACDE")


def test_confidence_semantics_never_equated():
    assert confidence_features(52, "experimental_bfactor") == {
        "bfactor_a2": 52,
        "plddt": None,
        "low_confidence": None,
    }
    assert confidence_features(52, "alphafold_plddt") == {
        "bfactor_a2": None,
        "plddt": 52,
        "low_confidence": True,
    }
    assert confidence_features(120, "experimental_bfactor")["bfactor_a2"] == 120
    assert confidence_features(None, "alphafold_plddt")["plddt"] is None
    assert confidence_features(90, "unknown")["plddt"] is None
    with pytest.raises(ValueError, match="pLDDT"):
        confidence_features(101, "alphafold_plddt")


def test_isolated_atom_sasa_and_reference_table(tmp_path):
    path = tmp_path / "single.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  \nEND\n"
    )
    parsed = parse_structure(path)
    rows, _ = map_reference(parsed, "A", {"chain": "A"})
    features, info = residue_features(
        parsed, rows, {"sasa_points": 100, "dssp": "off"}, path, "experimental_bfactor"
    )
    expected = 4 * math.pi * (1.7 + 1.4) ** 2
    assert features[0]["asa_a2"] == pytest.approx(expected)
    assert features[0]["rsa"] == pytest.approx(expected / 129)
    assert info["rsa_clipped"] is False


def test_dssp_unavailable_is_explicit(monkeypatch):
    monkeypatch.setattr("proteinmod.structure_features.shutil.which", lambda name: None)
    assert secondary_structure({}, FIXTURE, "auto")[1]["status"] == "unavailable"
    with pytest.raises(ValueError, match="not installed"):
        secondary_structure({}, FIXTURE, "required")


def test_missing_site_features_remain_null(extracted):
    _, features, _, _, _ = extracted
    assert features[2]["rsa"] is None
    assert features[6]["ca"] is None
    assert features[6]["contact_degree"] is None
    assert all(row["plddt"] is None for row in features)


def test_missing_pdb_bfactor_is_not_zero_plddt(tmp_path):
    path = tmp_path / "missing_b.pdb"
    line = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C  "
    path.write_text(line[:60] + "      " + line[66:] + "\nEND\n")
    parsed = parse_structure(path)
    rows, _ = map_reference(parsed, "A", {"chain": "A"})
    features, _ = residue_features(
        parsed, rows, {"sasa_points": 60, "dssp": "off"}, path, "alphafold_plddt"
    )
    assert features[0]["plddt"] is None
    assert features[0]["low_confidence"] is None


def test_documented_contact_density(extracted):
    _, features, _, _, _ = extracted
    assert features[0]["contact_degree"] == 3
    assert features[0]["contact_density_per_a3"] == pytest.approx(3 / (4 * math.pi * 8**3 / 3))
