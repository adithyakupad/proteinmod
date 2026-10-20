from pathlib import Path

import numpy as np
import pytest

from proteinmod.data import checksum
from proteinmod.edge_chemistry import annotate
from proteinmod.masked_structure import extract, validate_mask
from proteinmod.structural_graph import variant_inputs
from proteinmod.structure_mapping import map_reference, parse_structure

PATH = Path(__file__).parent / "fixtures/structure_tiny.cif"
REF = "ACDEFGHM"


def test_masked_scope_preserves_full_reference_rejection_and_frozen_inputs():
    graph, atoms, _, _ = extract(PATH, REF, [1, 2, 4, 5, 6], checksum(PATH))
    audit = graph["audit"]
    assert audit["scope"] == "masked-reference-v1" and audit["masked_accepted"]
    assert not audit["full_reference_M2_audit"]["accepted"]
    assert not map_reference(parse_structure(PATH), REF, {"chain": "A", "min_coverage": 1.0})[1][
        "accepted"
    ]
    assert len(graph["residues"]) == len(REF)
    assert graph["static"].shape == (5, 34)
    assert annotate(graph, atoms).shape[1] == 29
    a, fa, reasons = variant_inputs(["A1V", "F5Y"], REF, graph)
    b, fb, _ = variant_inputs(["F5Y", "A1V"], REF, graph)
    assert not reasons and a.shape == (5, 25) and fa.shape == (270,)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fa, fb)
    assert variant_inputs(["D3A"], REF, graph)[2]
    with pytest.raises(ValueError, match="checksum"):
        extract(PATH, REF, [1], "wrong")


@pytest.mark.parametrize("mask", [[], [0], [9], [1, 1], [2, 1], [True], [1.0], [3], [7], [8]])
def test_bad_masks_and_incompatible_sites_fail_closed(mask):
    rows, _ = map_reference(parse_structure(PATH), REF, {"chain": "A"})
    with pytest.raises(ValueError):
        validate_mask(rows, REF, mask)


def test_mismatch_must_be_excluded_and_ambiguous_mapping_never_passes():
    graph, _, _, rows = extract(PATH, "CCDEFGHM", [2, 4, 5, 6], checksum(PATH))
    assert graph["audit"]["full_reference_M2_audit"]["differences"] == ["C1A"]
    assert not graph["audit"]["reference_construct_sequence_compatible"]
    with pytest.raises(ValueError, match="canonical identity"):
        validate_mask(rows, "CCDEFGHM", [1, 2])
    with pytest.raises(ValueError, match="complete reference"):
        validate_mask([], REF, [1])
