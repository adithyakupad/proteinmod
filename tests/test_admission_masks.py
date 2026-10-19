import pytest

from proteinmod.admission_masks import compatible_positions, gfp_mutations


def test_zero_based_author_codec_never_parses_outcomes():
    assert gfp_mutations("SA0V:SD2G\tPOISON_LABEL\tNOT_A_NUMBER", "ACDE") == (("A1V", "D3G"), None)
    assert gfp_mutations("\tPOISON", "ACDE") == ((), None)
    assert gfp_mutations("SA0*\tPOISON", "ACDE") == (None, "noncanonical_substitution")
    with pytest.raises(ValueError, match="WT/reference"):
        gfp_mutations("SA1V\tPOISON", "ACDE")
    with pytest.raises(ValueError, match="WT/reference"):
        gfp_mutations("SA0*:SA1V\tPOISON", "ACDE")


def test_mask_excludes_missing_mismatched_and_noncanonical_sites():
    row = {
        "status": "mapped",
        "has_ca": True,
        "noncanonical": False,
        "reference_aa": "A",
        "structure_aa": "A",
    }
    residues = [
        {"reference_position": 1, **row},
        {"reference_position": 2, **row, "structure_aa": "V"},
        {"reference_position": 3, **row, "has_ca": False},
        {"reference_position": 4, **row, "noncanonical": True},
        {"reference_position": 5, **row, "status": "missing_coordinates"},
    ]
    assert compatible_positions(residues) == [1]
