from pathlib import Path

import pytest

from proteinmod.structure_mapping import map_reference, parse_structure, unique_alignment

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = "ACDEFGHM"


@pytest.mark.parametrize("suffix", ["pdb", "cif"])
def test_author_insertion_missing_and_noncanonical(suffix):
    parsed = parse_structure(FIXTURES / f"structure_tiny.{suffix}")
    rows, audit = map_reference(parsed, REFERENCE, {"chain": "A", "min_coverage": 0.5})
    assert audit["accepted"]
    assert audit["coordinate_coverage"] == 0.75
    assert audit["missing_positions"] == [3, 7]
    assert (rows[1]["author_number"], rows[1]["insertion_code"]) == (10, "A")
    assert rows[2]["status"] == "missing_coordinates"
    assert rows[6]["status"] == "missing_ca"
    assert rows[7]["structure_aa"] == "M" and rows[7]["noncanonical"]
    assert parsed["altlocs"][0]["selected"] == "B"
    assert rows[1]["record"]["residue"]["CA"].coord[0] == 4
    if suffix == "cif":
        assert rows[2]["author_number"] == 11


def test_ambiguous_alignment_rejected():
    with pytest.raises(ValueError, match="Ambiguous"):
        unique_alignment("AAAA", "AAA")


def test_tags_and_construct_deletion():
    pairs, _, _ = unique_alignment("ACDE", "GGACDEHH")
    assert pairs == {0: 2, 1: 3, 2: 4, 3: 5}
    pairs, _, _ = unique_alignment("ACDEFG", "ACEFG")
    assert 2 not in pairs
    assert pairs[3] == 2


def test_alternate_chains_require_selection(tmp_path):
    text = (FIXTURES / "structure_tiny.pdb").read_text()
    first = text.replace("END\n", "")
    second = []
    for line in text.splitlines():
        if line.startswith("SEQRES"):
            second.append(line[:11] + "B" + line[12:])
        elif line.startswith(("ATOM  ", "HETATM")):
            second.append(line[:21] + "B" + line[22:])
    path = tmp_path / "chains.pdb"
    path.write_text(first + "\n".join(second) + "\nEND\n")
    parsed = parse_structure(path)
    _, audit = map_reference(parsed, REFERENCE, {"min_coverage": 0.5})
    assert not audit["accepted"]
    assert "Ambiguous" in audit["errors"][0]
    rows, audit = map_reference(parsed, REFERENCE, {"chain": "B", "min_coverage": 0.5})
    assert audit["accepted"] and rows[0]["chain"] == "B"


def test_engineered_mismatch_must_be_approved():
    parsed = parse_structure(FIXTURES / "structure_tiny.pdb")
    settings = {"chain": "A", "min_coverage": 0.5, "min_identity": 0.8}
    rows, audit = map_reference(parsed, "CCDEFGHM", settings)
    assert not audit["accepted"] and audit["differences"] == ["C1A"]
    assert rows[0]["status"] == "sequence_mismatch"
    settings["allowed_mismatches"] = ["C1A"]
    assert map_reference(parsed, "CCDEFGHM", settings)[1]["accepted"]


def test_low_coverage_and_unknown_chain():
    parsed = parse_structure(FIXTURES / "structure_tiny.pdb")
    assert not map_reference(parsed, REFERENCE, {"chain": "A", "min_coverage": 0.9})[1]["accepted"]
    assert not map_reference(parsed, REFERENCE, {"chain": "Z"})[1]["accepted"]


def test_model_selection_and_negative_numbering(tmp_path):
    atoms = "ATOM      1  CA  ALA A  -2       0.000   0.000   0.000  1.00 20.00           C  \n"
    path = tmp_path / "ensemble.pdb"
    path.write_text(
        "MODEL        1\n"
        + atoms
        + "ENDMDL\nMODEL        2\n"
        + atoms.replace("0.000", "1.000", 1)
        + "ENDMDL\nEND\n"
    )
    parsed = parse_structure(path, 1)
    rows, audit = map_reference(parsed, "A", {"chain": "A"})
    assert audit["accepted"] and parsed["model_count"] == 2
    assert rows[0]["author_number"] == -2
    assert rows[0]["record"]["residue"]["CA"].coord[0] == 1
    with pytest.raises(ValueError, match="model_index"):
        parse_structure(path, 2)


def test_free_amino_acid_ligand_not_mapped_as_polymer(tmp_path):
    text = (FIXTURES / "structure_tiny.pdb").read_text().replace("END\n", "")
    text += (
        "HETATM   90  CA  GLY A  99      30.000   0.000   0.000  1.00 20.00           C  \nEND\n"
    )
    path = tmp_path / "ligand.pdb"
    path.write_text(text)
    parsed = parse_structure(path)
    rows, audit = map_reference(parsed, REFERENCE, {"chain": "A", "min_coverage": 0.5})
    assert audit["accepted"] and len(rows) == 8
    assert parsed["model"]["A"][("H_GLY", 99, " ")]  # retained as a SASA occluder


def test_ambiguous_residue_microheterogeneity_rejected(tmp_path):
    path = tmp_path / "microheterogeneity.pdb"
    lines = []
    for serial, name, alt in [(1, "SER", "A"), (2, "CYS", "B")]:
        lines.append(
            f"ATOM  {serial:5d}  CA {alt}{name} A   1       0.000   0.000   0.000  0.50 20.00           C  "
        )
    path.write_text("\n".join(lines) + "\nEND\n")
    with pytest.raises(ValueError, match="microheterogeneity"):
        parse_structure(path)
