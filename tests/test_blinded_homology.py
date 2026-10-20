import pytest

from proteinmod.blinded_homology import alignment_diagnostics, genotype_cells


def test_diagnostics_keep_denominators_explicit():
    rows = alignment_diagnostics("ACDEFGHIK", "ACDQQQQEFGHIK")
    g = rows["global"]
    assert g["identical"] == 9
    assert g["identity"] == g["identity_shorter"] == 1
    assert g["identity_longer"] == pytest.approx(9 / 13)
    assert g["unpaired_b"] == 4
    assert g["unpaired_a"] == 0


def test_genotype_whitelist_does_not_request_fitness():
    class Blind(dict):
        def __getitem__(self, key):
            assert key in ("A", "B", "C")
            return super().__getitem__(key)

    assert genotype_cells(Blind(A="26", B="H", C="A")) == {"A": "26", "B": "H", "C": "A"}


def test_workbook_projection_ignores_poisoned_target_cell(tmp_path):
    import zipfile

    from proteinmod.blinded_homology import tem_genotypes

    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    source = tmp_path / "synthetic.xlsx"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr(
            "xl/sharedStrings.xml", f'<sst xmlns="{ns}"><si><t>H</t></si><si><t>A</t></si></sst>'
        )
        z.writestr(
            "xl/worksheets/sheet2.xml",
            f'''<worksheet xmlns="{ns}"><sheetData>
        <row r="3"><c r="A3"><v>26</v></c><c r="B3" t="s"><v>0</v></c>
        <c r="C3" t="s"><v>1</v></c><c r="R3" t="s"><v>999999</v></c>
        <c r="S3"><v>NOT_A_NUMBER</v></c></row></sheetData></worksheet>''',
        )
    assert tem_genotypes(source) == [{"source_row": 3, "position": 26, "wt": "H", "mutant": "A"}]
