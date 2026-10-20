import importlib.util
import io

import pytest

spec = importlib.util.spec_from_file_location("sod1_supplement", "scripts/sod1_supplement.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_header_parser_stops_before_any_body_xml():
    header = b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row>'
    handle = io.BytesIO(header + b"UNPARSEABLE_OUTCOME_BODY")
    assert module.first_row(handle) == [
        {"coordinate": "A1", "type": "s", "value": "0", "inline": None}
    ]
    assert handle.tell() == len(header)


def test_only_selected_shared_strings_are_decoded():
    raw = b"<sst><si><t>hgvs_pro</t></si><si><BROKEN_SECRET></si><si><t>score</t></si></sst>"
    assert module.selected_strings(raw, {0, 2}) == {0: "hgvs_pro", 2: "score"}
    with pytest.raises(ValueError):
        module.selected_strings(raw, {4})


def test_inherited_excel_namespace_and_numeric_header_rejection():
    header = b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:x14ac="urn:excel"><sheetData><row r="1" x14ac:dyDescent="0.25"><c r="A1" t="s"><v>0</v></c></row>'
    assert module.first_row(io.BytesIO(header))[0]["value"] == "0"
    numeric = b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><row r="1"><c r="B1"><v>0.5</v></c></row>'
    with pytest.raises(ValueError, match="Unverified numeric"):
        module.first_row(io.BytesIO(numeric))
