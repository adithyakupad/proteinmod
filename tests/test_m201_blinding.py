import importlib.util
import io
import json

import pytest

spec = importlib.util.spec_from_file_location("m201", "scripts/audit_m201_zenodo.py")
m201 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m201)


def test_metadata_opaque_exclusions(monkeypatch):
    original = json.loads

    def guarded(text, *args, **kwargs):
        assert "POISON" not in text
        return original(text, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    result = m201.metadata_projection(
        '{"id":1,"metadata":{"license":{"id":"cc-by-4.0"},"description":"POISON"},'
        '"files":[{"key":"data.zip","description":"POISON"}],"stats":{"score":"POISON"}}'
    )
    assert result["id"] == 1
    assert "stats" not in result and "description" not in result["metadata"]


def test_only_genotype_cells_decoded():
    names, rows = m201.project_tsv(io.BytesIO(b"nt_seq\taa_seq\tfitness\tWT\nATG\tM\t\xff\t\xff\n"))
    assert names == ["nt_seq", "aa_seq", "fitness", "WT"]
    assert rows == [{"nt_seq": "ATG", "aa_seq": "M"}]
    with pytest.raises(ValueError, match="width"):
        m201.project_tsv(io.BytesIO(b"nt_seq\taa_seq\tfitness\nATG\tM\n"))
    with pytest.raises(ValueError, match="Unique"):
        m201.project_tsv(io.BytesIO(b"nt_seq\taa_seq\taa_seq\n"))
