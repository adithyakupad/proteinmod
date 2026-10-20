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


def test_notebook_outputs_never_decoded(monkeypatch):
    original = json.loads

    def guarded(text, *args, **kwargs):
        assert "POISON" not in text
        return original(text, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    sources = m201.notebook_code(
        '{"cells":[{"cell_type":"code","source":["x=1"],'
        '"outputs":[{"text":"POISON"}]},{"cell_type":"markdown","source":"POISON"}]}'
    )
    assert sources == ["x=1"]


def test_archived_m201_identity_and_stop_state():
    import hashlib
    from pathlib import Path

    root = Path("benchmarks/m201")
    for path, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    members = json.loads((root / "evidence/member-audit.json").read_text())
    tables = [r for r in members if "genotype_projection" in r]
    assert len(tables) == 3 and len({r["member_sha256"] for r in tables}) == 1
    assert all(r["member_CRC_verified"] and not r["whole_archive_MD5_verified"] for r in tables)
    assert all(
        not r["experimental_values_decoded"] and not r["score_flags_decoded"] for r in tables
    )
    for record in [16266645, 11175470, 15302825]:
        metadata = json.loads((root / f"evidence/{record}-metadata.json").read_text())
        assert metadata["metadata"]["license"]["id"] == "cc-by-4.0"
        assert "stats" not in metadata and "description" not in metadata["metadata"]
    audit = json.loads((root / "genotype-structure-audit.json").read_text())
    assert [r["eligible_unique_nonWT"] for r in audit["families"]] == [8499, 6463]
    assert all(r["prior_selected_genotypes_present"] == 96 for r in audit["families"])
    status = json.loads((root / "qualification.json").read_text())
    assert status["status"] == "NO_GO" and not status["four_family_manifest_issued"]
    assert not status["outcome_values_decoded"] and not status["score_flags_decoded"]
    assert status["models_fit"] == status["predictions"] == status["author_emails"] == 0
    assert not (root / "final_manifest.json").exists()
    owner = json.loads((root / "SOD1-owner-disposition.json").read_text())
    assert owner["decision"] == "ACCEPT_FOR_NEW_PROSPECTIVE_CONSIDERATION"
