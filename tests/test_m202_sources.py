import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
spec = importlib.util.spec_from_file_location("m202", "scripts/audit_m202_sources.py")
m202 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m202)


def test_source_only_projection(monkeypatch):
    original = json.loads

    def guarded(text, *args, **kwargs):
        assert "POISON" not in text
        return original(text, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    body = (
        b'{"cells":[{"cell_type":"code","source":["x=1"],'
        b'"outputs":[{"text":"POISON"}]},'
        b'{"cell_type":"markdown","source":"POISON"}]}'
    )
    assert m202.project_source("pinned.ipynb", body) == "x=1"
    with pytest.raises(ValueError, match="registered"):
        m202.project_source("fitness.txt", b"POISON")


def test_m202_pinned_sources_and_history():
    root = Path("benchmarks/m202")
    for filename, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
    plan = json.loads((root / "source-plan.json").read_text())
    pinned = {m["name"]: m for g in plan for m in g["members"]}
    records = json.loads((root / "evidence/source-audit.json").read_text())
    assert len(records) == len(pinned) == 32
    for record in records:
        assert record["member"] == pinned[record["member"]["name"]]
        assert record["CRC_verified"] and not record["notebook_outputs_decoded"]
        assert not record["source_executed"]
        assert (
            hashlib.sha256(Path(record["source_path"]).read_bytes()).hexdigest()
            == record["source_sha256"]
        )
    github = json.loads((root / "evidence/author-source-identity.json").read_text())
    assert (
        github["projection_sha256"]
        == hashlib.sha256((root / "evidence/Fig2.source.txt").read_bytes()).hexdigest()
    )


def test_m202_admission_stops_without_outcomes():
    root = Path("benchmarks/m202")
    gate = json.loads((root / "qualification.json").read_text())
    assert gate["decision"] == "FAIL_ADMISSION_STOP"
    assert gate["sigma"].startswith("OMITTED entirely")
    for name in [
        "manifest_issued",
        "execution_command_issued",
        "outcomes_accessed",
        "distributions_accessed",
        "source_selection_flag_values_accessed",
    ]:
        assert gate[name] is False
    assert gate["models_fit"] == gate["predictions"] == gate["emails"] == gate["tags"] == 0
    assert not (root / "final_manifest.json").exists()
    assert not (root / "evidence/primary-methods.txt").exists()
