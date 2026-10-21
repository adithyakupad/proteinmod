"""Composition and preservation controls; numeric infrastructure only."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location("m224_script", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_draft_snapshots_and_main_are_preserved():
    module = load_script("scripts/verify_m224.py")
    result = module.verify_bytes()
    assert result["historical_main_files"] == 953
    assert result["source_snapshot_files"] == {"62": 16, "63": 12}


def test_tampered_routing_archive_rejected(tmp_path):
    module = load_script("scripts/verify_m224.py")
    (tmp_path / "benchmarks/m224").mkdir(parents=True)
    (tmp_path / "bad.txt").write_text("tampered")
    manifest = {
        "main_files": {},
        "sources": {
            "62": {"files": {"conftest.py": {"preserved_at": "bad.txt", "sha256": "0" * 64}}}
        },
    }
    (tmp_path / "benchmarks/m224/preservation.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_bytes(tmp_path)


def test_combined_collection_guard_is_additive_and_finite():
    module = load_script("conftest.py")
    module.validate_synthetic_changes(
        [
            "A\ttests/test_m221_synthetic.py",
            "A\ttests/test_m223_checkpoint.py",
            "A\ttests/test_m224_integration.py",
        ]
    )
    assert module.TESTS == {
        "test_m221_synthetic.py",
        "test_m223_checkpoint.py",
        "test_m224_integration.py",
    }
    for changes in ([], ["M\tsrc/proteinmod/edge.py"], ["A\tunrelated.py"]):
        with pytest.raises(ValueError):
            module.validate_synthetic_changes(changes)
