import copy
import hashlib
import io
import json
from pathlib import Path

import pytest

from proteinmod.data import checksum
from proteinmod.structure_pipeline import run_structure
from proteinmod.structure_sources import acquire_structure, verify_structure

FIXTURE = Path(__file__).parent / "fixtures" / "structure_tiny.pdb"


@pytest.fixture
def structure_config(tmp_path):
    return {
        "structure": {
            "name": "synthetic fixture",
            "kind": "local",
            "path": str(FIXTURE),
            "sha256": checksum(FIXTURE),
            "source": "handcrafted offline fixture",
            "license": "CC0 synthetic test data",
            "confidence": "experimental_bfactor",
        },
        "reference": {
            "sequence": "ACDEFGHM",
            "source": "synthetic reference",
            "sha256": hashlib.sha256(b"ACDEFGHM").hexdigest(),
        },
        "mapping": {"chain": "A", "min_coverage": 0.5, "min_identity": 0.98},
        "geometry": {"sasa_points": 60, "contact_cutoff_a": 8, "dssp": "off"},
        "structure_run": {"output": str(tmp_path / "run")},
        "variants": [
            {"id": "multi", "mutations": ["G6A", "A1C"]},
            {"id": "missing", "mutations": ["D3A"]},
        ],
    }


def test_offline_pipeline_and_history_guard(structure_config):
    result = run_structure(structure_config, render=False)
    assert result["accepted"] and result["variants"] == 2
    output = Path(result["output"])
    for name in (
        "audit.json",
        "residues.json",
        "residues.csv",
        "contacts.csv",
        "variants.json",
        "variants.csv",
        "report.md",
        "manifest.json",
    ):
        assert (output / name).exists()
    audit = json.loads((output / "audit.json").read_text())
    assert audit["confidence_summary"]["n_low_plddt"] is None
    assert audit["sasa"]["secondary_structure"]["status"] == "disabled"
    assert json.loads((output / "variants.json").read_text())[1]["summary"]["n_unmapped"] == 1
    for historical in ("benchmarks/amylase", "benchmarks/amylase_m1"):
        structure_config["structure_run"]["output"] = historical
        with pytest.raises(ValueError, match="overwrite"):
            run_structure(structure_config, render=False)


def test_failed_mapping_produces_audit(structure_config):
    structure_config["mapping"]["min_coverage"] = 0.95
    with pytest.raises(ValueError, match="rejected"):
        run_structure(structure_config, render=False)
    output = Path(structure_config["structure_run"]["output"])
    assert json.loads((output / "audit.json").read_text())["accepted"] is False
    assert not (output / "variants.json").exists()


def test_verified_local_source_and_semantics(structure_config):
    spec = structure_config["structure"]
    assert acquire_structure(spec) == FIXTURE
    for field in ("sha256", "source", "license", "confidence"):
        modified = copy.deepcopy(spec)
        modified.pop(field)
        with pytest.raises(ValueError):
            verify_structure(modified)
    modified = {**spec, "kind": "pdb", "confidence": "alphafold_plddt"}
    with pytest.raises(ValueError, match="Experimental"):
        verify_structure(modified)


def test_explicit_pdb_download_atomic(tmp_path, monkeypatch):
    path = tmp_path / "1ABC.pdb"
    spec = {
        "kind": "pdb",
        "id": "1ABC",
        "format": "pdb",
        "path": str(path),
        "sha256": hashlib.sha256(b"coordinates").hexdigest(),
    }
    monkeypatch.setattr(
        "proteinmod.structure_sources.urlopen", lambda *args, **kwargs: io.BytesIO(b"coordinates")
    )
    assert acquire_structure(spec) == path
    assert path.with_suffix(".pdb.source.json").exists()
    monkeypatch.setattr(
        "proteinmod.structure_sources.urlopen", lambda *args, **kwargs: io.BytesIO(b"bad")
    )
    with pytest.raises(ValueError, match="checksum"):
        acquire_structure(spec)
    assert path.read_bytes() == b"coordinates"


def test_alphafold_uses_exact_provider_entry(tmp_path, monkeypatch):
    responses = [
        json.dumps(
            [{"entryId": "AF-P12345-F1", "cifUrl": "https://alphafold.ebi.ac.uk/files/exact.cif"}]
        ).encode(),
        b"coordinates",
    ]
    urls = []

    def mock_open(url, **kwargs):
        urls.append(url)
        return io.BytesIO(responses.pop(0))

    monkeypatch.setattr("proteinmod.structure_sources.urlopen", mock_open)
    spec = {
        "kind": "alphafold",
        "accession": "P12345",
        "entry_id": "AF-P12345-F1",
        "format": "cif",
        "path": str(tmp_path / "model.cif"),
        "sha256": hashlib.sha256(b"coordinates").hexdigest(),
    }
    acquire_structure(spec)
    assert urls == [
        "https://alphafold.ebi.ac.uk/api/prediction/P12345",
        "https://alphafold.ebi.ac.uk/files/exact.cif",
    ]
    monkeypatch.setattr(
        "proteinmod.structure_sources.urlopen", lambda *args, **kwargs: io.BytesIO(b"[]")
    )
    with pytest.raises(ValueError, match="ambiguous/absent"):
        acquire_structure(spec)


def test_invalid_confidence_fails_with_audit(structure_config, tmp_path):
    path = tmp_path / "invalid_confidence.pdb"
    lines = []
    for line in FIXTURE.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = line[:60] + "120.00" + line[66:]
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    structure_config["structure"].update(
        path=str(path), sha256=checksum(path), confidence="alphafold_plddt"
    )
    with pytest.raises(ValueError, match="pLDDT"):
        run_structure(structure_config, render=False)
    audit = json.loads(
        (Path(structure_config["structure_run"]["output"]) / "audit.json").read_text()
    )
    assert audit["pipeline_status"] == "failed"
