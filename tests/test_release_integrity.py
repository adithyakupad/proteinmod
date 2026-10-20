import hashlib
import json

import pytest

from proteinmod import release_integrity


def test_only_exact_reviewed_amendments_are_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(release_integrity, "ROOT", tmp_path)
    old = hashlib.sha256(b"old").hexdigest()
    new = hashlib.sha256(b"new").hexdigest()
    path = tmp_path / "README.md"
    path.write_bytes(b"new")
    ledger = tmp_path / "release_candidate"
    ledger.mkdir()
    (ledger / "amendments.json").write_text(
        json.dumps({"README.md": {"after_sha256": new, "prior_blobs": {old: "blob"}}})
    )
    monkeypatch.setattr(release_integrity.subprocess, "check_output", lambda *a, **k: b"old")
    release_integrity.verify_digest(path, old)
    path.write_bytes(b"unreviewed")
    with pytest.raises(AssertionError):
        release_integrity.verify_digest(path, old)
    artifact = tmp_path / "benchmarks/m19/predictions.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"new")
    with pytest.raises(AssertionError, match="Protected"):
        release_integrity.verify_digest(artifact, old)
