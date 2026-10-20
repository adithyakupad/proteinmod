"""Explicit closeout amendments; scientific artifacts never receive exceptions."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AMENDABLE = frozenset(
    {
        "README.md",
        "pyproject.toml",
        "scripts/verify_readability.py",
        "scripts/verify_m184.py",
        "scripts/verify_sod1.py",
        "scripts/run_m19.py",
        "tests/test_m19_evidence.py",
    }
)


def verify_digest(path, expected):
    path = Path(path)
    relative = path.resolve().relative_to(ROOT).as_posix()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == expected:
        return
    if relative not in AMENDABLE:
        raise AssertionError(f"Protected historical file changed: {relative}")
    ledger = json.loads((ROOT / "release_candidate/amendments.json").read_text())
    entry = ledger[relative]
    assert actual == entry["after_sha256"], relative
    blob = entry["prior_blobs"][expected]
    original = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=ROOT)
    assert hashlib.sha256(original).hexdigest() == expected, relative
