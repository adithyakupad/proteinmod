"""Opaque hash verification only; does not parse any historical research data."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def verify(local_refs=False):
    manifest = json.loads((ROOT / "benchmarks/m223/preservation.json").read_text())
    for path, expected in manifest["main_files_sha256"].items():
        current = ROOT / path
        if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Historical tracked file changed: {path}")
    if local_refs and git("rev-parse", "main").decode().strip() != manifest["main_commit"]:
        raise RuntimeError("Local main changed")
    freeze_path = ROOT / "benchmarks/m223/implementation-freeze.json"
    if freeze_path.is_file():
        for path, expected in json.loads(freeze_path.read_text())["sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen implementation changed: {path}")
    if local_refs:
        for ref, commit in manifest["preserved_local_refs"].items():
            if git("rev-parse", ref).decode().strip() != commit:
                raise RuntimeError(f"Preserved ref changed: {ref}")
    print(f"Verified {len(manifest['main_files_sha256'])} historical main files unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-refs", action="store_true")
    arguments = parser.parse_args()
    verify(arguments.local_refs)
