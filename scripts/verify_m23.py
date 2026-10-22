"""Verify preserved historical bytes with the two declared CI routing overlays."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ArchivedRoot:
    def __truediv__(self, path):
        mapping = json.loads((ROOT / "benchmarks/m23/preservation.json").read_text())[
            "archived_paths"
        ]
        return ROOT / mapping.get(str(path), str(path))


def verify(local_refs=False, result=None, m225_result=None):
    packet = ROOT / "benchmarks/m23"
    inventory = json.loads((packet / "preservation.json").read_text())
    for path, expected in inventory["files"].items():
        actual = ROOT / inventory["archived_paths"].get(path, path)
        if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed historical file: {path}")
    if local_refs:
        for ref, expected in inventory["local_refs"].items():
            if (
                subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
                != expected
            ):
                raise ValueError(f"Changed ref: {ref}")
    freeze = packet / "implementation-freeze.json"
    if freeze.exists():
        for path, expected in json.loads(freeze.read_text())["sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen implementation changed: {path}")
    spec = importlib.util.spec_from_file_location("legacy_m225", ROOT / "scripts/verify_m225.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ArchivedRoot()
    module.verify(result_path=m225_result)
    if result:
        record = json.loads(result.read_text())
        if record["engineering"] != "PASS" or len(record["units"]) != 40:
            raise ValueError("Incomplete or failed M23 evidence")
        for path, expected in record["source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError("Result source mismatch")
    print(
        json.dumps(
            {
                "historical_files_preserved": len(inventory["files"]),
                "original_ci_paths_archived": len(inventory["archived_paths"]),
                "local_refs_checked": len(inventory["local_refs"]) if local_refs else None,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-refs", action="store_true")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--m225-result", type=Path)
    args = parser.parse_args()
    verify(args.local_refs, args.result, args.m225_result)
