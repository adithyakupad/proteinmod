"""Opaque preservation, registration and synthetic bridge evidence verification."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(local_refs=False, result_path=None):
    packet = ROOT / "benchmarks/m225"
    inventory = json.loads((packet / "preservation.json").read_text())
    for path, expected in inventory["files"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Parent artifact changed: {path}")
    registered = json.loads((packet / "preregistration.json").read_text())
    if (
        hashlib.sha256((packet / "PREREGISTRATION.md").read_bytes()).hexdigest()
        != registered["plan_sha256"]
    ):
        raise ValueError("Preregistered plan changed")
    freeze_path = packet / "implementation-freeze.json"
    if freeze_path.exists():
        for path, expected in json.loads(freeze_path.read_text())["sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bridge implementation changed: {path}")
    if local_refs:
        for ref, expected in inventory["local_refs"].items():
            if (
                subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
                != expected
            ):
                raise ValueError(f"Historical ref changed: {ref}")
    if result_path:
        result = json.loads(result_path.read_text())
        expected_cases = {
            (s, seed) for s in registered["scenarios"] for seed in registered["seeds"]
        }
        actual = {(r["scenario"], r["seed"]) for r in result["records"]}
        if (
            result["decision"] != "PASS"
            or len(result["records"]) != 20
            or actual != expected_cases
            or any(
                not r["pass"]
                or r["uninterrupted_state_sha256"] != r["resumed_state_sha256"]
                or any(effect != 0 for effect in r["paired_effects"])
                for r in result["records"]
            )
        ):
            raise ValueError("Bridge evidence failed exact all-case gate")
        for path, expected in result["source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError("Result implementation identity mismatch")
    print(
        json.dumps(
            {
                "parent_files_unchanged": len(inventory["files"]),
                "local_refs_checked": len(inventory["local_refs"]) if local_refs else None,
                "bridge_result_checked": str(result_path) if result_path else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-refs", action="store_true")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    verify(args.local_refs, args.result)
