"""Verify integration bytes and numeric-only evidence; no historical trainer imports."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def check_hash(path, expected):
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Preservation hash mismatch: {path}")


def verify_bytes(root=ROOT, local_refs=False):
    manifest = read(root / "benchmarks/m224/preservation.json")
    for path, expected in manifest["main_files"].items():
        check_hash(root / path, expected)
    for source in manifest["sources"].values():
        for entry in source["files"].values():
            check_hash(root / entry["preserved_at"], entry["sha256"])
    freeze = root / "benchmarks/m224/implementation-freeze.json"
    if freeze.is_file():
        for path, expected in read(freeze)["sha256"].items():
            check_hash(root / path, expected)
    if local_refs:
        for ref, expected in manifest["local_refs"].items():
            current = subprocess.check_output(
                ["git", "rev-parse", ref], cwd=root, text=True
            ).strip()
            if current != expected:
                raise ValueError(f"Preserved ref changed: {ref}")
    return {
        "historical_main_files": len(manifest["main_files"]),
        "source_snapshot_files": {n: len(s["files"]) for n, s in manifest["sources"].items()},
        "local_refs_checked": len(manifest["local_refs"]) if local_refs else None,
    }


def verify_numeric(compare=None):
    if not __debug__:
        raise RuntimeError("M22.1 legacy evidence validation requires ordinary Python assertions")
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("m221_evidence", ROOT / "scripts/verify_m221.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = read(ROOT / "benchmarks/m221/results.json")
    config = read(ROOT / "configs/m221_synthetic.json")
    module.validate_result(original, config)
    maximum = None
    if compare:
        fresh = read(compare)
        module.validate_result(fresh, config)
        if (
            fresh["config_sha256"] != original["config_sha256"]
            or fresh["implementation_sha256"] != original["implementation_sha256"]
        ):
            raise ValueError("M22.1 fresh-run config/source identity mismatch")
        maximum = module.compare_metrics(original, fresh)
    m223 = read(ROOT / "benchmarks/m223/results.json")
    for case in m223["seeds"]:
        if (
            not case["full_state_sha256"]
            == case["resumed_state_sha256"]
            == case["repeat_state_sha256"]
        ):
            raise ValueError("Recorded M22.3 exact replay mismatch")
    return {
        "m221_pairs": len(original["records"]),
        "m221_max_metric_difference": maximum,
        "m223_exact_replay_seeds": len(m223["seeds"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m221-compare", type=Path)
    parser.add_argument("--local-refs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {**verify_bytes(local_refs=args.local_refs), **verify_numeric(args.m221_compare)}
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))
