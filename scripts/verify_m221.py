"""Verify synthetic evidence and opaque historical hashes, without research imports."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proteinmod.synthetic_regression import summarize

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "benchmarks/m221"


def read(path):
    return json.loads(path.read_text())


def compare_metrics(original, repeated):
    """Declared cross-platform metric tolerance, separate from exact local replay."""
    left = {(r["scenario"], r["seed"]): r for r in original["records"]}
    right = {(r["scenario"], r["seed"]): r for r in repeated["records"]}
    assert left.keys() == right.keys()
    maximum = 0.0
    for key in left:
        for epoch in ("12", "120"):
            for split in ("train", "validation", "test"):
                for metric in ("mse", "spearman"):
                    first = left[key]["checkpoints"][epoch][split][metric]
                    second = right[key]["checkpoints"][epoch][split][metric]
                    if first is None or second is None:
                        assert first is second
                    else:
                        np.testing.assert_allclose(first, second, rtol=1e-8, atol=1e-9)
                        maximum = max(maximum, abs(first - second))
    return maximum


def validate_result(result, config):
    assert result["biological_inputs"] is False and result["failures"] == []
    records = result["records"]
    expected = {
        (scenario["id"], seed) for scenario in config["scenarios"] for seed in config["seeds"]
    }
    assert len(records) == len(expected)
    assert {(r["scenario"], r["seed"]) for r in records} == expected
    for row in records:
        assert all(
            row[key]
            for key in ("full_replay_exact", "prefix_exact", "resume_exact", "disk_roundtrip_exact")
        )
        assert set(row["checkpoints"]) == {"12", "120"}
        assert len(row["train_mse_trace"]) == 120
        for epoch in (12, 120):
            assert (
                row["train_mse_trace"][epoch - 1] == row["checkpoints"][str(epoch)]["train"]["mse"]
            )
        for split in ("train", "validation", "test"):
            for metric in ("mse", "spearman"):
                first = row["checkpoints"]["12"][split][metric]
                last = row["checkpoints"]["120"][split][metric]
                expected_effect = None if first is None or last is None else last - first
                assert row["paired_effects"][split][metric] == expected_effect
    assert result["summary"] == summarize(records)


def verify(compare=None):
    config_file = ROOT / "configs/m221_synthetic.json"
    config, prereg = read(config_file), read(PACKET / "preregistration.json")
    assert hashlib.sha256(config_file.read_bytes()).hexdigest() == prereg["config_sha256"]
    assert (
        hashlib.sha256((PACKET / "PREREGISTRATION.md").read_bytes()).hexdigest()
        == prereg["plan_sha256"]
    )
    for relative, expected in read(PACKET / "implementation-freeze.json")["files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    preservation = read(PACKET / "preservation.json")
    for relative, expected in preservation["historical_main_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    result = read(PACKET / "results.json")
    assert result["config_sha256"] == prereg["config_sha256"]
    for relative, expected in result["implementation_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    validate_result(result, config)
    maximum = None
    if compare:
        fresh = read(compare)
        validate_result(fresh, config)
        assert fresh["config_sha256"] == result["config_sha256"]
        assert fresh["implementation_sha256"] == result["implementation_sha256"]
        maximum = compare_metrics(result, fresh)
    return {
        "synthetic_pairs": len(result["records"]),
        "historical_main_files_unchanged": len(preservation["historical_main_hashes"]),
        "paired_summaries_recomputed": True,
        "same_environment_exact_replays": True,
        "cross_platform_comparison_max_metric_difference": maximum,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.compare), indent=2))
