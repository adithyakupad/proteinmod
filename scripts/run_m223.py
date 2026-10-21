"""Run only M22.3 artificial-number replay; never imports historical trainers."""

import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path

import numpy as np

from proteinmod.synthetic_checkpoint import (
    Contract,
    canonical,
    digest,
    environment,
    evaluate,
    load,
    numeric_data,
    save,
    train,
)


def run(output):
    root = Path(__file__).resolve().parents[1]
    output = output.resolve()
    if not output.is_relative_to(root / "runs" / "m223"):
        raise ValueError("Output must be a new directory below runs/m223")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for seed in (0, 1, 2):
        x, y, groups, held, target = numeric_data(seed)
        contract = Contract(x, y, groups, seed)
        full = train(contract, 120)
        prefix = train(contract, 12)
        path = output / f"seed-{seed}-epoch-12.json"
        trusted = save(path, prefix, contract)
        resumed = train(contract, 120, load(path, contract, trusted))
        repeated = train(contract, 120)
        if not canonical(full) == canonical(resumed) == canonical(repeated):
            raise RuntimeError("Same-environment exact replay failed")
        metrics = {}
        for label, state in (("epoch12", prefix), ("full120", full), ("resumed120", resumed)):
            metrics[label] = {
                "train": evaluate(state, contract, x, y),
                "validation": evaluate(state, contract, held[:50], target[:50]),
                "test": evaluate(state, contract, held[50:], target[50:]),
            }
        if metrics["full120"] != metrics["resumed120"]:
            raise RuntimeError("Paired endpoint metrics differ")
        cases.append(
            {
                "seed": seed,
                "identity": contract.identity,
                "checkpoint_file_sha256": trusted,
                "prefix_state_sha256": digest(prefix),
                "full_state_sha256": digest(full),
                "resumed_state_sha256": digest(resumed),
                "repeat_state_sha256": digest(repeated),
                "updates": full["updates"],
                "metrics": metrics,
            }
        )
    result = {
        "milestone": "M22.3",
        "schema": 1,
        "environment": environment(),
        "exact_replay": True,
        "seeds": cases,
        "paired_resume_minus_full": {
            "mse_mean": 0.0,
            "mse_sample_sd": 0.0,
            "spearman_mean": 0.0,
            "spearman_sample_sd": 0.0,
        },
        "cross_platform_policy": {
            "rtol": 1e-8,
            "atol": 1e-9,
            "scope": "fresh-run endpoint metrics only; not checkpoint resume",
        },
    }
    (output / "results.json").write_bytes(canonical(result) + b"\n")
    return result


def compare(result, reference):
    if len(result["seeds"]) != len(reference["seeds"]):
        raise ValueError("Seed count mismatch")
    maximum = 0.0
    for actual, expected in zip(result["seeds"], reference["seeds"], strict=True):
        if actual["seed"] != expected["seed"]:
            raise ValueError("Seed identity mismatch")
        actual_identity = {k: v for k, v in actual["identity"].items() if k != "environment"}
        expected_identity = {k: v for k, v in expected["identity"].items() if k != "environment"}
        if canonical(actual_identity) != canonical(expected_identity):
            raise ValueError("Fresh-run protocol/input/implementation identity differs")
        for horizon in ("epoch12", "full120", "resumed120"):
            for split in ("train", "validation", "test"):
                for metric in ("mse", "spearman"):
                    a = actual["metrics"][horizon][split][metric]
                    b = expected["metrics"][horizon][split][metric]
                    if not np.isclose(a, b, rtol=1e-8, atol=1e-9):
                        raise ValueError(
                            f"Cross-platform metric tolerance failed: {horizon}/{split}/{metric}"
                        )
                    maximum = max(maximum, abs(a - b))
    return maximum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = run(args.output)
    if args.compare:
        difference = compare(result, json.loads(args.compare.read_text()))
        print(
            f"Fresh-run endpoint tolerance passed; max absolute metric difference={difference:.17g}"
        )
    print("Exact replay passed for 3 seeds; 120 epochs = disk checkpoint 12 + continuation 108")
