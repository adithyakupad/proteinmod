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
    pack,
    save,
    train,
    unpack,
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
                "fixtures": {
                    "x": pack(x),
                    "y": pack(y),
                    "held": pack(held),
                    "target": pack(target),
                    "groups": groups,
                },
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
            "fixture_rtol": 0.0,
            "fixture_atol": 1e-14,
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
        portable_fields = {"environment", "input_sha256", "preprocessing"}
        actual_identity = {k: v for k, v in actual["identity"].items() if k not in portable_fields}
        expected_identity = {
            k: v for k, v in expected["identity"].items() if k not in portable_fields
        }
        if canonical(actual_identity) != canonical(expected_identity):
            different = [
                k for k in actual_identity if actual_identity.get(k) != expected_identity.get(k)
            ]
            raise ValueError(f"Fresh-run protocol/implementation identity differs: {different}")
        different = [
            k
            for k in ("input_sha256", "preprocessing")
            if actual["identity"][k] != expected["identity"][k]
        ]
        if different:
            print(f"Seed {actual['seed']} portable numeric identity differences: {different}")
        for case in (actual, expected):
            fixtures = case["fixtures"]
            hashed = digest({k: fixtures[k] for k in ("x", "y", "groups")})
            if hashed != case["identity"]["input_sha256"]:
                raise ValueError("Fixture/input hash mismatch")
        if actual["fixtures"]["groups"] != expected["fixtures"]["groups"]:
            raise ValueError("Fixture group membership differs")
        maximum_fixture = 0.0
        components = [
            (actual["fixtures"][k], expected["fixtures"][k]) for k in ("x", "y", "held", "target")
        ]
        components += [
            (actual["identity"]["preprocessing"][k], expected["identity"]["preprocessing"][k])
            for k in ("mean", "scale")
        ]
        for a_pack, b_pack in components:
            if a_pack["shape"] != b_pack["shape"]:
                raise ValueError("Portable numeric shape mismatch")
            shape = tuple(a_pack["shape"])
            a, b = unpack(a_pack, shape), unpack(b_pack, shape)
            if not np.allclose(a, b, rtol=0, atol=1e-14):
                raise ValueError("Portable fixture/preprocessing rounding tolerance failed")
            maximum_fixture = max(maximum_fixture, float(np.max(np.abs(a - b))))
        print(f"Seed {actual['seed']} max fixture/preprocessing difference={maximum_fixture:.17g}")
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
