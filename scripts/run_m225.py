"""Execute only the preregistered artificial M22.1 checkpoint substitution bridge."""

import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from proteinmod import synthetic_bridge as bridge
from proteinmod import synthetic_regression as original
from proteinmod.synthetic_checkpoint import CheckpointError, canonical, digest, environment

ROOT = Path(__file__).resolve().parents[1]


def reject(action):
    try:
        action()
    except CheckpointError:
        return
    raise RuntimeError("Required rejection control was accepted")


def case_run(output, config, scenario, seed):
    data = original.generate(config, scenario, seed)
    contract = bridge.Contract(*data["train"], scenario["id"], seed)
    full = original.fit(config, *data["train"], seed)
    independent = original.fit(config, *data["train"], seed)
    prefix = original.fit(config, *data["train"], seed, stop=12)[12]
    bridge.require_equal(prefix, full[12])
    for epoch in (12, 120):
        bridge.require_equal(full[epoch], independent[epoch])
    path = output / f"{scenario['id']}-{seed}-12.json"
    trusted = bridge.save(path, prefix, contract)
    restored = bridge.load(path, contract, trusted)
    bridge.require_equal(prefix, restored)
    resumed = bridge.resume(path, contract, trusted)
    bridge.require_equal(full[120], resumed)
    terminal = output / f"{scenario['id']}-{seed}-120.json"
    terminal_sha = bridge.save(terminal, resumed, contract)
    bridge.require_equal(resumed, bridge.load(terminal, contract, terminal_sha))
    metrics = {}
    effects = []
    for epoch, state in ((12, restored), (120, resumed)):
        metrics[str(epoch)] = {}
        for split, arrays in data.items():
            before = original.metrics(full[epoch], *arrays)
            after = original.metrics(state, *arrays)
            if before != after:
                raise RuntimeError("Exact endpoint metric comparison failed")
            metrics[str(epoch)][split] = {"uninterrupted": before, "bridge": after}
            effects.extend(after[k] - before[k] for k in ("mse", "spearman"))
    bad = output / f"{scenario['id']}-{seed}-corrupt.json"
    bad.write_bytes(path.read_bytes()[:-1])
    reject(lambda: bridge.resume(bad, contract, trusted))
    payload = bridge.encode(prefix, contract)
    payload["state"]["step"] = 0
    corrupted = canonical({"payload": payload, "payload_sha256": digest(payload)})
    bad.write_bytes(corrupted)
    reject(lambda: bridge.resume(bad, contract, hashlib.sha256(corrupted).hexdigest()))
    changed = bridge.Contract(contract.x, contract.y + 1, scenario["id"], seed)
    reject(lambda: bridge.resume(path, changed, trusted))
    # Deliberate evaluation-only intervention; contract and saved training inputs are separate.
    for split in ("validation", "test"):
        x, y = data[split]
        original.metrics(restored, x * 1e5 + 3, -y * 7)
    bridge.require_equal(resumed, bridge.resume(path, contract, trusted))
    bridge.require_equal(prefix, restored)
    if any(value != 0 for value in effects):
        raise RuntimeError("Nonzero paired bridge effect")
    return {
        "scenario": scenario["id"],
        "seed": seed,
        "pass": True,
        "prefix_state_sha256": original.snapshot_digest(prefix),
        "uninterrupted_state_sha256": original.snapshot_digest(full[120]),
        "resumed_state_sha256": original.snapshot_digest(resumed),
        "checkpoint_sha256": {"12": trusted, "120": terminal_sha},
        "training_sha256": contract.identity["training_sha256"],
        "schedule_sha256": contract.identity["schedule_sha256"],
        "updates": {"12": prefix["step"], "120": resumed["step"]},
        "exact_components": [
            "parameters",
            "m",
            "v",
            "rng",
            "counters",
            "preprocessing",
            "trace",
            "metadata",
        ],
        "controls": [
            "corruption",
            "optimizer_counter_reset",
            "changed_training",
            "heldout_isolation",
        ],
        "metrics": metrics,
        "paired_effects": effects,
    }


def run(output, compare=None):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "runs/m225"):
        raise ValueError("Output must be new under runs/m225")
    output.mkdir(parents=True, exist_ok=False)
    config = bridge.registered_config()
    prereg = json.loads((ROOT / "benchmarks/m225/preregistration.json").read_text())
    if (
        hashlib.sha256((ROOT / "benchmarks/m225/PREREGISTRATION.md").read_bytes()).hexdigest()
        != prereg["plan_sha256"]
    ):
        raise ValueError("Registered plan changed")
    records = []
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            try:
                record = case_run(output, config, scenario, seed)
            except (ValueError, RuntimeError, OSError, TypeError, KeyError, OverflowError) as error:
                record = {
                    "scenario": scenario["id"],
                    "seed": seed,
                    "pass": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            records.append(record)
            print(
                f"{scenario['id']} seed {seed}: {'PASS' if record['pass'] else 'FAIL'}", flush=True
            )
    maximum = None
    portable_failure = None
    if compare:
        maximum = 0.0
        reference = {
            (r["scenario"], r["seed"]): r for r in json.loads(compare.read_text())["records"]
        }
        for row in records:
            if not row["pass"]:
                continue
            for epoch in ("12", "120"):
                for split in ("train", "validation", "test"):
                    for metric in ("mse", "spearman"):
                        a = row["metrics"][epoch][split]["uninterrupted"][metric]
                        b = reference[(row["scenario"], row["seed"])]["checkpoints"][epoch][split][
                            metric
                        ]
                        maximum = max(maximum, abs(a - b))
                        if not np.isclose(a, b, rtol=1e-8, atol=1e-9):
                            portable_failure = "Existing M22.1 fresh-run tolerance failed"
    passed = len(records) == 20 and all(r["pass"] for r in records) and portable_failure is None
    effects = [effect for r in records if r["pass"] for effect in r["paired_effects"]]
    seed_summaries = {}
    for scenario in config["scenarios"]:
        name = scenario["id"]
        seed_summaries[name] = {}
        selected = [r for r in records if r["scenario"] == name and r["pass"]]
        for epoch in ("12", "120"):
            seed_summaries[name][epoch] = {}
            for split in ("train", "validation", "test"):
                seed_summaries[name][epoch][split] = {}
                for metric in ("mse", "spearman"):
                    differences = [
                        r["metrics"][epoch][split]["bridge"][metric]
                        - r["metrics"][epoch][split]["uninterrupted"][metric]
                        for r in selected
                    ]
                    seed_summaries[name][epoch][split][metric] = {
                        "n_seeds": len(differences),
                        "mean": float(np.mean(differences)) if differences else None,
                        "sample_sd": float(np.std(differences, ddof=1))
                        if len(differences) > 1
                        else None,
                        "conditional_fixture_interval": [0.0, 0.0] if passed else None,
                    }
    result = {
        "schema": "m225-bridge-result-v1",
        "decision": "PASS" if passed else "FAIL",
        "preregistration": prereg,
        "environment": environment(),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "src/proteinmod/synthetic_bridge.py",
                "src/proteinmod/synthetic_regression.py",
                "src/proteinmod/synthetic_checkpoint.py",
                "scripts/run_m225.py",
            )
        },
        "records": records,
        "seed_summaries": seed_summaries,
        "portable_max_metric_difference": maximum,
        "portable_failure": portable_failure,
        "paired_resume_minus_full": {
            "n_cases": sum(r["pass"] for r in records),
            "mean": float(np.mean(effects)) if effects else None,
            "sample_sd": float(np.std(effects, ddof=1)) if len(effects) > 1 else None,
            "conditional_fixture_interval": [0.0, 0.0] if passed else None,
        },
        "limits": "Conditional engineering fixtures; no population/biological inference or cross-platform resume",
    }
    (output / "results.json").write_bytes(canonical(result) + b"\n")
    if not passed:
        raise RuntimeError("M22.5 FAIL; all cases retained in results.json")
    print(f"PASS: all 20 exact local bridges; separate portable max metric difference={maximum}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    run(args.output, args.compare)
