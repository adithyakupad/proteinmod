"""Run only the registered artificial regression diagnostic; never read protein data."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Set before numerical imports; record values with the result.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import scipy

from proteinmod.synthetic_regression import (
    fit,
    generate,
    load_checkpoint,
    metrics,
    save_checkpoint,
    snapshot_digest,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def run(output):
    config_path = ROOT / "configs/m221_synthetic.json"
    config = json.loads(config_path.read_text())
    prereg = json.loads((ROOT / "benchmarks/m221/preregistration.json").read_text())
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != prereg["config_sha256"]:
        raise ValueError("Registered synthetic config was changed")
    output = output.resolve()
    if not output.is_relative_to(ROOT / "runs/m221"):
        raise ValueError("Output must be a new directory under ignored runs/m221")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    started = time.perf_counter()
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            data = generate(config, scenario, seed)
            states = fit(config, *data["train"], seed)
            fresh = fit(config, *data["train"], seed)
            prefix = fit(config, *data["train"], seed, stop=12)[12]
            assert snapshot_digest(prefix) == snapshot_digest(states[12])
            checkpoints, digests = {}, {}
            for epoch, state in states.items():
                path = output / f"{scenario['id']}-{seed}-{epoch}.npz"
                save_checkpoint(path, state)
                restored = load_checkpoint(path)
                assert snapshot_digest(restored) == snapshot_digest(state)
                assert snapshot_digest(fresh[epoch]) == snapshot_digest(state)
                checkpoints[str(epoch)] = {
                    split: metrics(restored, *arrays) for split, arrays in data.items()
                }
                digests[str(epoch)] = snapshot_digest(restored)
                assert checkpoints[str(epoch)] == {
                    split: metrics(fresh[epoch], *arrays) for split, arrays in data.items()
                }
            resumed = fit(
                config,
                *data["train"],
                seed,
                resume=load_checkpoint(output / f"{scenario['id']}-{seed}-12.npz"),
            )[120]
            assert snapshot_digest(resumed) == snapshot_digest(states[120])
            effects = {
                split: {
                    metric: (
                        None
                        if checkpoints["12"][split][metric] is None
                        or checkpoints["120"][split][metric] is None
                        else checkpoints["120"][split][metric] - checkpoints["12"][split][metric]
                    )
                    for metric in ("mse", "spearman")
                }
                for split in data
            }
            records.append(
                {
                    "scenario": scenario["id"],
                    "seed": seed,
                    "checkpoints": checkpoints,
                    "paired_effects": effects,
                    "state_sha256": digests,
                    "training_sha256": states[12]["training_sha256"],
                    "train_mse_trace": states[120]["train_mse_trace"],
                    "full_replay_exact": True,
                    "prefix_exact": True,
                    "resume_exact": True,
                    "disk_roundtrip_exact": True,
                }
            )
            print(f"{scenario['id']} seed {seed}: complete; exact replay/prefix/resume", flush=True)
    result = {
        "schema": "m221-artificial-results-v1",
        "preregistration_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "benchmarks/m221/preregistration.json"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "config_sha256": prereg["config_sha256"],
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), ROOT / "src/proteinmod/synthetic_regression.py"]
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "dtype": "float64",
            "blas_thread_environment": 1,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
        "summary": summarize(records),
        "biological_inputs": False,
        "failures": [],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
