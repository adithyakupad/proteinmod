"""Run reviewed M16 qualification commands from an isolated repository checkout.

Use the pinned environment. Cached acquisition/features are explicitly reused;
refit outputs and supervised weights have distinct identities. No network or merge.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def compare_csv(expected, actual, tolerance=1e-6):
    with Path(expected).open() as handle:
        old = list(csv.reader(handle))
    with Path(actual).open() as handle:
        new = list(csv.reader(handle))
    assert len(old) == len(new) and old[0] == new[0], (expected, actual)
    maximum = 0.0
    for left, right in zip(old[1:], new[1:], strict=True):
        assert len(left) == len(right)
        for name, a, b in zip(old[0], left, right, strict=True):
            if a == b:
                continue
            identity = {
                "id",
                "protein_id",
                "assay_id",
                "sequence",
                "mutations",
                "protocol",
                "fold",
                "family",
                "model",
                "mode",
                "seed",
                "fraction",
                "cohort",
                "role",
                "task",
                "repeat",
                "dataset",
                "source",
                "control",
                "target",
                "a",
                "b",
                "background",
            }
            assert name not in identity and not name.startswith(("measured", "observed")), (
                expected,
                name,
                a,
                b,
            )
            try:
                x, y = float(a), float(b)
            except ValueError:
                raise AssertionError((expected, a, b)) from None
            assert math.isfinite(x) and math.isfinite(y)
            maximum = max(maximum, abs(x - y))
            assert abs(x - y) <= tolerance, (expected, a, b)
    return {"rows": len(old) - 1, "maximum_absolute_difference": maximum}


def protected():
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for root in ("benchmarks", "configs")
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def scientific(value):
    """Exclude transport/timing identities, retaining selections, metrics and tensor digests."""
    if isinstance(value, list):
        return [scientific(v) for v in value]
    if isinstance(value, dict):
        incidental = {
            "checkpoint_sha256",
            "state_sha256",
            "checkpoint_path",
            "state_path",
            "files",
            "path",
            "seconds",
            "peak_rss_bytes",
        }
        return {
            k: scientific(v)
            for k, v in value.items()
            if k not in incidental and not k.endswith("_seconds")
        }
    return value


def commands(phase, output):
    py = sys.executable
    if phase == "tests":
        return [
            ("ruff", [py, "-m", "ruff", "check", "."]),
            ("format", [py, "-m", "ruff", "format", "--check", "."]),
            ("pytest", [py, "-m", "pytest", "-q"]),
            ("pip-check", [py, "-m", "pip", "check"]),
            ("cli-help", [py, "-m", "proteinmod.cli", "--help"]),
        ]
    if phase == "replay":
        result = [
            ("m5-m6-reference-replay", [py, "scripts/replay_m7_references.py"]),
            ("m7-refit-selected", [py, "scripts/verify_m7.py"]),
            ("m8-source-audit", [py, "scripts/audit_m8.py"]),
            ("m9-shift-audit", [py, "scripts/audit_m9.py"]),
        ]
        result += [(f"m{n}-replay", [py, f"scripts/verify_m{n}.py"]) for n in range(8, 12)]
        # These verifiers write validation summaries: use copied evidence, never archives.
        import shutil

        for n in range(12, 16):
            destination = output / f"m{n}-evidence"
            shutil.copytree(f"benchmarks/m{n}", destination)
            result.append(
                (f"m{n}-replay", [py, f"scripts/verify_m{n}.py", "--directory", str(destination)])
            )
        return result
    result = [
        ("m0-refit", [py, "-m", "proteinmod.cli", "run", "--output", "runs/m16-m0"]),
        (
            "m1-refit",
            [
                py,
                "-m",
                "proteinmod.cli",
                "run-m1",
                "--config",
                "configs/amylase_m1.toml",
                "--output",
                "runs/m16-m1",
            ],
        ),
        ("m2-rebuild", [py, "scripts/verify_structures.py", "--output", "runs/m16-m2"]),
        (
            "m3-refit",
            [
                py,
                "-m",
                "proteinmod.cli",
                "run-m3",
                "--config",
                "configs/amylase_m3.toml",
                "--output",
                "runs/m16-m3",
            ],
        ),
    ]
    modules = {
        5: "hybrid",
        6: "stability",
        7: "multitask",
        8: "conditioning",
        9: "assay",
        10: "reliability",
        11: "epistasis",
        12: "family",
        13: "uncertainty",
    }
    for n in range(4, 16):
        if n >= 14:
            module = "edge" if n == 14 else "lora"
            command = [py, "-m", f"proteinmod.{module}_experiments", "--output", f"runs/m16-m{n}"]
        else:
            config = json.loads(Path(f"configs/m{n}.json").read_text())
            if n == 4:
                config["experiment"]["output"] = f"runs/m16-m{n}"
            else:
                config["output"] = f"runs/m16-m{n}"
            path = output / f"m{n}-config.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            if n == 4:
                command = [py, "-m", "proteinmod.cli", "run-m4", "--config", str(path)]
            else:
                command = [py, "-m", f"proteinmod.{modules[n]}_experiments"]
                if n in (5, 6, 9, 10, 11):
                    command.append("run")
                command += ["--config", str(path)]
        result.append((f"m{n}-refit", command))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("tests", "replay", "refit", "compare"))
    args = parser.parse_args()
    output = Path("runs/m16") / args.phase
    output.mkdir(parents=True, exist_ok=False)
    if args.phase == "compare":
        results = {}
        for n in range(16):
            if n == 2:
                continue
            directory = {0: "amylase", 1: "amylase_m1", 3: "amylase_m3"}.get(n, f"m{n}")
            results[f"m{n}"] = {}
            for old in sorted(Path("benchmarks", directory).glob("*predictions.csv")):
                new = Path(f"runs/m16-m{n}", old.name)
                try:
                    results[f"m{n}"][old.name] = {"status": "PASS", **compare_csv(old, new)}
                except (AssertionError, OSError) as error:
                    results[f"m{n}"][old.name] = {"status": "FAIL", "reason": str(error)}
            for name in ("metrics.json", "runs.json", "baseline_runs.json"):
                old, new = Path("benchmarks", directory, name), Path(f"runs/m16-m{n}", name)
                if not old.exists():
                    continue
                try:
                    assert scientific(json.loads(old.read_text())) == scientific(
                        json.loads(new.read_text())
                    )
                    results[f"m{n}"][name] = {"status": "PASS", "scientific_fields_exact": True}
                except (AssertionError, OSError) as error:
                    results[f"m{n}"][name] = {"status": "FAIL", "reason": str(error)}
        (output / "predictions.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        sys.exit(any(r["status"] != "PASS" for files in results.values() for r in files.values()))
    before = protected()
    records = []
    environment = {
        **os.environ,
        "LOKY_MAX_CPU_COUNT": "4",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MPLCONFIGDIR": "/tmp/proteinmod-m16-mpl",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for name, command in commands(args.phase, output):
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as handle:
            code = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, env=environment, check=False
            ).returncode
        assert protected() == before, f"Historical artifacts changed during {name}"
        records.append(
            {
                "name": name,
                "command": command,
                "exit_code": code,
                "seconds": time.perf_counter() - start,
            }
        )
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print(name, code, round(records[-1]["seconds"], 3), flush=True)
    (output / "historical_hashes.json").write_text(json.dumps(before, indent=2) + "\n")
    sys.exit(any(r["exit_code"] for r in records))


if __name__ == "__main__":
    main()
