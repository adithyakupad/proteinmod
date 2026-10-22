"""Fail-closed M23 delegation through the existing synthetic CI plugin."""

import subprocess
from pathlib import Path

import pytest

BASE = "9f3fc9b6d158539e5758fe88c04d5af4f5b39b64"
EXTRA = {
    "src/proteinmod/geometry_study.py",
    "src/proteinmod/m23_collection.py",
    "scripts/run_m23.py",
    "scripts/verify_m23.py",
    "scripts/report_m23.py",
    "tests/test_m23_geometry.py",
    ".github/workflows/m23-geometry.yml",
}
MODIFIED = {"src/proteinmod/m225_collection.py", ".github/workflows/m225-bridge.yml"}
TESTS = {
    "test_m221_synthetic.py",
    "test_m223_checkpoint.py",
    "test_m224_integration.py",
    "test_m225_bridge.py",
    "test_m23_geometry.py",
}


def validate_changes(lines):
    if not lines:
        raise ValueError("No M23 changes")
    for line in lines:
        status, path = line.split("\t", 1)
        if not (
            (status == "A" and (path in EXTRA or path.startswith("benchmarks/m23/")))
            or (status == "M" and path in MODIFIED)
        ):
            raise ValueError(f"M23 CI rejects historical/unrelated change: {line}")


def configure(config):
    root = Path(__file__).resolve().parents[2]
    changes = subprocess.check_output(
        ["git", "diff", "--name-status", BASE, "HEAD"], cwd=root, text=True
    ).splitlines()
    try:
        validate_changes(changes)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    config.m23_only = True
    config.m225_only = False


def ignore(path):
    return (
        path.name not in TESTS if path.name.startswith("test_") and path.suffix == ".py" else None
    )
