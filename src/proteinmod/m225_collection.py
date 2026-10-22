"""Additive synthetic-only CI collection without editing PR #64's frozen router."""

import os
import subprocess
from pathlib import Path

import pytest

PARENT = "3482d76961115078e97a1489c0d80f3325da8402"
EXTRA = {
    "pytest.ini",
    ".github/workflows/m225-bridge.yml",
    "src/proteinmod/synthetic_bridge.py",
    "src/proteinmod/m225_collection.py",
    "scripts/run_m225.py",
    "scripts/verify_m225.py",
    "tests/test_m225_bridge.py",
}
TESTS = {
    "test_m221_synthetic.py",
    "test_m223_checkpoint.py",
    "test_m224_integration.py",
    "test_m225_bridge.py",
}


def validate_changes(lines):
    if not lines:
        raise ValueError("No M22.5 additions")
    for line in lines:
        status, path = line.split("\t", 1)
        if status != "A" or not (path in EXTRA or path.startswith("benchmarks/m225/")):
            raise ValueError(f"M22.5 cannot hide a parent or unrelated change: {line}")


def pytest_configure(config):
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    if os.environ.get("GITHUB_ACTIONS") == "true" and branch == "codex/m23-representation-geometry":
        from proteinmod.m23_collection import configure

        configure(config)
        return
    config.m225_only = (
        os.environ.get("GITHUB_ACTIONS") == "true" and branch == "codex/m225-checkpoint-bridge"
    )
    if config.m225_only:
        changes = subprocess.check_output(
            ["git", "diff", "--name-status", PARENT, "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).splitlines()
        try:
            validate_changes(changes)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error


def pytest_ignore_collect(collection_path, config):
    if getattr(config, "m23_only", False):
        from proteinmod.m23_collection import ignore

        return ignore(collection_path)
    if (
        config.m225_only
        and collection_path.name.startswith("test_")
        and collection_path.suffix == ".py"
    ):
        return collection_path.name not in TESTS
    return None
