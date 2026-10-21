"""Fail-closed synthetic-only CI collection on the dedicated M22.1 branch.

Normal/local collection is unchanged. No historical test is imported on this
isolated CI branch, and changing any old file aborts rather than hiding its tests.
"""

import os
import subprocess
from pathlib import Path

import pytest

BASE = "402bd8815dcd8857d416f47b1b41fdb922cf471d"
BRANCH = "codex/m221-synthetic-horizon"
ALLOWED = {
    "conftest.py",
    "configs/m221_synthetic.json",
    "src/proteinmod/synthetic_regression.py",
    "scripts/run_m221.py",
    "scripts/verify_m221.py",
    "tests/test_m221_synthetic.py",
    ".github/workflows/m221-synthetic.yml",
}


def validate_synthetic_changes(lines):
    for line in lines:
        status, path = line.split("\t", 1)
        if status != "A" or not (path in ALLOWED or path.startswith("benchmarks/m221/")):
            raise ValueError(f"Synthetic-only CI cannot hide an unrelated change: {line}")
    if not lines:
        raise ValueError("No synthetic changes to validate")


def pytest_configure(config):
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    config.m221_only = os.environ.get("GITHUB_ACTIONS") == "true" and branch == BRANCH
    if config.m221_only:
        root = Path(__file__).resolve().parent
        changes = subprocess.check_output(
            ["git", "diff", "--name-status", BASE, "HEAD"], cwd=root, text=True
        ).splitlines()
        try:
            validate_synthetic_changes(changes)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error


def pytest_ignore_collect(collection_path, config):
    if not config.m221_only:
        return None
    if collection_path.name.startswith("test_") and collection_path.suffix == ".py":
        return collection_path.name != "test_m221_synthetic.py"
    return None


def pytest_report_header(config):
    if config.m221_only:
        return "M22.1 synthetic-only CI: additive scope verified; historical tests not collected"
    return None
