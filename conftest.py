"""Fail-closed combined synthetic collection on the dedicated integration branch."""

import os
import subprocess
from pathlib import Path

import pytest

BASE = "402bd8815dcd8857d416f47b1b41fdb922cf471d"
BRANCH = "codex/m224-synthetic-integration"
ALLOWED = {
    "conftest.py",
    "configs/m221_synthetic.json",
    "src/proteinmod/synthetic_regression.py",
    "src/proteinmod/synthetic_checkpoint.py",
    "scripts/run_m221.py",
    "scripts/verify_m221.py",
    "scripts/run_m223.py",
    "scripts/check_m223_preservation.py",
    "scripts/verify_m224.py",
    "tests/test_m221_synthetic.py",
    "tests/test_m223_checkpoint.py",
    "tests/test_m224_integration.py",
    ".github/workflows/m221-synthetic.yml",
    ".github/workflows/m223-synthetic.yml",
    ".github/workflows/m224-integration.yml",
}
TESTS = {"test_m221_synthetic.py", "test_m223_checkpoint.py", "test_m224_integration.py"}


def validate_synthetic_changes(lines):
    for line in lines:
        status, path = line.split("\t", 1)
        packet = any(path.startswith(f"benchmarks/{name}/") for name in ("m221", "m223", "m224"))
        if status != "A" or not (path in ALLOWED or packet):
            raise ValueError(f"Synthetic-only CI cannot hide an unrelated change: {line}")
    if not lines:
        raise ValueError("No synthetic changes to validate")


def pytest_configure(config):
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    config.m224_only = os.environ.get("GITHUB_ACTIONS") == "true" and branch == BRANCH
    if config.m224_only:
        changes = subprocess.check_output(
            ["git", "diff", "--name-status", BASE, "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).splitlines()
        try:
            validate_synthetic_changes(changes)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error


def pytest_ignore_collect(collection_path, config):
    if not config.m224_only:
        return None
    if collection_path.name.startswith("test_") and collection_path.suffix == ".py":
        return collection_path.name not in TESTS
    return None


def pytest_report_header(config):
    if config.m224_only:
        return "M22.4 additive scope verified; only M22.1/M22.3/M22.4 synthetic tests collected"
    return None
