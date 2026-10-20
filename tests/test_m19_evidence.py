"""Offline reconstruction of archived M19 metrics; no external outcomes or caches."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def test_m19_archive():
    root = Path("benchmarks/m19")
    if not (root / "summary.json").exists():
        import pytest

        pytest.skip("Study not yet archived")
    for path, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    grouped = defaultdict(list)
    with (root / "predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = row["fold"], float(row["fraction"]), int(row["seed"]), row["model"]
            grouped[key + (row["protein_id"],)].append(
                (float(row["target"]), float(row["prediction"]))
            )
    records = json.loads((root / "runs.json").read_text())
    assert len(records) == 480
    for row in records:
        key = row["fold"], row["fraction"], row["seed"], row["model"]
        scores = []
        for pid in row["per_protein"]:
            values = np.asarray(grouped[key + (pid,)])
            score = float(spearmanr(values[:, 0], values[:, 1]).statistic)
            assert abs(score - row["per_protein"][pid]["spearman_rho"]) < 1e-12
            scores.append(score)
        assert abs(np.mean(scores) - row["rho"]) < 1e-12
        if row["model"].startswith("expanded") or row["model"] == "shuffled_train":
            assert len(row["selection"]["candidates"]) == 9
    effects = json.loads((root / "family-effects.json").read_text())
    primary = [
        r["minus_historical_unscaled"]
        for r in effects
        if r["fraction"] == 1 and r["model"] == "expanded_selected"
    ]
    summary = json.loads((root / "summary.json").read_text())
    assert abs(np.mean(primary) - summary["primary_gain"]) < 1e-12
    assert summary["success"] == (
        np.mean(primary) >= 0.02
        and sum(v > 0 for v in primary) >= 3
        and not summary["scalar_failures"]
    )
