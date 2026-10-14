"""Tiny end-to-end M3 run exercises ridge, neural selection, predictions and manifests."""

import json

import numpy as np
import pytest

from proteinmod import structural_experiments as experiments


def test_matched_runner(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    rows = [
        {"id": str(i), "target": i / 10, "sequence": "ACD", "mutations": ["A1V"]} for i in range(8)
    ]
    partitions = {"train": np.arange(4), "validation": np.array([4, 5]), "test": np.array([6, 7])}
    features = {"structural_ridge": np.arange(16, dtype=float).reshape(8, 2)}
    dynamic = np.zeros((8, 3, 25), dtype=np.float32)
    dynamic[:, 0, 0] = 1
    graph = {
        "static": np.ones((3, 2)),
        "edge_index": np.array([[0, 1], [1, 0]]),
        "edge_features": np.ones((2, 3)),
    }
    monkeypatch.setattr(
        experiments, "prepare", lambda _: (rows, partitions, features, dynamic, graph, {})
    )
    config = {
        "m3": {
            "output": str(tmp_path / "output"),
            "cache": str(tmp_path / "cache"),
            "alphas": [1.0],
            "seeds": [0, 1],
            "fractions": [1.0],
        },
        "gnn": {
            "device": "cpu",
            "cpu_threads": 1,
            "hidden": 4,
            "layers": [0, 2],
            "learning_rate": 0.01,
            "weight_decays": [0.01],
            "batch_size": 2,
            "max_epochs": 2,
            "patience": 1,
        },
    }
    result = experiments.run_structural(config)
    assert {r["model"] for r in result} == {"structural_ridge", "structural_mlp", "structural_gnn"}
    output = tmp_path / "output"
    runs = json.loads((output / "runs.json").read_text())
    assert len(runs) == 6
    assert all(r["n_train"] == 4 for r in runs)
    audit = json.loads((output / "audit.json").read_text())
    assert audit["gnn_reproducibility"]["max_prediction_difference"] == 0
    assert (output / "test_predictions.csv").read_text().count("\n") == 13
