"""Tiny variable-protein fit proves test labels cannot change selection or predictions."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.hybrid_experiments import fit, output_path
from proteinmod.multiprotein_data import protocols


def fixture():
    rows, groups = [], {}
    for k, p in enumerate(("a", "b", "c")):
        n = k + 2
        graph = {
            "static": np.arange(n * 4, dtype=np.float32).reshape(n, 4) / 10,
            "edge_index": np.array([[0, 1], [1, 0]]),
            "edge_features": np.ones((2, 3), dtype=np.float32),
            "nodes": [{"reference_position": j + 1} for j in range(n)],
        }
        dynamic = np.zeros((6, n, 25), dtype=np.float32)
        dynamic[:, 0, 0] = 1
        dynamic[:, 0, 1] = np.arange(6) / 6
        groups[p] = {
            "graph": graph,
            "dynamic": dynamic,
            "indices": np.arange(len(rows), len(rows) + 6),
            "residues": np.random.default_rng(k).normal(size=(6, n, 6)).astype(np.float32),
            "token_indices": np.arange(6),
        }
        rows.extend(
            {
                "id": f"{p}:{j}",
                "protein_id": p,
                "assay_id": p,
                "sequence": p + str(j),
                "target": j,
                "partition": "train" if j < 2 else "validation" if j < 4 else "test",
            }
            for j in range(6)
        )
    return rows, groups


def test_held_out_labels_and_reproducibility():
    rows, groups = fixture()
    roles = protocols(rows, ["a", "b", "c"])[1][2]
    settings = {
        "device": "cpu",
        "dimension": 4,
        "heads": 2,
        "layers": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 3,
        "cpu_threads": 1,
    }
    a, selected, _ = fit(groups, rows, roles, settings, 0, "bidirectional")
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -9999
    b, other, _ = fit(groups, changed, roles, settings, 0, "bidirectional")
    np.testing.assert_array_equal(a, b)
    assert selected == other
    assert np.isfinite(a).all()


def test_history_output_protected():
    for output in (
        "benchmarks",
        "benchmarks/m4",
        "benchmarks/amylase_m3",
        "benchmarks/m4/new",
        ".",
    ):
        with pytest.raises(ValueError, match="historical"):
            output_path({"output": output})
