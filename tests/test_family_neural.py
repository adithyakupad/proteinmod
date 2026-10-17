"""Tiny G3 fit verifies fresh variable-topology training and held-out label isolation."""

import copy

import numpy as np
import pytest

pytest.importorskip("torch")
from proteinmod.family_data import folds
from proteinmod.family_experiments import aggregate
from proteinmod.hybrid_experiments import fit


def test_g3_neural_fit_test_labels_and_reproducibility():
    rows, groups = [], {}
    for k in range(8):
        pid = f"p{k}"
        n = 2 + k % 2
        graph = {
            "static": np.zeros((n, 4), np.float32),
            "edge_index": np.array([[0, 1], [1, 0]]),
            "edge_features": np.ones((2, 3), np.float32),
            "nodes": [{"reference_position": i + 1} for i in range(n)],
        }
        dynamic = np.zeros((6, n, 25), np.float32)
        dynamic[:, 0, 0] = 1
        dynamic[:, 0, 1] = np.arange(6) / 6
        groups[pid] = {
            "graph": graph,
            "dynamic": dynamic,
            "indices": np.arange(len(rows), len(rows) + 6),
            "token_indices": np.arange(6),
            "residues": np.random.default_rng(k).normal(size=(6, n, 6)).astype(np.float32),
        }
        rows.extend(
            {
                "id": f"{pid}:{j}",
                "protein_id": pid,
                "family": f"f{k // 2}",
                "assay_id": pid,
                "sequence": f"{pid}:{j}",
                "target": j,
                "partition": "train" if j < 2 else "validation" if j < 4 else "test",
            }
            for j in range(6)
        )
    roles = next(r for p, _, r in folds(rows, ["f0", "f1", "f2", "f3"]) if p == "G3")
    settings = {
        "dimension": 4,
        "heads": 2,
        "layers": 2,
        "device": "cpu",
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 3,
        "cpu_threads": 1,
    }
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999 - i
    for mode in ("structure_only", "weighted_fusion", "bidirectional"):
        pred, selection, state = fit(groups, rows, roles, settings, 0, mode)
        again, selected, other = fit(groups, changed, roles, settings, 0, mode)
        np.testing.assert_array_equal(pred, again)
        assert selection == selected
        assert all(np.array_equal(v.numpy(), other[k].numpy()) for k, v in state.items())
        result = aggregate(rows, roles["test"], pred)
        assert set(result["per_family"]) == {"f0"} and np.isfinite(result["family_macro_spearman"])


def test_family_runner_protects_archived_outputs_and_preregistration():
    import json
    from pathlib import Path

    from proteinmod.family_experiments import run

    config = json.loads(Path("configs/m12.json").read_text())
    for output in (".", "benchmarks", "benchmarks/m11", "benchmarks/m12/new"):
        with pytest.raises(ValueError, match="archived"):
            run({**config, "output": output})
    with pytest.raises(ValueError, match="Preregistered"):
        run({**config, "seeds": [999]})
