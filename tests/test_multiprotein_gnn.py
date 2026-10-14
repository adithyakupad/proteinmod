"""Tiny variable-topology training integration and held-out label independence."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.multiprotein_data import protocols
from proteinmod.multiprotein_experiments import graph_fit
from proteinmod.structural_gnn import ResidueGNN


def test_variable_graph_training_ignores_test_targets():
    rows, groups = [], {}
    for k, p in enumerate(["a", "b", "c"]):
        n = k + 2
        graph = {
            "static": np.ones((n, 4), dtype=np.float32),
            "edge_index": np.array([[0, 1], [1, 0]]),
            "edge_features": np.ones((2, 3), dtype=np.float32),
        }
        dynamic = np.zeros((4, n, 25), dtype=np.float32)
        dynamic[:, 0, 0] = 1
        dynamic[:, 0, 1] = np.arange(4)
        ix = np.arange(len(rows), len(rows) + 4)
        groups[p] = {"graph": graph, "dynamic": dynamic, "indices": ix}
        rows.extend(
            {
                "id": f"{p}:{j}",
                "protein_id": p,
                "assay_id": p,
                "sequence": p + str(j),
                "target": j,
                "partition": "train" if j < 2 else "validation" if j == 2 else "test",
            }
            for j in range(4)
        )
    roles = protocols(rows, ["a", "b", "c"])[1][2]
    settings = {
        "device": "cpu",
        "hidden": 4,
        "layers": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 2,
    }
    a, selected, _ = graph_fit(groups, rows, roles, settings, 0, 1)
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -1234
    b, other, _ = graph_fit(groups, changed, roles, settings, 0, 1)
    np.testing.assert_array_equal(a, b)
    assert selected == other and np.isfinite(a).all()
    g = groups["a"]["graph"]
    m = ResidueGNN(g["static"], g["edge_index"], g["edge_features"], hidden=4, layers=2)
    topology = (m.static, m.source, m.target, m.edge_features, m.degree)
    x = torch.as_tensor(groups["a"]["dynamic"])
    torch.testing.assert_close(m(x), m(x, topology), rtol=0, atol=0)


def test_full_runner_small_collection(tmp_path, monkeypatch):
    from proteinmod import multiprotein_experiments as experiment

    version = experiment.importlib.metadata.version

    def without_transformers(name):
        if name == "transformers":
            raise experiment.importlib.metadata.PackageNotFoundError(name)
        return version(name)

    monkeypatch.setattr(experiment.importlib.metadata, "version", without_transformers)

    rows, groups = [], {}
    for k, p in enumerate(["a", "b", "c"]):
        ix = np.arange(len(rows), len(rows) + 8)
        graph = {
            "static": np.ones((k + 2, 4), dtype=np.float32),
            "edge_index": np.array([[0, 1], [1, 0]]),
            "edge_features": np.ones((2, 3), dtype=np.float32),
        }
        dynamic = np.zeros((8, k + 2, 25), dtype=np.float32)
        dynamic[:, 0, 0] = 1
        dynamic[:, 0, 1] = np.arange(8)
        groups[p] = {"graph": graph, "dynamic": dynamic, "indices": ix}
        rows.extend(
            {
                "id": f"{p}:{j}",
                "protein_id": p,
                "assay_id": p,
                "sequence": ["AC", "DE", "FG"][k] + "ACDEFGHI"[j],
                "target": j,
                "partition": "train" if j < 4 else "validation" if j < 6 else "test",
                "structural_valid": True,
                "structural_exclusions": [],
            }
            for j in range(8)
        )
    features = {
        n: np.arange(24 * 4, dtype=float).reshape(24, 4)
        for n in ["sequence_ridge", "mutation_count_ridge", "esm_sites", "structural_ridge"]
    }
    monkeypatch.setattr(
        experiment, "model_inputs", lambda _: (rows, features, np.ones(24, dtype=bool), groups, {})
    )
    config = {
        "collection": [],
        "experiment": {
            "output": str(tmp_path / "results"),
            "fold_order": ["a", "b", "c"],
            "seeds": [0],
            "alphas": [1.0],
            "cpu_threads": 1,
        },
        "gnn": {
            "device": "cpu",
            "hidden": 4,
            "layers": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "max_epochs": 2,
            "patience": 1,
            "batch_size": 4,
        },
    }
    result = experiment.run_collection(config)
    assert {r["protocol"] for r in result} == {"G1", "G2"}
    assert any(r["model"] == "within_one_hot" for r in result)
    assert all(r["fold"] in ["a", "b", "c"] for r in result if r["protocol"] == "G2")
    assert (tmp_path / "results" / "predictions.csv").is_file()
