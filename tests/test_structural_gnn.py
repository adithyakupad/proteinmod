"""Offline neural checks; the optional torch CI job runs these on tiny graphs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.structural_gnn import ResidueGNN, train_gnn


def tiny():
    return {
        "static": np.arange(12, dtype=np.float32).reshape(3, 4) / 12,
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": np.array([[0.5, 1, 1]] * 4, dtype=np.float32),
    }


def test_node_permutation_invariance_and_mutation_signal():
    torch.manual_seed(1)
    graph = tiny()
    x = torch.zeros(2, 3, 25)
    x[0, 0, 0] = 1
    x[1, 1:, 0] = 1
    model = ResidueGNN(**graph, hidden=4, layers=2).eval()
    original = model(x)
    order = np.array([2, 0, 1])
    inverse = np.argsort(order)
    permuted = ResidueGNN(
        graph["static"][order],
        inverse[graph["edge_index"]],
        graph["edge_features"],
        hidden=4,
        layers=2,
    )
    state = {
        k: v
        for k, v in model.state_dict().items()
        if k not in ("static", "source", "target", "edge_features", "degree")
    }
    permuted.load_state_dict(state, strict=False)
    torch.testing.assert_close(original, permuted(x[:, order]))
    assert not torch.isclose(original[0], original[1])
    original.sum().backward()
    assert model.messages[0].weight.grad.abs().sum() > 0


def test_tiny_training_reproduces_and_test_labels_cannot_select():
    graph = tiny()
    x = np.zeros((8, 3, 25), dtype=np.float32)
    x[:, 0, 0] = 1
    x[:, 0, 1] = np.arange(8) / 8
    y = np.arange(8, dtype=float) / 10
    settings = {
        "device": "cpu",
        "cpu_threads": 1,
        "hidden": 4,
        "learning_rate": 0.01,
        "weight_decays": [0.01],
        "batch_size": 2,
        "max_epochs": 3,
        "patience": 2,
    }
    indices = (np.arange(4), np.array([4, 5]), np.array([6, 7]))
    a, p, _ = train_gnn(graph, x, y, *indices, settings, seed=0, layers=2)
    b, q, _ = train_gnn(graph, x, y, *indices, settings, seed=0, layers=2)
    np.testing.assert_array_equal(p, q)
    assert a["selection"] == b["selection"]
    y[6:] = -1000
    c, r, _ = train_gnn(graph, x, y, *indices, settings, seed=0, layers=2)
    assert a["selection"] == c["selection"]
    np.testing.assert_array_equal(p, r)
    assert np.isfinite(p).all()
    assert a["train"]["rmse"] >= 0


def test_no_message_passing_control_and_disjoint_contract():
    graph = tiny()
    model = ResidueGNN(**graph, hidden=4, layers=0)
    assert len(model.messages) == 0
    assert model(torch.zeros(1, 3, 25)).shape == (1,)
    with pytest.raises(ValueError, match="CPU"):
        train_gnn(graph, None, None, None, None, None, {"device": "mps"}, 0, 2)
    with pytest.raises(ValueError, match="disjoint"):
        train_gnn(
            graph,
            np.zeros((3, 3, 25)),
            np.zeros(3),
            np.array([0]),
            np.array([0]),
            np.array([2]),
            {
                "device": "cpu",
                "cpu_threads": 1,
                "hidden": 4,
                "learning_rate": 0.01,
                "weight_decays": [0.01],
                "batch_size": 1,
                "max_epochs": 1,
                "patience": 1,
            },
            0,
            2,
        )
