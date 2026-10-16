"""Offline edge-model invariances, gradients, held-out-label isolation and replay."""

import copy

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from proteinmod.edge_chemistry import EDGE_NAMES
from proteinmod.edge_experiments import fit, make_model, topology
from proteinmod.edge_gnn import EdgeGNN
from proteinmod.pretrained_structure import site_probe


def group():
    rng = np.random.default_rng(6)
    dynamic = rng.normal(size=(9, 4, 25)).astype(np.float32) * 0.1
    dynamic[:, :, 0] = [1, 0, 1, 0]
    return {
        "graph": {
            "static": rng.normal(size=(4, 34)).astype(np.float32),
            "nodes": list(range(4)),
            "edge_index": np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
            "edge_features": np.ones((6, 3), np.float32),
        },
        "dynamic": dynamic,
        "indices": np.arange(9),
        "edge_features": rng.normal(size=(6, len(EDGE_NAMES))).astype(np.float32),
    }


def test_node_and_mutation_set_invariance_gradients_and_shape():
    torch.manual_seed(4)
    g = group()
    graph = topology(g, "edge_full", 0)
    model = EdgeGNN()
    x = torch.tensor(g["dynamic"])
    pred = model(x, graph)
    pred.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    perm = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(perm)
    static, src, dst, e, degree = graph
    changed = (static[perm], inverse[src], inverse[dst], e, degree[:, perm])
    torch.testing.assert_close(pred, model(x[:, perm], changed), rtol=1e-5, atol=1e-6)
    emb = np.random.default_rng(1).normal(size=(4, 128))
    np.testing.assert_allclose(
        site_probe(emb, g["dynamic"]), site_probe(emb[perm], g["dynamic"][:, perm])
    )
    with pytest.raises(ValueError, match="25"):
        model(x[:, :, :24], graph)
    zero = EdgeGNN(zero_message=True)
    flipped = (static, dst, src, e * 32, degree)
    torch.testing.assert_close(zero(x, graph), zero(x, flipped), rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in zero.parameters())


def test_fixed_fit_test_label_blindness_and_state_replay(tmp_path):
    g = group()
    rows = [{"id": str(i), "protein_id": "p", "assay_id": "p", "target": i / 9} for i in range(9)]
    roles = {"train": np.arange(5), "validation": np.arange(5, 7), "test": np.arange(7, 9)}
    settings = {
        "device": "cpu",
        "hidden": 4,
        "layers": 2,
        "cpu_threads": 1,
        "batch_size": 2,
        "max_epochs": 2,
        "patience": 2,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
    }
    p, sel, state = fit({"p": g}, rows, roles, settings, 0, "edge_full", 8)
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = 9999
    q, sel2, state2 = fit({"p": g}, changed, roles, settings, 0, "edge_full", 8)
    np.testing.assert_array_equal(p, q)
    assert sel == sel2
    assert all(torch.equal(state[k], state2[k]) for k in state)
    path = tmp_path / "state.pt"
    torch.save(state, path)
    model = make_model(g, "edge_full", settings)
    model.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        replay = model(
            torch.tensor(g["dynamic"][roles["test"]]), topology(g, "edge_full", 8)
        ).numpy()
    np.testing.assert_array_equal(p, replay)
