"""Frozen M5 equivalence, missing modalities and independent kcal/mol head checks."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.hybrid import HybridRegressor
from proteinmod.stability_experiments import train_head
from proteinmod.stability_model import StabilityModel, encoder_digest


def fixture():
    graph = {
        "static": np.arange(12, dtype=np.float32).reshape(3, 4) / 12,
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": np.ones((4, 3), dtype=np.float32),
    }
    torch.manual_seed(3)
    original = HybridRegressor(graph, 6, 4, 2, 2, "bidirectional")
    model = StabilityModel(graph, 6, 4, 2, 2, original.state_dict())
    residues = torch.randn(2, 5, 6)
    dynamic = torch.zeros(2, 3, 25)
    dynamic[0, [0, 2], 0] = 1
    positions = torch.tensor([0, 2, 4])
    mutations = torch.zeros(2, 5, dtype=torch.bool)
    mutations[0, [0, 4]] = True
    return original, model, residues, dynamic, positions, mutations


def test_frozen_encoder_matches_historical_m5_and_only_head_gets_gradients():
    original, model, s, g, positions, mutations = fixture()
    original.head = torch.nn.Identity()
    expected = original(s, g, None, positions)
    features = model.represent(s, g, None, positions, mutations)
    torch.testing.assert_close(expected, features[:, :25], rtol=0, atol=0)
    assert features.shape == (2, 28)
    digest = encoder_digest(
        {k: v for k, v in original.state_dict().items() if not k.startswith("head.")}
    )
    model.train()
    assert not model.backbone.training
    model.target_mean.fill_(2.0)
    model.target_scale.fill_(3.0)
    prediction = model(s, g, None, positions, mutations)
    torch.testing.assert_close(
        prediction, model.stability_head(features).squeeze(-1).double() * 3 + 2
    )
    prediction.sum().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.stability_head.parameters()
    )
    assert encoder_digest(original.state_dict()) == digest


def test_missing_modality_cannot_contribute_and_partial_mapping_is_explicit():
    _, model, s, g, positions, mutations = fixture()
    mask = torch.tensor([[False, True], [True, False]])
    a = model.represent(s, g, None, positions, mutations, mask)
    changed = s.clone()
    changed[0] = float("nan")
    dg = g.clone()
    dg[1] = float("nan")
    b = model.represent(changed, dg, None, positions, mutations, mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (a[0, :12] == 0).all() and (a[1, 12:24] == 0).all()
    assert torch.isfinite(a).all()
    # A mutation without a graph node still has its sequence query and explicit coverage.
    mutations[0, 1] = True
    partial = model.represent(s, g, None, positions, mutations)
    assert partial[0, -1] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="modality"):
        model.represent(s, g, None, positions, mutations, torch.zeros(2, 2, dtype=torch.bool))
    wrong = g.clone()
    wrong[0, 0, 0] = 0
    with pytest.raises(ValueError, match="disagree"):
        model.represent(s, wrong, None, positions, mutations)


def test_stability_head_reproduces_and_uses_train_only_scaling():
    rows = [
        {
            "id": str(i),
            "target": float(i) / 10,
            "target_mask": True,
            "task_type": "ddg",
            "target_unit": "kcal/mol",
            "target_convention": "positive_destabilizing",
            "assay_id": "a",
            "conditions": {"pH": 7.4},
            "protein_id": "p",
        }
        for i in range(10)
    ]
    features = np.random.default_rng(0).normal(size=(10, 28)).astype(np.float32)
    roles = {"train": np.arange(6), "validation": np.array([6, 7]), "test": np.array([8, 9])}
    settings = {
        "device": "cpu",
        "cpu_threads": 1,
        "dimension": 4,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "max_epochs": 3,
        "patience": 2,
        "batch_size": 3,
    }
    a, selected, state = train_head(features, rows, roles, settings, 0)
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999
    b, other, second = train_head(features, changed, roles, settings, 0)
    np.testing.assert_array_equal(a, b)
    assert selected == other
    assert selected["train_only_target_mean"] == pytest.approx(0.25)
    assert all(
        torch.equal(state["stability_head"][k], v) for k, v in second["stability_head"].items()
    )


def test_head_loading_checks_encoder_schema_and_physical_calibration():
    _, model, s, g, positions, mutations = fixture()
    state = {
        "encoder_tensor_sha256": encoder_digest(model.backbone.state_dict()),
        "feature_schema": "m6-frozen-hybrid-v1",
        "stability_head": model.stability_head.state_dict(),
        "target_mean": 1.2345678901,
        "target_scale": 0.12345678901,
    }
    model.load_head(state)
    assert float(model.target_mean) == state["target_mean"]
    pred = model(s, g, None, positions, mutations)
    assert pred.dtype == torch.float64 and torch.isfinite(pred).all()
    with pytest.raises(ValueError, match="different"):
        model.load_head({**state, "encoder_tensor_sha256": "bad"})
    with pytest.raises(ValueError, match="calibration"):
        model.load_head({**state, "target_scale": 0})
