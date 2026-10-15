"""Unpaired masking, shared gradients, mapping and held-out target isolation."""

import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.multitask import MultitaskModel, calibrate, masked_loss
from proteinmod.multitask_experiments import audit_combined, fit, output_path


def example():
    graph = {
        "static": np.ones((3, 4), dtype=np.float32),
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": np.ones((4, 3), dtype=np.float32),
    }
    torch.manual_seed(1)
    model = MultitaskModel(graph, 6, 4)
    s = torch.randn(2, 5, 6)
    d = torch.zeros(2, 3, 25)
    d[:, [0, 2], 0] = 1
    positions = torch.tensor([0, 2, 4])
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, [0, 4]] = True
    return model, (s, d, None, positions, mask), graph


def test_mask_before_arithmetic_and_reject_all_missing():
    p = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[0.0, float("nan")], [float("nan"), 0.0]])
    mask = torch.isfinite(y)
    loss = masked_loss(p, y, mask, (1, 3))
    assert loss.item() == pytest.approx((1 + 3 * 16) / 4)
    loss.backward()
    assert p.grad[0, 1] == p.grad[1, 0] == 0
    y[~mask] = -9999
    assert masked_loss(p, y, mask, (1, 3)).item() == loss.item()
    with pytest.raises(ValueError, match="missing"):
        masked_loss(p, y, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="Nonfinite"):
        masked_loss(p, torch.full_like(y, float("nan")), mask)


def test_both_tasks_backpropagate_shared_encoder_but_not_other_head():
    model, args, _ = example()
    for j, task in enumerate(("function", "stability")):
        model.zero_grad(set_to_none=True)
        out = model(*args)
        mask = torch.zeros_like(out, dtype=torch.bool)
        mask[:, j] = True
        masked_loss(out, torch.zeros_like(out), mask).backward()
        for prefix in ("sequence", "structure", "s_to_g", "g_to_s"):
            grads = [
                p.grad for n, p in model.encoder.backbone.named_parameters() if n.startswith(prefix)
            ]
            assert any(g is not None and g.abs().sum() > 0 for g in grads)
        other = "stability" if task == "function" else "function"
        assert all(p.grad is None or not p.grad.any() for p in model.heads[other].parameters())


def test_frozen_mode_save_load_and_permutation_invariance():
    model, args, graph = example()
    model.encoder.backbone.requires_grad_(False)
    model(*args).sum().backward()
    assert all(p.grad is None for p in model.encoder.backbone.parameters())
    expected = model(*args)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    restored = MultitaskModel(graph, 6, 4)
    restored.load_state_dict(torch.load(buf, weights_only=True))
    torch.testing.assert_close(restored(*args), expected, rtol=0, atol=0)
    # Graph node order changes together with reference correspondence and topology.
    s, d, _, pos, mut = args
    order = torch.tensor([2, 0, 1])
    old = model.encoder.backbone.structure
    inverse = torch.argsort(order)
    top = (
        old.static[order],
        inverse[old.source],
        inverse[old.target],
        old.edge_features,
        old.degree[:, order],
    )
    torch.testing.assert_close(
        model(s, d[:, order], top, pos[order], mut), expected, rtol=1e-5, atol=1e-6
    )
    # Mutation sets are masks, so insertion order is erased before attention/readout.
    reversed_mask = torch.zeros_like(mut)
    reversed_mask[:, [4, 0]] = True
    torch.testing.assert_close(model(s, d, None, pos, reversed_mask), expected, rtol=0, atol=0)
    available = torch.tensor([[True, False], [False, True]])
    assert torch.isfinite(model(*args, modality_mask=available)).all()


def test_calibrators_and_combined_leakage():
    rows = [
        {
            "id": str(i),
            "target": float(i),
            "assay_id": "a",
            "sequence": str(i),
            "protein_id": str(i),
            "target_unit": "kcal/mol",
            "target_convention": "positive_destabilizing",
        }
        for i in range(6)
    ]
    train = np.array([0, 1, 2])
    for task in ("function", "stability"):
        y, c = calibrate(rows, train, task)
        changed = copy.deepcopy(rows)
        for r in changed[3:]:
            r["target"] = -1e9
        yy, cc = calibrate(changed, train, task)
        np.testing.assert_array_equal(y, yy)
        assert c == cc and np.isnan(y[3:]).all()
    roles = {"train": np.array([0, 1]), "validation": np.array([2, 3]), "test": np.array([4, 5])}
    other = copy.deepcopy(rows)
    for r in other:
        r["protein_id"] += "other"
        r["sequence"] += "other"
    audit_combined(
        {"function": rows, "stability": other}, {"function": roles, "stability": roles}, True
    )
    other[0]["protein_id"] = rows[4]["protein_id"]
    with pytest.raises(ValueError, match="Cross-task"):
        audit_combined(
            {"function": rows, "stability": other}, {"function": roles, "stability": roles}, True
        )
    with pytest.raises(ValueError, match="historical"):
        output_path("benchmarks/m6/overwritten")


def test_training_never_uses_test_outcomes_and_reproduces():
    _, args, graph = example()
    rng = np.random.default_rng(0)
    rows, groups = {}, {}
    roles = {"train": np.arange(4), "validation": np.arange(4, 6), "test": np.arange(6, 8)}
    for task in ("function", "stability"):
        rows[task] = [
            {
                "id": f"{task}{i}",
                "target": i / 10,
                "protein_id": task,
                "assay_id": task,
                "mutations": ("A1V", "A5V"),
                "sequence": str(i),
                "target_unit": "kcal/mol",
                "target_convention": "positive_destabilizing",
            }
            for i in range(8)
        ]
        groups[task] = {
            task: {
                "indices": np.arange(8),
                "graph": {**graph, "nodes": [{"reference_position": i} for i in (1, 3, 5)]},
                "residues": rng.normal(size=(8, 5, 6)).astype(np.float32),
                "dynamic": np.tile(args[1][:1].numpy(), (8, 1, 1)),
                "token_indices": np.arange(8),
            }
        }
    settings = {
        **json.loads(Path("configs/m7.json").read_text())["training"],
        "dimension": 4,
        "max_epochs": 3,
        "warmup_epochs": 1,
        "cpu_threads": 1,
    }
    a, selected, state = fit(rows, groups, {t: roles for t in rows}, settings, 0, "joint")
    changed = copy.deepcopy(rows)
    for task in changed:
        for i in roles["test"]:
            changed[task][i]["target"] = -9999
    b, other, second = fit(changed, groups, {t: roles for t in rows}, settings, 0, "joint")
    for task in rows:
        np.testing.assert_array_equal(a[task], b[task])
    for key in ("epoch", "score", "trace", "calibration", "examples_processed", "gradient"):
        assert selected[key] == other[key]
    assert selected["gradient"]["norms"][0] > 0 and selected["gradient"]["norms"][1] > 0
    assert all(torch.equal(state[k], v) for k, v in second.items())
