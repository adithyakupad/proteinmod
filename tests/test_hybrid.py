"""Offline Q/K/V, mutation correspondence, masking and gradient checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.hybrid import MODES, CrossAttention, HybridRegressor


def graph():
    return {
        "static": np.arange(12, dtype=np.float32).reshape(3, 4) / 12,
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": np.ones((4, 3), dtype=np.float32),
    }


def test_attention_identity_qkv_mask_and_all_masked_gradients():
    attention = CrossAttention(4, 2)
    for layer in (attention.q, attention.k, attention.v, attention.out):
        layer.weight.data.copy_(torch.eye(4))
    q = torch.tensor([[[1.0, 0, 2, 0]]], requires_grad=True)
    context = torch.tensor([[[2.0, 0, 1, 0], [0, 1, 0, 1], [float("nan")] * 4]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    result, weights = attention(q, context, torch.ones(1, 1, dtype=torch.bool), mask)
    manual = torch.softmax(torch.tensor([2.0, 0]) / np.sqrt(2), dim=0)
    torch.testing.assert_close(weights[0, 0, 0, :2], manual)
    torch.testing.assert_close(result[0, 0, :2], torch.tensor([2 * manual[0], manual[1]]))
    assert (weights[..., 2] == 0).all()
    result.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(context.grad).all()
    zero, weights = attention(
        q, context, torch.ones(1, 1, dtype=torch.bool), torch.zeros_like(mask)
    )
    assert torch.equal(zero, torch.zeros_like(zero)) and (weights == 0).all()
    zero.sum().backward()
    assert torch.isfinite(q.grad).all()
    with pytest.raises(ValueError, match="divisible"):
        CrossAttention(3, 2)
    with pytest.raises(ValueError, match="dimensions"):
        attention(q, context, mask, mask)


@pytest.mark.parametrize("mode", MODES)
def test_modes_shapes_permutation_wt_and_gradients(mode):
    torch.manual_seed(3)
    model = HybridRegressor(graph(), 6, dimension=4, heads=2, mode=mode)
    residues = torch.randn(2, 5, 6)
    dynamic = torch.zeros(2, 3, 25)
    dynamic[0, [0, 2], 0] = 1
    positions = torch.tensor([0, 2, 4])  # missing reference residues never become graph keys
    a, weights = model(residues, dynamic, None, positions, diagnostics=True)
    b = model(residues, dynamic, None, positions, mutation_order=torch.tensor([1, 0]))
    torch.testing.assert_close(a, b)
    assert a.shape == (2,) and torch.isfinite(a).all()
    for w in weights.values():
        assert (w[1] == 0).all()  # WT has no mutation queries
        assert w.shape[2] == 2
    if "seq_queries_struct" in weights:
        assert weights["seq_queries_struct"].shape[-1] == 3
    if "struct_queries_seq" in weights:
        assert weights["struct_queries_seq"].shape[-1] == 5
    a.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if hasattr(model, "s_to_g"):
        assert model.s_to_g.q.weight.grad.abs().sum() > 0
    if hasattr(model, "g_to_s"):
        assert model.g_to_s.q.weight.grad.abs().sum() > 0
    if hasattr(model, "structure"):
        assert model.structure.messages[0].weight.grad.abs().sum() > 0


def test_reference_mutation_correspondence_and_masked_sequence_tokens():
    model = HybridRegressor(graph(), 6, dimension=4, heads=2)
    residues = torch.randn(1, 5, 6)
    dynamic = torch.zeros(1, 3, 25)
    dynamic[0, 1, 0] = 1
    positions = torch.tensor([0, 2, 4])
    sm = torch.tensor([[True, False, True, True, True]])
    a = model(residues, dynamic, None, positions, sequence_mask=sm)
    residues[:, 1] = float("nan")
    b = model(residues, dynamic, None, positions, sequence_mask=sm)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="correspondence"):
        model(residues, dynamic, None, torch.tensor([0, 0, 4]))
    gm = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="Mutation"):
        model(residues, dynamic, None, positions, graph_mask=gm)
    sm[:, 2] = False
    with pytest.raises(ValueError, match="Mutation"):
        model(residues, dynamic, None, positions, sequence_mask=sm)


def test_graph_node_permutation_and_query_reference_positions():
    torch.manual_seed(7)
    g = graph()
    model = HybridRegressor(g, 6, dimension=4, heads=2)
    residues = torch.randn(1, 5, 6)
    dynamic = torch.zeros(1, 3, 25)
    dynamic[0, 1, 0] = 1
    positions = torch.tensor([0, 2, 4])
    seen = []
    hook = model.s_to_g.q.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach()))
    a = model(residues, dynamic, None, positions)
    hook.remove()
    torch.testing.assert_close(seen[0][:, 0], model.sequence(residues)[:, 2])
    order = np.array([2, 0, 1])
    inverse = np.argsort(order)
    permuted = HybridRegressor(
        {
            "static": g["static"][order],
            "edge_index": inverse[g["edge_index"]],
            "edge_features": g["edge_features"],
        },
        6,
        dimension=4,
        heads=2,
    )
    state = {
        k: v
        for k, v in model.state_dict().items()
        if k
        not in {
            "structure." + name
            for name in ("static", "source", "target", "edge_features", "degree")
        }
    }
    permuted.load_state_dict(state, strict=False)
    torch.testing.assert_close(a, permuted(residues, dynamic[:, order], None, positions[order]))
