"""Tiny randomly initialized ESM offline: actual LoRA paths, masks and replay."""

import copy

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
import torch
from transformers import EsmConfig, EsmModel

from proteinmod.lora_experiments import fit
from proteinmod.lora_model import AdaptedESM, LoRALinear, site_representation


def tiny():
    torch.manual_seed(0)
    model = EsmModel(
        EsmConfig(
            vocab_size=33,
            hidden_size=12,
            num_hidden_layers=2,
            num_attention_heads=3,
            intermediate_size=24,
            pad_token_id=1,
            mask_token_id=32,
            position_embedding_type="rotary",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        ),
        add_pooling_layer=False,
    )
    settings = {
        "layers": [1],
        "projections": ["query", "value"],
        "rank": 2,
        "alpha": 4,
        "dropout": 0.0,
    }
    return model, settings


def test_lora_zero_identity_gradients_frozen_weights_and_capacity(tmp_path):
    base, settings = tiny()
    model = AdaptedESM(base, settings, 4, "lora", 0)
    frozen = AdaptedESM(base, settings, 4, "frozen_head", 0)
    matched = AdaptedESM(base, settings, 4, "matched_head", 0)
    counts = [
        sum(p.numel() for p in x.parameters() if p.requires_grad) for x in (model, frozen, matched)
    ]
    assert counts[0] == counts[2] > counts[1]
    assert all(not p.requires_grad for p in base.parameters())
    layer = model.tail.attention.self.query
    assert isinstance(layer, LoRALinear)
    x = torch.randn(3, 6, 12, requires_grad=True)
    ext = torch.zeros(3, 1, 1, 6)
    mask = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 0]])
    before = model.frozen_digest()
    features = model.features(x, ext, mask)
    with torch.no_grad():
        hidden = frozen.norm(frozen.tail(x, attention_mask=ext))[:, 1:-1]
        expected = site_representation(hidden[:-1], hidden[-1:].expand(2, -1, -1), mask)
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        model(prefix=x, extended_mask=ext, mutation_mask=mask),
        frozen(features=expected),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        frozen(features=expected), matched(features=expected), rtol=0, atol=0
    )
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(prefix=x, extended_mask=ext, mutation_mask=mask).square().sum().backward()
    assert x.grad[-1].abs().sum() > 0, "WT delta path must receive gradients"
    assert layer.adapter.b.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    optimizer.step()
    assert model.frozen_digest() == before
    assert not torch.equal(model.features(x, ext, mask), expected)
    path = tmp_path / "adapter.pt"
    torch.save(model.selected_state(), path)
    restored = AdaptedESM(base, settings, 4, "lora", 0)
    restored.load_selected(torch.load(path, weights_only=True))
    torch.testing.assert_close(
        model(prefix=x, extended_mask=ext, mutation_mask=mask),
        restored(prefix=x, extended_mask=ext, mutation_mask=mask),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="schema"):
        restored.load_selected({})


def test_mutation_set_and_padding_masks():
    base, settings = tiny()
    model = AdaptedESM(base, settings, 4)
    x = torch.randn(3, 7, 12)
    extended = torch.zeros(3, 1, 1, 7)
    extended[:, :, :, 5:] = torch.finfo(torch.float32).min
    mask = torch.tensor([[1.0, 0, 1, 0, 0], [0, 1, 0, 0, 0]])
    a = model.features(x, extended, mask)
    changed = x.clone()
    changed[:, 5:] += 100
    torch.testing.assert_close(a, model.features(changed, extended, mask), rtol=1e-5, atol=1e-6)
    bad = mask.clone()
    bad[:, -1] = 1
    with pytest.raises(ValueError, match="padding"):
        model.features(x, extended, bad)
    h = torch.randn(2, 4, 12)
    wt = torch.randn(2, 4, 12)
    m = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]])
    permutation = [2, 0, 3, 1]
    torch.testing.assert_close(
        site_representation(h, wt, m),
        site_representation(h[:, permutation], wt[:, permutation], m[:, permutation]),
    )


def test_held_out_labels_do_not_change_adapters_or_selection():
    base, settings = tiny()
    rng = np.random.default_rng(2)
    group = {
        "prefix": rng.normal(size=(10, 6, 12)).astype(np.float32),
        "mutation_mask": np.tile([1.0, 0, 1, 0], (9, 1)).astype(np.float32),
        "indices": np.arange(9),
        "frozen": np.zeros((9, 25), np.float32),
    }
    rows = [{"protein_id": "p", "assay_id": "p", "target": i} for i in range(9)]
    roles = {"train": np.arange(5), "validation": np.arange(5, 7), "test": np.arange(7, 9)}
    config = {
        "lora": settings,
        "training": {
            "hidden": 4,
            "cpu_threads": 1,
            "learning_rates": [0.001],
            "adapter_lr_ratio": 0.1,
            "weight_decay": 0.01,
            "max_epochs": 2,
            "patience": 2,
            "batch_size": 3,
        },
    }
    a, sa, wa = fit(base, {"p": group}, rows, roles, config, 0, "lora")
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = 999
    b, sb, wb = fit(base, {"p": group}, changed, roles, config, 0, "lora")
    np.testing.assert_array_equal(a, b)
    assert sa == sb
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_run_rejects_historical_outputs_and_existing_checkpoints(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from proteinmod.lora_experiments import run

    config = json.loads(Path("configs/m15.json").read_text())
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/m15.json").write_text(json.dumps(config))
    old = Path("benchmarks/m14/metrics.json")
    old.parent.mkdir(parents=True)
    old.write_text("preserved")
    with pytest.raises(ValueError, match="runs/"):
        run(config | {"output": "benchmarks/m14"})
    cache = Path(config["cache"]) / "weights" / "reused"
    cache.mkdir(parents=True)
    weight = cache / "old.pt"
    weight.write_bytes(b"preserved weights")
    with pytest.raises(ValueError, match="Existing"):
        run(config | {"output": "runs/reused"})
    assert old.read_text() == "preserved"
    assert weight.read_bytes() == b"preserved weights"
