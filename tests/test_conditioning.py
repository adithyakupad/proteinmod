"""Leakage, controls, physical units, frozen serving inputs and mutation-set invariance."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.conditioning_experiments import history, infer_stability, output_path
from proteinmod.conditioning_features import mutation_features
from proteinmod.stability_conditioning import (
    ConditionedHead,
    controls,
    fit_stability,
    neural_function,
    ridge_function,
    scalar_inputs,
)
from proteinmod.stability_data import ddg_target


def fixture():
    rows = [
        {
            "id": str(i),
            "protein_id": "p" + str(i // 4),
            "assay_id": "a",
            "target": float(i % 4),
            "task_type": "ddg",
            "target_unit": "kcal/mol",
            "target_convention": "positive_destabilizing",
            "target_mask": True,
            "conditions": {"pH": 7.4},
        }
        for i in range(12)
    ]
    roles = {"train": np.arange(6), "validation": np.arange(6, 9), "test": np.arange(9, 12)}
    x = np.random.default_rng(0).normal(size=(12, 5))
    return rows, roles, x


def test_stability_source_is_native_and_independent_of_source_test_labels(tmp_path):
    rows, roles, x = fixture()
    a, model, selected = fit_stability(x, rows, roles, [0.1, 10])
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999
    b, second, other = fit_stability(x, changed, roles, [0.1, 10])
    np.testing.assert_array_equal(model.predict(a.transform(x)), second.predict(b.transform(x)))
    assert selected == other
    assert ddg_target(-4.184, "kJ/mol", "positive_stabilizing") == pytest.approx(1)
    z, evidence = infer_stability(x, x[:4], rows, roles, [0.1, 10], {"test": "fixture"}, tmp_path)
    zz, _ = infer_stability(x, x[:4], changed, roles, [0.1, 10], {"test": "fixture"}, tmp_path)
    np.testing.assert_array_equal(z, zz)
    assert evidence["weights"]["unit"] == "kcal/mol"
    assert evidence["weights"]["convention"] == "positive_destabilizing"
    assert evidence["ood"]["confidence_available"] is False
    cache = next(tmp_path.rglob("predictions.npy"))
    cache.write_bytes(b"broken")
    with pytest.raises(ValueError, match="provenance"):
        infer_stability(x, x[:4], rows, roles, [0.1, 10], {"test": "fixture"}, tmp_path)


def test_controls_are_label_blind_role_local_and_match_training_scale():
    rows, roles, _ = fixture()
    z = np.arange(12, dtype=float)
    result = controls(z, rows, roles, 0)
    for ix in roles.values():
        for p in {rows[i]["protein_id"] for i in ix}:
            local = [i for i in ix if rows[i]["protein_id"] == p]
            assert set(result["shuffled"][local]) == set(z[local])
    assert result["dummy"][roles["train"]].mean() == pytest.approx(z[roles["train"]].mean())
    assert result["dummy"][roles["train"]].std() == pytest.approx(z[roles["train"]].std())
    changed = copy.deepcopy(rows)
    for r in changed:
        r["target"] = -999
    other = controls(z, changed, roles, 0)
    for key in result:
        np.testing.assert_array_equal(result[key], other[key])
    constant = controls(np.ones(12), rows, roles, 0)
    assert (constant["dummy"] == 1).all()


def test_scalar_missingness_and_train_only_calibration():
    z = np.array([1.0, 2.0, 3.0, np.nan, 9999])
    x, calibration = scalar_inputs(z, np.array([0, 1, 2]))
    assert calibration["mean_kcal_mol"] == 2
    assert x[3].tolist() == [0, 0]
    y, other = scalar_inputs(z * np.array([1, 1, 1, 1, 100]), np.array([0, 1, 2]))
    assert other == calibration
    np.testing.assert_array_equal(x[:4], y[:4])
    with pytest.raises(ValueError, match="Nonfinite"):
        scalar_inputs(z, np.array([0, 1]), np.ones(5, dtype=bool))
    with pytest.raises(ValueError, match="no training"):
        scalar_inputs(z, np.array([3]))


def test_function_fits_and_residuals_ignore_test_labels_and_reproduce():
    rows, roles, h = fixture()
    scalar, _ = scalar_inputs(np.arange(12.0), roles["train"])
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999
    for base in (None, np.arange(12.0) / 10):
        p, a = ridge_function(np.column_stack((h, scalar)), rows, roles, [0.1, 10], base)
        q, b = ridge_function(np.column_stack((h, scalar)), changed, roles, [0.1, 10], base)
        np.testing.assert_array_equal(p, q)
        assert a == b
    settings = {
        "cpu_threads": 1,
        "hidden": 4,
        "max_epochs": 4,
        "patience": 3,
        "batch_size": 3,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
    }
    p, a, state = neural_function(h, scalar, rows, roles, settings, 0)
    q, b, second = neural_function(h, scalar, changed, roles, settings, 0)
    np.testing.assert_array_equal(p, q)
    assert a == b and all(torch.equal(v, second["head"][k]) for k, v in state["head"].items())
    model = ConditionedHead(h.shape[1], 4)
    model.load_state_dict(state["head"])
    x = (h - state["representation_mean"]) / state["representation_scale"]
    with torch.no_grad():
        pred = (
            model(torch.tensor(x, dtype=torch.float32), torch.tensor(scalar, dtype=torch.float32))
            .double()
            .numpy()
        )
    np.testing.assert_array_equal(pred * state["target_scale"] + state["target_mean"], p)
    assert state["stability_input_unit"] == "kcal/mol"
    # The trainable object contains only the new head, never an encoder or stability model.
    assert all(name.startswith("head.") for name, _ in model.named_parameters())


def test_mutation_feature_order_wt_and_historical_guard(monkeypatch, tmp_path):
    import proteinmod.conditioning_features as module

    rows = [{"mutations": ("A1V", "E5F")}, {"mutations": ()}]
    values = np.random.default_rng(0).normal(size=(2, 5, 480)).astype(np.float32)
    group = {
        "p": {
            "indices": np.arange(2),
            "token_indices": np.arange(2),
            "residues": values,
            "graph": {"audit": {}},
        }
    }
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        module, "residue_cache", lambda *args: (np.zeros((1, 5, 480)), {"encoder": {}}, tmp_path)
    )
    monkeypatch.setattr(module, "variant_inputs", lambda *args: (None, np.zeros(270), []))
    spec = [{"protein_id": "p", "reference": {"sequence": "ACDEE"}, "dataset": {}}]
    a, _ = mutation_features(rows, group, spec, {}, tmp_path)
    rows[0]["mutations"] = tuple(reversed(rows[0]["mutations"]))
    b, _ = mutation_features(rows, group, spec, {}, tmp_path)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["structural_ridge"][1, -2] == 0
    assert a["esm_sites_ridge"][1, -1] == 0
    assert len(history()) >= 176
    with pytest.raises(ValueError, match="historical"):
        output_path("benchmarks/m7/subdirectory")
