"""Synthetic-only optimization, replay, leakage and statistical contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from proteinmod import synthetic_regression as diagnostic

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/m221_synthetic.json").read_text())


def toy_data():
    rng = np.random.default_rng(801)
    return rng.normal(size=(16, 16)), rng.normal(size=16)


def test_analytic_gradient_matches_finite_differences():
    x, y = toy_data()
    parameters = diagnostic.initialize(CONFIG, 2)
    _, gradients = diagnostic.loss_gradient(parameters, x, y)
    for name, index in [("w1", (0, 0)), ("b1", (0,)), ("w2", (0,)), ("b2", (0,))]:
        original = parameters[name][index]
        epsilon = 1e-6
        parameters[name][index] = original + epsilon
        plus = diagnostic.loss_gradient(parameters, x, y)[0]
        parameters[name][index] = original - epsilon
        minus = diagnostic.loss_gradient(parameters, x, y)[0]
        parameters[name][index] = original
        assert gradients[name][index] == pytest.approx((plus - minus) / (2 * epsilon), abs=1e-8)


def test_data_replay_and_split_independence():
    data = diagnostic.generate(CONFIG, CONFIG["scenarios"][0], 0)
    repeated = diagnostic.generate(CONFIG, CONFIG["scenarios"][0], 0)
    for name in data:
        for first, second in zip(data[name], repeated[name], strict=True):
            np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(data["train"][0], data["test"][0])
    assert not np.array_equal(data["validation"][0], data["test"][0][:256])
    assert len(data["train"][0]) == 512


def test_full_prefix_disk_and_optimizer_resume_replay(tmp_path):
    x, y = toy_data()
    full = diagnostic.fit(CONFIG, x, y, 0)
    repeated = diagnostic.fit(CONFIG, x, y, 0)
    prefix = diagnostic.fit(CONFIG, x, y, 0, stop=12)[12]
    for epoch in [12, 120]:
        assert diagnostic.snapshot_digest(full[epoch]) == diagnostic.snapshot_digest(
            repeated[epoch]
        )
    assert diagnostic.snapshot_digest(prefix) == diagnostic.snapshot_digest(full[12])
    saved = tmp_path / "state.npz"
    diagnostic.save_checkpoint(saved, prefix)
    restored = diagnostic.load_checkpoint(saved)
    assert diagnostic.snapshot_digest(restored) == diagnostic.snapshot_digest(prefix)
    resumed = diagnostic.fit(CONFIG, x, y, 0, resume=restored)[120]
    assert diagnostic.snapshot_digest(resumed) == diagnostic.snapshot_digest(full[120])
    with pytest.raises(FileExistsError):
        diagnostic.save_checkpoint(saved, prefix)


def test_validation_and_test_changes_cannot_affect_training_or_scaler():
    data = diagnostic.generate(CONFIG, CONFIG["scenarios"][1], 1)
    states = diagnostic.fit(CONFIG, *data["train"], 1)
    changed = copy.deepcopy(data)
    for name in ["validation", "test"]:
        changed[name] = (changed[name][0] * 1e6 + 3, -changed[name][1] * 7)
        before = diagnostic.snapshot_digest(states[12])
        diagnostic.metrics(states[12], *changed[name])
        assert diagnostic.snapshot_digest(states[12]) == before
    actual = diagnostic.fit(CONFIG, *changed["train"], 1)
    for epoch in [12, 120]:
        assert diagnostic.snapshot_digest(actual[epoch]) == diagnostic.snapshot_digest(
            states[epoch]
        )
    np.testing.assert_array_equal(states[12]["mean"], data["train"][0].mean(axis=0))
    np.testing.assert_array_equal(states[12]["scale"], data["train"][0].std(axis=0))


def test_resume_rejects_changed_training_data_or_settings():
    x, y = toy_data()
    prefix = diagnostic.fit(CONFIG, x, y, 0, stop=12)[12]
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnostic.fit(CONFIG, x, y + 1, 0, resume=prefix)
    config = copy.deepcopy(CONFIG)
    config["optimizer"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnostic.fit(config, x, y, 0, resume=prefix)
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnostic.fit(CONFIG, x, y, 1, resume=prefix)


def test_metrics_use_fixed_state_and_undefined_ranks_remain_null():
    x, y = toy_data()
    state = diagnostic.fit(CONFIG, x, y, 0, stop=12)[12]
    prediction = diagnostic.predict(state["parameters"], (x - state["mean"]) / state["scale"])
    observed = diagnostic.metrics(state, x, y)
    assert observed["mse"] == float(np.mean((prediction - y) ** 2))
    assert diagnostic.metrics(state, x, np.ones(len(x)))["spearman"] is None
    assert diagnostic.metrics(state, x, prediction)["spearman"] == pytest.approx(1)


def test_paired_uncertainty_uses_seed_differences():
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    stats = diagnostic.paired_statistics(values)
    assert stats["mean"] == 0
    assert stats["sample_sd"] == pytest.approx(np.sqrt(2.5))
    assert stats["standard_error"] == pytest.approx(np.sqrt(0.5))
    assert stats["ci95"][1] == pytest.approx(1.9632431615)
    assert (stats["negative"], stats["positive"], stats["zero"]) == (2, 2, 1)
    with pytest.raises(ValueError):
        diagnostic.paired_statistics([0])


def test_rejects_nonfinite_training_without_silent_seed_replacement():
    x, y = toy_data()
    y[0] = np.nan
    with pytest.raises(ValueError, match="Finite synthetic"):
        diagnostic.fit(CONFIG, x, y, 0)


def test_synthetic_ci_scope_fails_closed_on_old_or_unrelated_changes():
    import runpy

    routing = runpy.run_path(str(ROOT / "conftest.py"))
    validate = routing["validate_synthetic_changes"]
    validate(["A\tsrc/proteinmod/synthetic_regression.py", "A\tbenchmarks/m221/REPORT.md"])
    for changes in [["M\tsrc/proteinmod/models.py"], ["A\tunrelated.py"], []]:
        with pytest.raises(ValueError):
            validate(changes)
