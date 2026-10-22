"""Focused synthetic geometry, selection and identity contracts; no biological inputs."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from proteinmod import geometry_study as g


def test_registered_pairing_and_covariance_identity():
    cfg = g.config()
    data = g.generate(0, cfg)
    repeat = g.generate(0, cfg)
    for scenario in data:
        for split in data[scenario]:
            for a, b in zip(data[scenario][split], repeat[scenario][split], strict=True):
                np.testing.assert_array_equal(a, b)
    for split in ("train", "validation", "test"):
        np.testing.assert_array_equal(data["A"][split][1], data["B"][split][1])
    for a, b in zip(data["B"]["train"], data["C"]["train"], strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(data["B"]["test"][0], data["C"]["test"][0])
    assert np.linalg.eigvalsh(g.correlation(12, -0.6)).min() > 0


def test_svd_scalar_and_diagonal_equivalence():
    cfg = g.config()
    x, y = g.generate(1, cfg)["B"]["train"]
    penalties = np.power(10.0, cfg["lambda_exponents"])
    states = {m: [g.ridge(x, y, m, float(l)) for l in penalties] for m in g.METHODS if m != "mlp"}
    maximum = g.numerical_controls(states, x, y, penalties, cfg)
    assert maximum < 1e-8


def test_training_only_preprocessing_and_heldout_label_noninterference():
    cfg = g.config()
    data = g.generate(2, cfg)["B"]
    x, y = data["train"]
    first = g.fit(x, y, "mlp", 0.1, 2, cfg)
    mean, sd, _ = g.preprocessing(x)
    for split in ("validation", "test"):
        hx, hy = data[split]
        hx[:] = 1e5
        hy[:] = -1e5
        g.metrics(hy, g.predict(first, hx))
    second = g.fit(*data["train"], "mlp", 0.1, 2, cfg)
    g.equal(first, second)
    np.testing.assert_array_equal(mean, np.asarray(first["mean"]))
    np.testing.assert_array_equal(sd, np.asarray(first["scale"]))
    assert first["step"] == 240


def test_deterministic_selection_nested_and_redundant_controls():
    cfg = g.config()
    large = np.power(10.0, cfg["lambda_exponents"])
    small = large[cfg["small_indices"]]
    np.testing.assert_allclose(small, [0.001, 0.1, 10, 1000], rtol=0, atol=1e-14)
    losses = [1.0] * 16
    losses[10] = 0.5
    assert (
        g.select(losses, cfg["small_indices"]) == g.select(losses, cfg["small_indices"] * 4) == 10
    )
    assert g.select([1.0] * 16, list(range(16))) == 0


def test_population_risk_and_recovery_are_distinct():
    cfg = g.config()
    scale = np.logspace(*cfg["scale_log10"], 12)
    truth = {"method": "unscaled", "coefficient": np.asarray(cfg["beta"]) / scale, "intercept": 0.0}
    assert g.population(truth, "C", "test", cfg) == pytest.approx(1.0)
    assert g.recovery(truth, "B", cfg) == pytest.approx(0.0)
    biased = copy.deepcopy(truth)
    biased["intercept"] = 2.0
    assert g.population(biased, "C", "test", cfg) == pytest.approx(5.0)
    assert g.recovery(biased, "B", cfg) == pytest.approx(0.0)


def test_exact_replay_rejects_changed_state_and_preserves_original():
    cfg = g.config()
    x, y = g.generate(3, cfg)["A"]["train"]
    state = g.fit(x, y, "mlp", 0.1, 3, cfg)
    changed = copy.deepcopy(state)
    changed["m"]["w1"][0, 0] += 1
    with pytest.raises(ValueError):
        g.equal(state, changed)
    assert state["m"]["w1"][0, 0] != changed["m"]["w1"][0, 0]


def test_geometry_and_independent_seed_statistics():
    x = np.tile(np.eye(12), (6, 1)) + np.random.default_rng(91).normal(0, 0.01, (72, 12))
    result = g.geometry(x)
    assert result["standardized"]["anisotropy"] == pytest.approx(1.0)
    assert 1 <= result["unscaled"]["effective_rank"] <= 12
    stats = g.paired([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert stats["n"] == 5 and stats["mean"] == 0
    assert stats["sd"] == pytest.approx(np.sqrt(2.5))
    assert g.wilson(0, 40)[0] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        g.paired([1.0])


def test_ci_scope_preserves_trainers_and_rejects_unrelated_changes():
    from proteinmod.m23_collection import validate_changes

    validate_changes(
        ["M\tsrc/proteinmod/m225_collection.py", "A\tsrc/proteinmod/geometry_study.py"]
    )
    for lines in ([], ["M\tsrc/proteinmod/synthetic_regression.py"], ["A\tunrelated.py"]):
        with pytest.raises(ValueError):
            validate_changes(lines)


def test_selection_does_not_accept_test_labels():
    spec = importlib.util.spec_from_file_location(
        "m23_runner", Path(__file__).parents[1] / "scripts/run_m23.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = g.config()
    x, y = g.generate(4, cfg)["A"]["train"]
    models = [
        g.ridge(x, y, "standardized", float(l)) for l in np.power(10.0, cfg["lambda_exponents"])
    ]
    data = g.generate(4, cfg)["A"]
    # Direct selection depends only on independently supplied validation losses.
    losses = [
        g.metrics(data["validation"][1][:16], g.predict(s, data["validation"][0])[:16])["mse"]
        for s in models
    ]
    before = g.select(losses, list(range(16)))
    data["test"][1][:] = 1e9
    assert g.select(losses, list(range(16))) == before
