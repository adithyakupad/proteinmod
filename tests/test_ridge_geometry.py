import copy

import numpy as np
import pytest

from proteinmod.ridge_geometry import fit, predict


def example():
    matrix = np.random.default_rng(1).normal(size=(30, 5))
    rows = [{"protein_id": "p", "assay_id": "a", "target": float(i)} for i in range(30)]
    roles = {"train": np.arange(16), "validation": np.arange(16, 23), "test": np.arange(23, 30)}
    return matrix, rows, roles


def test_rescaling_and_blinding():
    matrix, rows, roles = example()
    state, selection = fit(matrix, rows, roles, [0.1, 1], False, True)
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] *= -999
    other, selected = fit(matrix, changed, roles, [0.1, 1], False, True)
    assert selected == selection
    for key in state:
        np.testing.assert_array_equal(state[key], other[key])
    scaled, _ = fit(matrix * 7, rows, roles, [0.1, 1], False, True)
    np.testing.assert_allclose(predict(matrix, state), predict(matrix * 7, scaled), atol=1e-10)
    original, _ = fit(matrix, rows, roles, [0.1], False, False)
    rescaled, _ = fit(matrix * 7, rows, roles, [4.9], False, False)
    np.testing.assert_allclose(predict(matrix, original), predict(matrix * 7, rescaled), atol=1e-10)
    altered = matrix.copy()
    altered[roles["test"]] *= 1000
    unaffected, _ = fit(altered, rows, roles, [0.1, 1], False, True)
    for key in state:
        np.testing.assert_array_equal(state[key], unaffected[key])


def test_tie_and_invalid_energy(monkeypatch):
    matrix, rows, roles = example()
    monkeypatch.setattr("proteinmod.ridge_geometry.macro_score", lambda *args: 0.0)
    _, selection = fit(matrix, rows, roles, [0.1, 1], True, True)
    assert selection["penalty"] == 0.1
    with pytest.raises(ValueError, match="energy"):
        fit(np.ones_like(matrix), rows, roles, [1], False, True)


def test_float32_historical_replay():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from proteinmod.multiprotein_experiments import fit_targets

    matrix, rows, roles = example()
    matrix = matrix.astype(np.float32)
    state, _ = fit(matrix, rows, roles, [0.1], True, False)
    scaler = StandardScaler().fit(matrix[roles["train"]])
    features = scaler.transform(matrix).astype(float)
    target = fit_targets(rows, roles["train"])
    model = Ridge(alpha=0.1, solver="lsqr", tol=1e-8, max_iter=10000)
    model.fit(features[roles["train"]], target[roles["train"]])
    np.testing.assert_array_equal(predict(matrix, state), model.predict(features))
