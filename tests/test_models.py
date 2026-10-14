import hashlib
import json

import numpy as np
import pytest
from test_data import fixture_file

from proteinmod.benchmark import run
from proteinmod.models import fit_ridge, metrics, one_hot


def test_metrics():
    scores = metrics([1, 2, 3], [2, 3, 4])
    assert scores == pytest.approx({"spearman_rho": 1, "pearson_r": 1, "mae": 1, "rmse": 1})
    assert metrics([1, 2, 3], [2, 2, 2])["spearman_rho"] is None
    assert metrics([1, 1], [2, 3])["pearson_r"] is None
    assert metrics([0, 2], [0, 0])["rmse"] == pytest.approx(np.sqrt(2))


def test_full_one_hot():
    matrix = one_hot(["ACDE", "CCDF"])
    assert matrix.shape == (2, 80)
    assert matrix.sum(axis=1).tolist() == [[4], [4]]
    assert matrix[0, 0] == 1


def test_deterministic_validation_selection():
    x = np.array([[0], [1], [2], [3]])
    args = (x[:2], [0, 1], x[2:], [2, 3], [10, 0.01], 0)
    first, selection = fit_ridge(*args)
    second, selection2 = fit_ridge(*args)
    assert selection == selection2
    assert selection["alpha"] == 0.01
    np.testing.assert_array_equal(first.predict(x), second.predict(x))


def test_end_to_end_and_test_label_isolation(tmp_path):
    path = fixture_file(tmp_path)
    config = {
        "dataset": {"path": str(path), "source": "synthetic", "attribution": "fixture"},
        "reference": {
            "sequence": "ACDE",
            "source": "synthetic verified WT",
            "sha256": hashlib.sha256(b"ACDE").hexdigest(),
        },
        "run": {"output": str(tmp_path / "run"), "alphas": [0.01, 10], "seed": 0},
    }
    run(config)
    output = tmp_path / "run"
    original = json.loads((output / "metrics.json").read_text())["selection"]
    predictions = (output / "test_predictions.csv").read_text().splitlines()[1].split(",")[4:]
    fixture_file(
        tmp_path,
        [
            ["a", "ACDE", "1", "train", "False"],
            ["b", "CCDE", "2", "train", "True"],
            ["c", "ACDF", "999", "test", "False"],
        ],
    )
    run(config)
    assert json.loads((output / "metrics.json").read_text())["selection"] == original
    assert (output / "test_predictions.csv").read_text().splitlines()[1].split(",")[
        4:
    ] == predictions
    for artifact in ("qc.json", "manifest.json", "report.md"):
        assert (output / artifact).exists()
    del config["reference"]
    run(config)
    assert "mutation_count_ridge" in json.loads((output / "metrics.json").read_text())["skipped"]


@pytest.mark.parametrize("alphas", [[], [0], [-1], [float("nan")]])
def test_invalid_regularization(alphas):
    with pytest.raises(ValueError):
        fit_ridge(np.zeros((2, 1)), [0, 1], np.zeros((2, 1)), [0, 1], alphas, 0)
