import copy
import csv
import hashlib
import json

import numpy as np
import pytest
from test_embeddings import MockEncoder

from proteinmod.benchmark import run
from proteinmod.data import checksum
from proteinmod.embeddings import extract
from proteinmod.experiments import nested_subsets, predict_subset, run_experiments, summarize


def test_nested_subsets_deterministic_and_train_only():
    indices = np.arange(100, 200)
    first = nested_subsets(indices, [0.01, 0.05, 0.1, 0.25, 0.5, 1], 7)
    second = nested_subsets(indices, [0.01, 0.05, 0.1, 0.25, 0.5, 1], 7)
    previous = set()
    for fraction, values in first.items():
        np.testing.assert_array_equal(values, second[fraction])
        assert previous <= set(values) <= set(indices)
        assert len(values) >= 2
        previous = set(values)
    assert len(first[1]) == 100
    assert not np.array_equal(first[0.1], nested_subsets(indices, [0.1, 1], 8)[0.1])


@pytest.mark.parametrize("fractions", [[0, 1], [-1, 1], [1.1], [0.1], [1, 1]])
def test_invalid_fractions(fractions):
    with pytest.raises(ValueError):
        nested_subsets(np.arange(10), fractions, 0)


def test_constant_translation_and_test_label_isolation():
    x = np.random.default_rng(0).normal(size=(12, 3))
    features = {"mean": x, "delta": x - [10, 20, 30]}
    y = np.arange(12, dtype=float)
    args = (np.arange(6), np.arange(6, 9), np.arange(9, 12), [0.01, 1, 100], 0)
    first, selected = predict_subset(features, y, *args)
    changed_y = y.copy()
    changed_y[9:] = 999
    second, selected2 = predict_subset(features, changed_y, *args)
    assert selected == selected2
    for name, prediction in first.items():
        np.testing.assert_array_equal(prediction, second[name])
    np.testing.assert_allclose(first["mean"], first["delta"], atol=1e-10)


def test_seed_sd_is_train_subset_variability():
    records = [
        {
            "model": "x",
            "fraction": 0.1,
            "n_train": 2,
            "test": {"spearman_rho": None, "pearson_r": None, "mae": x, "rmse": x},
        }
        for x in [1, 2, 3]
    ]
    row = summarize(records)[0]
    assert row["mae_mean"] == 2
    assert row["mae_std"] == 1
    assert row["spearman_rho_mean"] is None


def test_matched_pipeline_and_m0_preservation(tmp_path, embedding_config, monkeypatch):
    path = tmp_path / "data.csv"
    sequences = ["ACDE", "ACDF", "ACDA", "ACAY", "CCDE", "ADDE", "AEDE", "ACYE", "ACAE"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "target", "set", "validation"])
        for i, sequence in enumerate(sequences):
            writer.writerow([sequence, i / 10, "train" if i < 7 else "test", i in (5, 6)])
    config = copy.deepcopy(embedding_config)
    config.update(
        {
            "dataset": {"path": str(path), "source": "synthetic", "attribution": "fixture"},
            "reference": {
                "sequence": "ACDE",
                "source": "synthetic WT",
                "sha256": hashlib.sha256(b"ACDE").hexdigest(),
            },
            "run": {"output": str(tmp_path / "m0"), "alphas": [0.01, 1], "seed": 0},
            "learning_curve": {
                "output": str(tmp_path / "m1"),
                "historical": str(tmp_path / "m0"),
                "fractions": [0.5, 1],
                "seeds": [0, 1, 2],
            },
        }
    )
    run(config)
    historical = tmp_path / "m0"
    before = {p.name: checksum(p) for p in historical.iterdir()}
    extract(config, sequences, "ACDE", checksum(path), encoder=MockEncoder())
    monkeypatch.setattr("proteinmod.experiments.plot_curves", lambda *args: None)
    scores = run_experiments(config)
    assert len(scores) == 6
    assert before == {p.name: checksum(p) for p in historical.iterdir()}
    output = tmp_path / "m1"
    stored = json.loads((output / "metrics.json").read_text())
    assert stored["m0_reproduction"]["passed"]
    with (output / "test_predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 2
    assert [row["sequence"] for row in predictions] == sequences[-2:]
    assert len(json.loads((output / "learning_curve_runs.json").read_text())) == 36
    config["learning_curve"]["output"] = str(historical)
    with pytest.raises(ValueError, match="overwrite"):
        run_experiments(config)
