"""Measured factorial provenance, symmetric interactions, leakage and sum-only models."""

import copy
import csv
import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.data import checksum
from proteinmod.epistasis_data import additive, collection, protocols
from proteinmod.epistasis_experiments import (
    PairResidual,
    contrast_predictions,
    historical,
    neural_fit,
    output_path,
    ridge_fit,
)
from proteinmod.mutation_interactions import (
    PAIR_WIDTH,
    SEQUENCE_WIDTH,
    geometry,
    pair_features,
    topology,
)


def landscape(tmp_path):
    records = []
    for alleles in itertools.product("AC", repeat=4):
        g = "".join(alleles)
        n = g.count("C")
        selected = 0 if n == 4 else int(10000 / 2**n)
        records.append(
            {
                "Variants": g,
                "HD": n,
                "Count input": 10000,
                "Count selected": selected,
                "Fitness": 999 if n == 1 else selected / 10000,
                "sequence": g,
            }
        )
    path = tmp_path / "source.csv"
    with path.open("w") as h:
        writer = csv.DictWriter(h, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    config = {
        "dataset": {"path": str(path), "sha256": checksum(path)},
        "reference": {"sequence": "AAAA"},
        "full_reference": {"sequence": "AAAA"},
        "positions": [1, 2, 3, 4],
        "expected_source_rows": 16,
        "fitness_floor": 0.01,
        "caps": {str(n): None for n in range(5)},
        "split_salt": "fixture",
        "test_position_pair": [1, 4],
        "validation_position_pair": [2, 4],
    }
    return config


def graph():
    edges = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return {
        "nodes": [{"plddt": None, "bfactor_a2": 20 + 10 * i} for i in range(4)],
        "lookup": {i + 1: i for i in range(4)},
        "edge_index": edges,
        "xyz": np.array([[3.8 * i, 0, 0] for i in range(4)]),
        "audit": {"confidence_semantics": "experimental_bfactor"},
    }


def test_counts_are_same_assay_and_published_single_column_is_not_label(tmp_path):
    config = landscape(tmp_path)
    rows, contrasts, audit = collection(config)
    single = next(r for r in rows if r["id"] == "CAAA")
    assert single["target"] == pytest.approx(np.log(0.5)) and single["published_fitness"] == 999
    assert audit["contrast_exclusions"]["censored_constituent"] > 0
    assert not any(c["target"] == "CCCC" for c in contrasts)
    assert next(r["target"] for r in rows if r["id"] == "CCCC") == pytest.approx(np.log(0.01))
    assert all(r["cohort"] for r in rows)


def test_additive_expectation_and_missing_single_policy_use_training_only(tmp_path):
    c = landscape(tmp_path)
    rows, contrasts, _ = collection(c)
    roles = dict(protocols(rows, c))["C1"]
    pred, state = additive(rows, roles["train"])
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    assert pred[lookup["CCAA"]] == pytest.approx(np.log(0.5) * 2)
    assert max(state["missing_single_counts"]) == 0
    train = [i for i in roles["train"] if rows[i]["id"] != "CAAA"]
    _, missing = additive(rows, train)
    assert missing["missing_single_counts"][lookup["CCAA"]] == 1
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999
    np.testing.assert_array_equal(pred, additive(changed, roles["train"])[0])
    epsilon = contrast_predictions(rows, contrasts, pred, roles["test"])
    assert all(r["predicted_log_epistasis"] == 0 for r in epsilon)
    assert any(
        abs(r["measured_linear_epistasis"] - r["measured_log_epistasis"]) > 0.01 for r in epsilon
    )


def test_protocol_unseen_pair_cannot_occur_on_any_training_background(tmp_path):
    c = landscape(tmp_path)
    rows, _, _ = collection(c)
    for protocol, roles in protocols(rows, c):
        assert not set(roles["train"]) & set(roles["test"])
        assert all(i in roles["train"] for i, r in enumerate(rows) if len(r["mutations"]) < 2)
        if protocol == "C2":
            assert all(
                not {"A1C", "A4C"} <= set(rows[i]["mutations"])
                for p in ("train", "validation")
                for i in roles[p]
            )
            assert all({"A1C", "A4C"} <= set(rows[i]["mutations"]) for i in roles["test"])


def test_pair_features_symmetry_cardinality_masks_units_and_controls():
    g = graph()
    tokens = ("A1C", "A3D", "A4E")
    a = pair_features(tokens, "AAAA", g, [1, 2, 3, 4])
    b = pair_features(tokens[::-1], "AAAA", g, [1, 2, 3, 4])
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, PAIR_WIDTH)
    assert pair_features(("A1C",), "AAAA", g, [1, 2, 3, 4]).shape == (0, PAIR_WIDTH)
    zero = pair_features(tokens, "AAAA", g, [1, 2, 3, 4], "zero3d")
    np.testing.assert_array_equal(a[:, :SEQUENCE_WIDTH], zero[:, :SEQUENCE_WIDTH])
    assert (zero[:, SEQUENCE_WIDTH:] == 0).all()
    shuffled = pair_features(tokens, "AAAA", g, [1, 2, 3, 4], "shuffled_distance", 1)
    assert not np.array_equal(shuffled[:, SEQUENCE_WIDTH:], a[:, SEQUENCE_WIDTH:])
    np.testing.assert_array_equal(
        shuffled, pair_features(tokens, "AAAA", g, [1, 2, 3, 4], "shuffled_distance", 1)
    )
    bio = pair_features(tokens, "AAAA", g, [1, 2, 3, 4], "shuffled_biochemical", 1)
    assert not np.array_equal(bio[:, :SEQUENCE_WIDTH], a[:, :SEQUENCE_WIDTH])
    values = geometry(1, 2, g, topology(g))
    assert values[0] == pytest.approx(3.8 / 30) and values[8] == values[9] == 0
    assert values[10] == pytest.approx(0.25) and values[11] == 1
    missing = copy.deepcopy(g)
    del missing["lookup"][2]
    out = geometry(1, 2, missing, topology(missing))
    assert out[0] == out[6] == out[7] == 0
    disconnected = copy.deepcopy(g)
    disconnected["edge_index"] = np.empty((2, 0), int)
    out = geometry(1, 2, disconnected, topology(disconnected))
    assert out[4] == out[5] == out[7] == 0
    wrong = copy.deepcopy(g)
    wrong["nodes"][0]["plddt"] = 50
    with pytest.raises(ValueError, match="typed"):
        geometry(1, 2, wrong, topology(wrong))


def test_sum_pair_model_empty_padding_gradients_invariance_and_reload(tmp_path):
    torch.manual_seed(0)
    model = PairResidual(7, 4)
    x = torch.randn(3, 6, 7)
    mask = torch.tensor([[False] * 6, [True] + [False] * 5, [True] * 6])
    x[0] = float("nan")
    prediction = model(x, mask)
    assert prediction[0] == 0 and torch.isfinite(prediction).all()
    torch.testing.assert_close(prediction, model(x.flip(1), mask.flip(1)))
    prediction.sum().backward()
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
    path = tmp_path / "head.pt"
    torch.save(model.state_dict(), path)
    other = PairResidual(7, 4)
    other.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(prediction, other(x, mask), rtol=0, atol=0)
    with pytest.raises(ValueError, match="mask"):
        model(x, mask.float())


def test_fits_ignore_test_labels_and_pair_ridge_preserves_anchor_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5))
    x[:2] = 0
    y = np.arange(12, dtype=float) / 10
    base = np.linspace(0, 1, 12)
    roles = {"train": np.arange(6), "validation": np.arange(6, 9), "test": np.arange(9, 12)}
    changed = y.copy()
    changed[roles["test"]] = -999
    a, sel, _ = ridge_fit(x, y, roles, [0.1, 10], base, False)
    b, other, _ = ridge_fit(x, changed, roles, [0.1, 10], base, False)
    np.testing.assert_array_equal(a, b)
    assert sel == other
    np.testing.assert_array_equal(a[:2], base[:2])
    pairs = rng.normal(size=(12, 3, 5)).astype(np.float32)
    mask = np.ones((12, 3), bool)
    mask[:2] = False
    settings = {
        "device": "cpu",
        "cpu_threads": 1,
        "hidden": 4,
        "scale_floor": 1e-6,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "max_epochs": 3,
        "patience": 2,
        "batch_size": 3,
    }
    p, s, _ = neural_fit(pairs, mask, y, base, roles, settings, 0)
    q, t, _ = neural_fit(pairs, mask, changed, base, roles, settings, 0)
    np.testing.assert_array_equal(p, q)
    assert s == t
    np.testing.assert_array_equal(p[:2], base[:2])


def test_historical_hash_guards_and_m10_forward_replay(tmp_path, monkeypatch):
    h = historical()
    assert sum(p.startswith("benchmarks/") for p in h) == 238
    assert "configs/m10.json" in h and "configs/m11.json" not in h
    for path in ("benchmarks", "benchmarks/m10/destroy"):
        with pytest.raises(ValueError, match="historical"):
            output_path(path)
    from proteinmod.reliability_experiments import history as m10_history

    for name in ("m9", "m10", "m11"):
        root = tmp_path / "benchmarks" / name
        root.mkdir(parents=True)
        (root / "evidence").write_text("x")
    root = tmp_path / "configs"
    root.mkdir()
    (root / "m9.json").write_text("{}")
    (root / "m11.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert set(m10_history()) == {"benchmarks/m9/evidence", "configs/m9.json"}


def test_archived_m10_cache_root_is_pinned_and_checksum_validated(tmp_path):
    import json

    from proteinmod.reliability_experiments import load_features

    path = tmp_path / "features.npz"
    np.savez_compressed(path, h=np.ones((2, 100)))
    (tmp_path / "manifest.json").write_text(json.dumps({"feature_sha256": checksum(path)}))
    values, _ = load_features({}, tmp_path)
    assert values["h"].shape == (2, 100)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_features({}, tmp_path)


def test_block_bootstrap_is_paired_deterministic_and_refuses_one_block():
    pytest.importorskip("matplotlib")
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/report_m11.py"
    spec = importlib.util.spec_from_file_location("report_m11", path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    y = np.arange(6.0)
    p = y + 0.2
    groups = ["a", "a", "b", "b", "c", "c"]
    settings = {"seed": 0, "resamples": 20}
    equal = report.paired_block_interval(y, p, p, groups, "mae", settings)
    assert equal["low"] == equal["high"] == 0
    a = report.paired_block_interval(y, y, p, groups, "mae", settings)
    assert a == report.paired_block_interval(y, y, p, groups, "mae", settings)
    assert a["low"] == pytest.approx(0.2) and a["high"] == pytest.approx(0.2)
    assert report.paired_block_interval(y, y, p, ["one"] * 6, "mae", settings)["low"] is None
