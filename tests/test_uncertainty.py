"""Offline uncertainty contracts, calibration quantiles, units and leakage controls."""

import copy
import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from proteinmod.data import checksum
from proteinmod.uncertainty import (
    bootstrap_indices,
    conformal_quantile,
    ensemble_summary,
    interval,
    interval_metrics,
    ordinal_error,
    selective_risk,
    shuffled,
    training_distance,
)
from proteinmod.uncertainty_calibration import calibration_state, prediction_intervals
from proteinmod.uncertainty_data import native_labels, roles_audit, uq_folds
from proteinmod.uncertainty_models import native_affine, ridge_ensemble, ridge_predict


def fixture():
    rows = [
        {
            "id": f"{f}:{p}:{j}",
            "family": f,
            "protein_id": f"{f}:{p}",
            "assay_id": f"{f}:{p}",
            "sequence": f"{f}:{p}:{j}",
            "target": float(j),
            "partition": "train" if j < 16 else "validation" if j < 20 else "test",
        }
        for f in ("a", "b", "c", "d")
        for p in (0, 1)
        for j in range(24)
    ]
    c = {
        "calibration_salt": "fixed",
        "g1_calibration_native_per_protein": 4,
        "native_scale_floor_kcal_mol": 0.05,
        "distance_floor": 0.05,
        "nominal_levels": [0.8, 0.9],
        "methods": [
            "validation_residual",
            "calibration_residual",
            "ensemble_scaled",
            "distance_scaled",
            "shuffled_scaled",
            "ensemble_normal",
        ],
    }
    return rows, np.array([-r["target"] for r in rows]), c


def test_four_roles_and_entire_calibration_family():
    rows, native, c = fixture()
    plans = uq_folds(rows, native, {"family_order": ["a", "b", "c", "d"]}, c)
    assert len(plans) == 13
    for protocol, _, roles in plans:
        roles_audit(rows, roles, protocol)
        assert sum(map(len, roles.values())) == len(rows)
    roles = next(r for p, _, r in plans if p == "G3")
    changed = copy.deepcopy(rows)
    changed[roles["calibration"][0]]["family"] = rows[roles["test"][0]]["family"]
    with pytest.raises(ValueError, match="family leakage"):
        roles_audit(changed, roles, "G3")
    bad = {**roles, "calibration": np.array([roles["test"][0]])}
    with pytest.raises(ValueError, match="leakage"):
        roles_audit(rows, bad, "G3")


def test_quantile_and_interval_score_reference_cases():
    assert conformal_quantile([1, 2, 3, 4], 0.8) == 4
    assert np.isinf(conformal_quantile([1, 2, 3, 4], 0.9))
    assert np.isinf(conformal_quantile([], 0.8))
    a = interval([1, 2], [0.5, 0.5], 2, "kcal/mol")
    assert a["lower"] == [0, 1] and a["upper"] == [2, 3]
    assert interval([1], [1], float("inf"), "kcal/mol")["lower"] == [None]
    values = interval_metrics([0, 3], [-1, -1], [1, 1], 0.8)
    assert values["coverage"] == 0.5 and values["mean_width"] == 2
    assert values["interval_score"] == pytest.approx(12)
    with pytest.raises(ValueError):
        interval_metrics([0], [1], [-1], 0.8)
    with pytest.raises(ValueError):
        conformal_quantile([float("nan")], 0.9)


def test_bootstrap_mask_shuffle_and_label_free_distance():
    rows, _, _ = fixture()
    train = np.arange(16)
    np.testing.assert_array_equal(
        bootstrap_indices(rows, train, 7), bootstrap_indices(rows, train, 7)
    )
    assert set(bootstrap_indices(rows, train, 7)) <= set(train)
    s = ensemble_summary([[1, 2, np.nan], [3, 4, np.nan]], np.array([True, True, False]))
    assert s["point_prediction"] == [2, 3, None] and s["aleatoric_variance"] is None
    assert s["uncertainty"][0] == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        ensemble_summary([[1], [np.inf]])
    x = np.arange(20).reshape(10, 2).astype(float)
    d = training_distance(x, [0, 1, 2])
    x[8:] += 1000
    again = training_distance(x, [0, 1, 2])
    np.testing.assert_allclose(d[:8], again[:8])
    result = shuffled(np.arange(8), ["a"] * 4 + ["b"] * 4, 3)
    assert set(result[:4]) == set(range(4)) and set(result[4:]) == set(range(4, 8))
    np.testing.assert_array_equal(result, shuffled(np.arange(8), ["a"] * 4 + ["b"] * 4, 3))


def test_calibration_ignores_test_labels_and_monotonic_intervals():
    rows, y, c = fixture()
    roles = next(
        r for p, _, r in uq_folds(rows, y, {"family_order": ["a", "b", "c", "d"]}, c) if p == "G3"
    )
    point = np.zeros(len(y))
    sd = np.linspace(0.1, 1, len(y))
    distance = np.ones(len(y))
    state = calibration_state(y, point, sd, distance, sd, rows, roles, c, "G3")
    y[roles["test"]] = 9999
    assert state == calibration_state(y, point, sd, distance, sd, rows, roles, c, "G3")
    assert state["family_block_audit"]["independent_calibration_families"] == 1
    assert state["family_block_audit"]["quantiles"] == {"0.8": None, "0.9": None}
    bounds = prediction_intervals(point, sd, distance, sd, rows, state, c)
    for method in c["methods"]:
        lo80, lo90 = (
            np.array(bounds[method + "_80"]["lower"]),
            np.array(bounds[method + "_90"]["lower"]),
        )
        hi80, hi90 = (
            np.array(bounds[method + "_80"]["upper"]),
            np.array(bounds[method + "_90"]["upper"]),
        )
        assert (lo90 <= lo80).all() and (hi90 >= hi80).all() and (lo80 <= hi80).all()


def test_ridge_and_native_head_no_test_or_calibration_label_leakage():
    rows, native, c = fixture()
    roles = next(
        r
        for p, _, r in uq_folds(rows, native, {"family_order": ["a", "b", "c", "d"]}, c)
        if p == "G3"
    )
    x = np.random.default_rng(4).normal(size=(len(rows), 5))
    pred, selection, state = ridge_ensemble(x, rows, roles, [0.1, 10], 4, 9)
    converted, head = native_affine(pred, native, roles["train"], 1e-8)
    changed = copy.deepcopy(rows)
    for i in np.r_[roles["test"], roles["calibration"]]:
        changed[i]["target"] = 10000 - i
        native[i] = 10000 - i
    again, chosen, _ = ridge_ensemble(x, changed, roles, [0.1, 10], 4, 9)
    np.testing.assert_array_equal(pred, again)
    assert selection == chosen
    other, metadata = native_affine(pred, native, roles["train"], 1e-8)
    np.testing.assert_array_equal(converted, other)
    assert head == metadata
    replay = ridge_predict(
        x, {k: np.asarray(json.loads(json.dumps(v.tolist()))) for k, v in state.items()}
    )
    np.testing.assert_array_equal(replay, pred)


def test_ordinal_error_selective_risk_and_permutation():
    np.testing.assert_array_equal(ordinal_error([1, 2, 3], [1, 2, 3]), [0, 0, 0])
    np.testing.assert_array_equal(ordinal_error([1, 2, 3], [3, 2, 1]), [1, 1, 1])
    np.testing.assert_array_equal(ordinal_error([1, 2, 3], [0, 0, 0]), [0.5, 0.5, 0.5])
    assert np.isnan(ordinal_error([1, 1], [0, 1])).all()
    a = selective_risk([1, 3, 2], [0.1, 0.9, 0.5], ["a", "b", "c"], [1 / 3, 1])
    assert a[0]["risk"] == 1 and a[1]["risk"] == 2
    order = [2, 0, 1]
    b = selective_risk(
        np.array([1, 3, 2])[order],
        np.array([0.1, 0.9, 0.5])[order],
        np.array(["a", "b", "c"])[order],
        [1 / 3, 1],
    )
    assert a == b


def test_primary_native_sign_units_and_multi_mutation_mask(tmp_path):
    reference = {
        "sequence": "AC",
        "sha256": hashlib.sha256(b"AC").hexdigest(),
        "source": "synthetic",
    }
    table = []
    for pos, wt in [(1, "A"), (2, "C")]:
        row = {"pdb_name": "1AAA.pdb", "pos": str(pos), "wt_aa": wt}
        row.update({"ddg_" + a: 0 if a == wt else 2.0 for a in "ACDEFGHIKLMNPQRSTVWY"})
        table.append(row)
    text = io.StringIO()
    w = csv.DictWriter(text, fieldnames=table[0])
    w.writeheader()
    w.writerows(table)
    raw = text.getvalue().encode()
    source = tmp_path / "measurements.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("data.csv", raw)
    m6 = {
        "sources": {"measurements": {"path": str(source), "sha256": checksum(source)}},
        "table_member": "data.csv",
        "table_sha256": hashlib.sha256(raw).hexdigest(),
        "assay": {
            "id": "native",
            "unit": "kcal/mol",
            "source_convention": "positive_stabilizing",
            "task_type": "ddg",
            "conditions": {"pH": 7.4},
        },
    }
    m12 = {"proteins": [{"protein_id": "p", "structure": {"id": "1AAA"}, "reference": reference}]}
    rows = [
        {"id": "single", "protein_id": "p", "sequence": "GC", "mutations": ("A1G",)},
        {"id": "double", "protein_id": "p", "sequence": "GD", "mutations": ("A1G", "C2D")},
    ]
    values, evidence = native_labels(rows, m12, m6)
    assert values[0] == -2 and np.isnan(values[1]) and evidence[1]["reason"].startswith("multiple")
    rows[1]["mutations"] = tuple(reversed(rows[1]["mutations"]))
    np.testing.assert_array_equal(values, native_labels(rows, m12, m6)[0])


def test_public_prediction_contract_serializes_masks_and_bounds():
    from proteinmod.uncertainty import predictive_result

    bounds = {"calibration_80": {"lower": [0, 0], "upper": [3, 3], "status": ["finite", "finite"]}}
    value = predictive_result(
        [[1, np.nan], [2, np.nan]],
        bounds,
        "kcal/mol",
        np.array([True, False]),
        {"target_convention": "positive_destabilizing"},
    )
    assert value["intervals"]["calibration_80"]["upper"] == [3, None]
    assert value["point_prediction"] == [1.5, None]
    assert json.loads(json.dumps(value, allow_nan=False)) == value
    with pytest.raises(ValueError):
        predictive_result([[1], [2]], bounds, "kcal/mol")


def test_public_result_rejects_reversed_or_half_missing_interval():
    from proteinmod.uncertainty import predictive_result

    for lower, upper in (([2], [1]), ([None], [1])):
        with pytest.raises(ValueError, match="ordered"):
            predictive_result([[1], [2]], {"bad": {"lower": lower, "upper": upper}}, "kcal/mol")


def test_saved_example_analysis_and_equal_family_aggregation():
    from proteinmod.uncertainty_report import analyze, macro

    predictions = []
    for j in range(3):
        row = {
            "dataset": "M12",
            "protocol": "G3",
            "fold": "f",
            "model": "test",
            "repeat": "0",
            "protein_id": "p",
            "family": "f",
            "id": str(j),
            "measured_native": str(3 * j) if j < 2 else "",
            "native_point": "0",
            "native_sd": str(0.1 + j),
            "native_sd_shuffled": str(2 - j),
            "distance_to_training": str(j + 1),
            "nearest_training_reference_identity": ".2",
            "structure_coverage": "1",
            "mutation_count": "1" if j < 2 else "2",
            "measured_original": str(j),
            "rank_point": str(j),
            "rank_sd": str(0.1 + j),
            "rank_sd_shuffled": str(2 - j),
            "deterministic_rank": str(j),
        }
        for level in (80, 90):
            row[f"calibration_residual_{level}_lower"] = "-1"
            row[f"calibration_residual_{level}_upper"] = "1"
        predictions.append(row)
    config = {
        "retained_fractions": [0.5, 1.0],
        "nominal_levels": [0.8, 0.9],
        "methods": ["calibration_residual"],
    }
    points, intervals, curves = analyze(predictions, config)
    assert points[0]["n_all"] == 3 and points[0]["n_native"] == 2
    assert points[0]["rank_spearman"] == 1 and points[0]["native_mae"] == 1.5
    assert intervals[0]["coverage"] == 0.5 and intervals[0]["interval_score"] == pytest.approx(12)
    assert any(
        r["risk"] == 0
        for r in curves
        if r["ranker"] == "ensemble" and r["risk_type"] == "native_mae"
    )
    result = macro(
        [
            {"family": "a", "protein_id": "a1", "metric": 0},
            {"family": "a", "protein_id": "a2", "metric": 0},
            {"family": "b", "protein_id": "b1", "metric": 1},
        ],
        (),
        ("metric",),
    )
    assert result[0]["metric"] == 0.5 and result[0]["replication_groups"] == 2
