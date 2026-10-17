"""Offline biochemical masks and invariant edge controls; no downloads."""

import numpy as np
import pytest

from proteinmod.edge_chemistry import EDGE_NAMES, annotate, controlled_graph, interaction


def test_atom_candidates_thresholds_and_missing_masks():
    a = {"NZ": np.array([0.0, 0, 0])}
    b = {"OD1": np.array([4.0, 0, 0]), "OD2": np.array([5.0, 0, 0])}
    value = interaction(a, b, "K", "D")
    assert value[0] == value[3] == 1
    assert interaction(a, {k: v + 0.01 for k, v in b.items()}, "K", "D")[0] == 0
    missing = interaction(a, {"OD1": b["OD1"]}, "K", "D")
    assert missing[0] == missing[3] == 0
    assert interaction(a, b, "H", "D")[0] == 0
    carbon_a, carbon_b = {"CB": np.zeros(3)}, {"CB": np.array([4.5, 0, 0])}
    assert interaction(carbon_a, carbon_b, "A", "A")[1] == 1
    assert interaction(carbon_a, {}, "A", "A")[4] == 0


def test_polar_candidate_requires_carbonyl_geometry_and_not_ca():
    a = {"C": np.array([-1.0, 0, 0]), "O": np.zeros(3), "N": np.array([-2.0, 0, 0])}
    b = {"N": np.array([3.0, 0, 0]), "C": np.array([4.0, 0, 0]), "O": np.array([5.0, 0, 0])}
    assert interaction(a, b, "A", "A")[2] == 1
    assert interaction(a, b, "A", "P")[2] == 0
    assert interaction(a, b, "A", "A", sequential=True)[2] == 0
    assert interaction({"CA": np.zeros(3)}, {"CA": np.ones(3)}, "A", "A")[5] == 0
    q = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    moved = [{k: v @ q + 123 for k, v in atoms.items()} for atoms in (a, b)]
    np.testing.assert_array_equal(interaction(a, b, "A", "A"), interaction(*moved, "A", "A"))


def tiny_graph():
    return {
        "nodes": [{"reference_aa": a, "polymer_position": i + 1} for i, a in enumerate("AKDE")],
        "static": np.zeros((4, 34), np.float32),
        "edge_index": np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        "edge_features": np.tile([3.8 / 8, 1.0, 1.0], (6, 1)).astype(np.float32),
    }


def test_controls_symmetric_capacity_and_topology():
    g = tiny_graph()
    x = annotate(g, [{} for _ in g["nodes"]])
    assert x.shape == (6, len(EDGE_NAMES))
    assert np.isfinite(x).all()
    for mode in (
        "edge_full",
        "edge_distance",
        "edge_no_chemistry",
        "edge_shuffled_chemistry",
        "edge_shuffled_topology",
        "zero_message",
    ):
        edges, values = controlled_graph(g, x, mode, 9)
        assert values.shape == x.shape
        np.testing.assert_array_equal(values[::2], values[1::2])
        assert sorted(np.bincount(edges[1], minlength=4)) == [1, 1, 2, 2]
        e2, v2 = controlled_graph(g, x, mode, 9)
        np.testing.assert_array_equal(edges, e2)
        np.testing.assert_array_equal(values, v2)
    with pytest.raises(ValueError, match="Unknown"):
        controlled_graph(g, x, "bad", 9)


def test_reporting_keeps_native_units_and_rank_metrics_separate():
    from proteinmod.edge_report import summarize

    rows = [
        {
            "cohort": cohort,
            "protocol": protocol,
            "model": "ridge",
            "seed": "0",
            "protein_id": "p",
            "family": "f",
            "n_mutations": "1",
            "target": str(i),
            "prediction": str(i + offset),
        }
        for cohort, protocol, offset in [("family", "G3", 50), ("native", "S2", 1)]
        for i in range(4)
    ]
    result = summarize(rows)
    rank = next(r for r in result["summary"] if r["cohort"] == "family")
    native = next(r for r in result["summary"] if r["cohort"] == "native")
    assert rank["macro_spearman"] == 1
    assert "macro_rmse_kcal_mol" not in rank
    assert native["macro_rmse_kcal_mol"] == 1
    assert next(r for r in result["per_protein"] if r["cohort"] == "family")["rmse"] is None
