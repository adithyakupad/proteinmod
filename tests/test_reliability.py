"""Confidence semantics, input masking, fallback, invariance and supervised isolation."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from proteinmod.confidence_fusion import MODES, ReliabilityFusion, frozen_representation
from proteinmod.hybrid import HybridRegressor
from proteinmod.reliability_experiments import fit, history, output_path
from proteinmod.structure_reliability import (
    CONFIDENCE_COLUMNS,
    FIELDS,
    StructureReliability,
    confidence_graph,
    reliability_features,
)


def graph(source="alphafold_plddt"):
    static = np.zeros((3, 34), np.float32)
    nodes = [
        {
            "reference_position": i,
            "plddt": p if source == "alphafold_plddt" else None,
            "bfactor_a2": p if source == "experimental_bfactor" else None,
        }
        for i, p in zip((1, 3, 5), (25, 70, 95), strict=True)
    ]
    for i, n in enumerate(nodes):
        static[i, CONFIDENCE_COLUMNS] = [
            (n["bfactor_a2"] or 0) / 100,
            (n["plddt"] or 0) / 100,
            n["bfactor_a2"] is None,
            n["plddt"] is None,
        ]
    return {
        "static": static,
        "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_features": np.ones((4, 3), np.float32),
        "nodes": nodes,
        "lookup": {1: 0, 3: 1, 5: 2},
        "xyz": np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]]),
        "audit": {
            "confidence_semantics": source,
            "m2_manifest_sha256": "fixture",
            "mapping": {
                "coordinate_coverage": 0.6,
                "aligned_identity": 1,
                "deposition": {"resolution": 2 if source == "experimental_bfactor" else None},
            },
        },
    }


def test_sources_units_missingness_and_mapping_are_distinct():
    a = reliability_features(("A1V",), "AAAAA", graph())
    b = reliability_features(("A1V",), "AAAAA", graph("experimental_bfactor"))
    assert a[3] == 0.25 and a[5] == 0 and a[7] == 1 and a[9] == 0
    assert b[3] == 0 and b[5] == 0.25 and b[7] == 0 and b[9] == 1
    assert b[15] == pytest.approx(0.4) and b[16] == 1
    assert a[15] == a[16] == 0
    assert reliability_features(("A2V",), "AAAAA", graph())[13] == 0
    assert reliability_features(("A2V",), "AAAAA", None)[-1] == 0
    damaged = graph()
    damaged["nodes"][0]["bfactor_a2"] = 25
    with pytest.raises(ValueError, match="interchanged"):
        StructureReliability.from_graph(damaged)
    with pytest.raises(ValueError, match="Resolution"):
        StructureReliability("alphafold_plddt", 1, 1, 2, "fixture")
    unknown = graph("unknown")
    assert reliability_features(("A1V",), "AAAAA", unknown)[3:11].tolist() == [0] * 8


def test_neutral_masks_legacy_confidence_and_shuffle_preserves_typed_marginal():
    g = graph()
    before = copy.deepcopy(g)
    neutral = confidence_graph(g, "neutral")
    np.testing.assert_array_equal(neutral["static"][:, CONFIDENCE_COLUMNS], [[0, 0, 1, 1]] * 3)
    shuffled = confidence_graph(g, "shuffled", 3)
    assert sorted(n["plddt"] for n in shuffled["nodes"]) == [25, 70, 95]
    np.testing.assert_array_equal(g["static"], before["static"])
    np.testing.assert_array_equal(g["edge_index"], shuffled["edge_index"])
    np.testing.assert_array_equal(confidence_graph(g, "shuffled", 3)["static"], shuffled["static"])
    a = reliability_features(("A1V", "A5C"), "AAAAA", g)
    b = reliability_features(("A5C", "A1V"), "AAAAA", g)
    np.testing.assert_array_equal(a, b)


def test_frozen_encoder_input_effect_multimutant_and_missing_structure_fallback():
    torch.manual_seed(4)
    g = graph()
    state = HybridRegressor(g, 480).state_dict()
    s = np.random.default_rng(0).normal(size=(2, 5, 480)).astype(np.float32)
    d = np.zeros((2, 3, 25), np.float32)
    d[0, [0, 2], 0] = 1
    mask = np.zeros((2, 5), bool)
    mask[0, [0, 4]] = True
    a = frozen_representation(state, g, s, d, mask)
    b = frozen_representation(state, confidence_graph(g, "neutral"), s, d, mask)
    assert not np.array_equal(a, b)
    assert a.shape == (2, 100) and a[0, -4] == 2 and a[1, -4] == 0
    # The unmapped site remains a full-reference sequence query; graph coverage is explicit.
    mask[0, 1] = True
    partial = frozen_representation(state, g, s, d, mask)
    assert partial[0, -1] == pytest.approx(2 / 3)
    missing = frozen_representation(state, None, s, None, mask)
    assert np.isfinite(missing).all() and (missing[:, 48:96] == 0).all()
    assert (missing[:, -2:] == 0).all()
    np.testing.assert_array_equal(missing, frozen_representation(state, None, s, None, mask))
    with pytest.raises(ValueError, match="Finite"):
        frozen_representation(state, None, s * np.nan, None, mask)


def test_gate_gradient_zero_mask_capacity_and_save_load(tmp_path):
    torch.manual_seed(1)
    model = ReliabilityFusion()
    h = torch.randn(4, 100)
    h[:, -4:] = torch.tensor([1, 1, 1, 1])
    q = torch.rand(4, len(FIELDS))
    p, gate, _, _ = model(h, q, "combined", True)
    p.sum().backward()
    assert sum(p.grad.abs().sum() for p in model.gate.parameters()) > 0
    assert (gate > 0).all() and (gate < 1).all()
    h[:, -2:] = 0
    h[:, 48:96] = 0
    p, gate, seq, _ = model(h, q, "combined", True)
    torch.testing.assert_close(p, seq, rtol=0, atol=0)
    assert (gate == 0).all()
    for mode in MODES:
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        other = ReliabilityFusion()
        other.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(model(h, q, mode), other(h, q, mode), rtol=0, atol=0)
    with pytest.raises(ValueError, match="dimensions"):
        model(h[:, :50], q)


def test_fit_ignores_test_labels_and_is_deterministic():
    rows = [
        {
            "id": str(i),
            "protein_id": "p",
            "assay_id": "a",
            "family": "abundance",
            "target": float(i % 5),
        }
        for i in range(12)
    ]
    roles = {"train": np.arange(6), "validation": np.arange(6, 9), "test": np.arange(9, 12)}
    rng = np.random.default_rng(0)
    h = rng.normal(size=(12, 100)).astype(np.float32)
    h[:, -4:] = [1, 1, 1, 1]
    missing = h.copy()
    missing[:, 48:96] = 0
    missing[:, -2:] = 0
    q = rng.random((12, len(FIELDS)), dtype=np.float32)
    settings = {
        "device": "cpu",
        "cpu_threads": 1,
        "hidden": 4,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "max_epochs": 3,
        "patience": 2,
        "batch_size": 3,
    }
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -999
    parameters = []
    for mode in ("unconditioned", "confidence_input", "gate", "combined", "gate_zero"):
        a, fallback, selected, state = fit(h, q, missing, rows, roles, settings, 0, mode)
        b, _, other, second = fit(h, q, missing, changed, roles, settings, 0, mode)
        np.testing.assert_array_equal(a[0], b[0])
        assert selected == other
        assert all(torch.equal(v, second["state"][k]) for k, v in state["state"].items())
        assert (fallback[1] == 0).all()
        parameters.append(selected["parameters"])
    assert len(set(parameters)) == 1


def test_history_guard_includes_all_m9_artifacts_and_old_configs():
    hashes = history()
    assert sum(p.startswith("benchmarks/") for p in hashes) == 219
    assert "configs/m9.json" in hashes and "configs/m10.json" not in hashes
    for path in ("benchmarks", "benchmarks/m9/overwrite", "benchmarks/m0"):
        with pytest.raises(ValueError, match="historical"):
            output_path(path)


def test_m9_history_remains_replayable_after_future_milestone_artifacts(tmp_path, monkeypatch):
    from proteinmod.assay_experiments import history as m9_history

    for name in ("amylase", "m8", "m9", "m10", "m11"):
        p = tmp_path / "benchmarks" / name
        p.mkdir(parents=True)
        (p / "evidence.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert set(m9_history()) == {"benchmarks/amylase/evidence.json", "benchmarks/m8/evidence.json"}
