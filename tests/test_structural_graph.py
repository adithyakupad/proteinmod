"""M3 graph and eligibility tests use tiny synthetic M2-compatible artifacts."""

import csv
import json

import numpy as np
import pytest

from proteinmod.data import checksum, write_json
from proteinmod.structural_experiments import output_directory
from proteinmod.structural_graph import load_graph, prepare_inputs, variant_inputs


@pytest.fixture
def graph(tmp_path):
    reference = "ACDE"
    rows = [
        {
            "reference_position": i + 1,
            "reference_aa": aa,
            "status": "mapped" if i != 2 else "sequence_mismatch",
            "ca": [i * 3.8, 0, 0],
            "rsa": 0.2,
            "contact_degree": 2,
            "nonlocal_contact_degree": 0,
            "bfactor_a2": 10,
            "plddt": None,
        }
        for i, aa in enumerate(reference)
    ]
    write_json(tmp_path / "residues.json", rows)
    write_json(tmp_path / "audit.json", {"accepted": True, "pipeline_status": "complete"})
    with (tmp_path / "contacts.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "distance_a", "spatial", "backbone"])
        w.writerows([(1, 2, 3.8, True, True), (2, 3, 3.8, True, True), (3, 4, 3.8, True, True)])
    write_json(
        tmp_path / "manifest.json",
        {
            "config": {"reference": {"sequence": reference}},
            "confidence_semantics": "experimental_bfactor",
            "artifact_sha256": {
                name: checksum(tmp_path / name)
                for name in ["residues.json", "contacts.csv", "audit.json"]
            },
        },
    )
    return load_graph(tmp_path, reference)


def test_induced_graph_and_missing_confidence(graph):
    assert len(graph["nodes"]) == 3
    assert graph["audit"]["undirected_edges"] == 1
    assert graph["audit"]["connected_components"] == 2
    assert graph["audit"]["isolated_nodes"] == 1
    assert graph["audit"]["descriptor_missing_nodes"]["plddt"] == 3
    assert np.isfinite(graph["static"]).all()


def test_permutation_wt_and_multi_mutations(graph):
    a, fa, reasons = variant_inputs(["A1V", "E4K"], "ACDE", graph)
    b, fb, _ = variant_inputs(["E4K", "A1V"], "ACDE", graph)
    assert not reasons
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fa, fb)
    np.testing.assert_allclose(fa[-6:], [2, 1, 1, 0.38, 0.38, 0.38])
    wt, fw, _ = variant_inputs([], "ACDE", graph)
    assert not wt.any() and not fw.any()
    _, _, reasons = variant_inputs(["D3A"], "ACDE", graph)
    assert reasons == [{"position": 3, "reason": "sequence_mismatch"}]
    with pytest.raises(ValueError, match="duplicate"):
        variant_inputs(["A1V", "A1G"], "ACDE", graph)


def test_eligibility_is_label_independent(graph):
    rows = [
        {"id": "1", "partition": "train", "mutations": ["A1V"], "target": 1},
        {"id": "2", "partition": "test", "mutations": ["D3A"], "target": -5},
    ]
    a = prepare_inputs(rows, "ACDE", graph)
    for row in rows:
        row["target"] = 999
    b = prepare_inputs(rows, "ACDE", graph)
    for left, right in zip(a[:3], b[:3], strict=True):
        np.testing.assert_array_equal(left, right)
    assert a[3] == b[3]


def test_artifact_and_reference_trust_boundary(graph, tmp_path):
    with pytest.raises(ValueError, match="reference"):
        load_graph(tmp_path, "AAAA")
    p = tmp_path / "residues.json"
    p.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="checksum"):
        load_graph(tmp_path, "ACDE")


def test_historical_output_rejected():
    for output in ["benchmarks", "benchmarks/amylase/new", "benchmarks/structures"]:
        with pytest.raises(ValueError, match="historical"):
            output_directory({"output": output})
