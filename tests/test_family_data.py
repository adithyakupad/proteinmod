"""Offline M12 family/group leakage, domain audit and data contract tests."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from proteinmod.data import checksum
from proteinmod.family_data import alignment, audit, collection, folds, similarity
from proteinmod.multiprotein_experiments import fit_targets, ridge_fit
from proteinmod.structure_mapping import map_reference, parse_structure


def examples():
    return [
        {
            "id": f"{f}:{p}:{j}",
            "family": f,
            "protein_id": f"{f}:{p}",
            "assay_id": f"{f}:{p}",
            "sequence": f"{f}:{p}:{j}",
            "target": float(j),
            "partition": role,
        }
        for f in ("a", "b", "c", "d")
        for p in (0, 1)
        for j, role in enumerate(("train", "train", "validation", "validation", "test", "test"))
    ]


def test_group_protocols_determinism_and_family_leakage():
    rows = examples()
    plan = folds(rows, ["a", "b", "c", "d"])
    assert len(plan) == 13
    assert [x[:2] for x in plan] == [x[:2] for x in folds(rows, ["a", "b", "c", "d"])]
    for protocol, _, roles in plan:
        audit(rows, roles, protocol)
        if protocol == "G3":
            sets = [{rows[i]["family"] for i in ix} for ix in roles.values()]
            assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    _, _, roles = next(p for p in plan if p[0] == "G3")
    changed = copy.deepcopy(rows)
    changed[roles["train"][0]]["family"] = rows[roles["test"][0]]["family"]
    with pytest.raises(ValueError, match="family leakage"):
        audit(changed, roles, "G3")
    changed = copy.deepcopy(rows)
    changed[roles["train"][0]]["sequence"] = rows[roles["test"][0]]["sequence"]
    with pytest.raises(ValueError, match="sequence leakage"):
        audit(changed, roles, "G3")


def test_domain_partial_audit_and_identity_ambiguity():
    domain = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
    result = alignment(domain + "A" * 40, "Y" * 40 + domain, "local")
    assert result["identity"] == 1 and result["aligned"] >= len(domain)
    assert alignment("AAAA", "AAA")["ambiguous_optimum"]
    c = {
        "proteins": [
            {"protein_id": "a", "family": "one", "reference": {"sequence": domain}},
            {"protein_id": "b", "family": "two", "reference": {"sequence": domain}},
        ],
        "similarity": {
            "cluster_identity_thresholds": [0.4, 0.8],
            "min_both_coverage": 0.7,
            "partial_min_aligned": 25,
            "partial_min_identity": 0.5,
            "partial_min_shorter_coverage": 0.6,
        },
    }
    with pytest.raises(ValueError, match="cross-family"):
        similarity(c)
    c["proteins"][1]["family"] = "one"
    assert similarity(c)["clusters"]["0.8"] == [["a", "b"]]


def test_training_only_normalization_and_test_blind_selection():
    rows = examples()
    roles = next(p[2] for p in folds(rows, ["a", "b", "c", "d"]) if p[0] == "G3")
    x = np.random.default_rng(7).normal(size=(len(rows), 5))
    pred, sel = ridge_fit(x, rows, roles, [0.1, 10])
    target = fit_targets(rows, roles["train"])
    for i in roles["test"]:
        rows[i]["target"] = 99999 - i
    other, chosen = ridge_fit(x, rows, roles, [0.1, 10])
    np.testing.assert_array_equal(pred, other)
    assert sel == chosen
    np.testing.assert_array_equal(target, fit_targets(rows, roles["train"]))
    assert np.isnan(target[roles["test"]]).all()


def test_adapter_mapping_missing_labels_and_external_family(tmp_path):
    path = tmp_path / "data.csv"
    records = [
        {"mutant": f"A1{x}", "mutated_sequence": x + "CDE", "DMS_score": str(j)}
        for j, x in enumerate("FGHIKLM")
    ]
    records[-1]["DMS_score"] = "nan"
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=records[0])
        w.writeheader()
        w.writerows(records)
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps({"rcsb_polymer_entity_feature": [{"type": "Pfam", "feature_id": "PFtest"}]})
    )
    spec = {
        "protein_id": "p",
        "assay_id": "assay",
        "family": "PFtest",
        "reference": {
            "sequence": "ACDE",
            "sha256": hashlib.sha256(b"ACDE").hexdigest(),
            "source": "synthetic",
        },
        "dataset": {"path": str(path), "sha256": checksum(path)},
        "classification": {"path": str(annotation), "sha256": checksum(annotation)},
    }
    config = {
        "schema": "m12-family-held-out-v1",
        "proteins": [spec],
        "partition_counts": [2, 2, 2],
        "cap_per_protein": 6,
        "split_salt": "test",
    }
    rows, evidence = collection(config)
    assert len(rows) == 6 and evidence["p"]["missing_labels"] == 1
    assert [r["id"] for r in rows] == [r["id"] for r in collection(config)[0]]
    spec["family"] = "wrong"
    with pytest.raises(ValueError, match="external"):
        collection(config)
    spec["family"] = "PFtest"
    path.write_text(path.read_text().replace("FCDE", "YCDE"))
    spec["dataset"]["sha256"] = checksum(path)
    with pytest.raises(ValueError, match="mismatch"):
        collection(config)


def test_explicit_construct_range_requires_exact_reference():
    parsed = parse_structure(Path(__file__).parent / "fixtures/structure_tiny.cif")
    rows, evidence = map_reference(
        parsed, "ACDEFGHM", {"chain": "A", "min_coverage": 0.5, "reference_polymer_range": [1, 8]}
    )
    assert evidence["accepted"] and rows[1]["polymer_position"] == 2
    for span in ([0, 8], [1, 7], [2, 9], [1.0, 8], [1, 8, 9]):
        assert not map_reference(
            parsed, "ACDEFGHM", {"chain": "A", "reference_polymer_range": span}
        )[1]["accepted"]


def test_ambiguous_family_or_order_rejected():
    rows = examples()
    with pytest.raises(ValueError, match="four classified"):
        folds(rows, ["a", "b", "c", "a"])
    rows[0]["family"] = "b"
    with pytest.raises(ValueError, match="four classified"):
        folds(rows, ["a", "b", "c", "d"])
