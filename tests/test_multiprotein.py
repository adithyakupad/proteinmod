"""Cross-protein protocol, assay-scale and label-leakage checks."""

import hashlib
import json

import numpy as np
import pytest

from proteinmod.data import checksum
from proteinmod.multiprotein_data import audit_roles, collection_rows, protocols, train_percentiles
from proteinmod.multiprotein_experiments import fit_targets, ridge_fit


def examples():
    return [
        {
            "id": f"{p}:{j}",
            "protein_id": p,
            "assay_id": p + "_assay",
            "sequence": p + str(j),
            "target": j * scale,
            "partition": part,
        }
        for p, scale in [("a", 1), ("b", 1000), ("c", 0.001)]
        for j, part in enumerate(["train", "train", "validation", "test"])
    ]


def test_protein_held_out_and_protein_alias_leakage():
    rows = examples()
    folds = protocols(rows, ["a", "b", "c"])
    assert len(folds) == 4
    for _, test, roles in folds[1:]:
        assert {rows[i]["protein_id"] for i in roles["test"]} == {test}
        audit_roles(rows, roles, held_out=True)
    bad = dict(folds[1][2], train=np.array([0, 1]), validation=np.array([2]), test=np.array([3]))
    with pytest.raises(ValueError, match="protein leakage"):
        audit_roles(rows, bad, held_out=True)
    with pytest.raises(ValueError, match="three"):
        protocols(rows, ["a", "b"])
    rows[4]["sequence"] = rows[0]["sequence"]
    with pytest.raises(ValueError, match="sequence leakage"):
        audit_roles(rows, folds[1][2], held_out=True)


def test_training_assay_rank_targets_ignore_scale_and_held_out_labels():
    np.testing.assert_allclose(train_percentiles([1, 1, 4, 8]), [0.25, 0.25, 0.625, 0.875])
    np.testing.assert_array_equal(train_percentiles([1, 2, 3]), train_percentiles([101, 102, 103]))
    rows = examples()
    train = np.array([0, 1, 4, 5])
    target = fit_targets(rows, train)
    np.testing.assert_array_equal(target[:2], target[4:6])
    assert np.isnan(target[8:]).all()
    for i in [2, 3, 6, 7, 8, 9, 10, 11]:
        rows[i]["target"] = -999
    np.testing.assert_array_equal(target, fit_targets(rows, train))


def test_test_labels_cannot_affect_ridge_selection():
    rows = examples()
    roles = protocols(rows, ["a", "b", "c"])[1][2]
    x = np.arange(len(rows) * 3).reshape(len(rows), 3)
    pred, selection = ridge_fit(x, rows, roles, [0.1, 10])
    for i in roles["test"]:
        rows[i]["target"] = 999 - i
    again, chosen = ridge_fit(x, rows, roles, [0.1, 10])
    np.testing.assert_array_equal(pred, again)
    assert selection == chosen


def test_crop_adapter_and_label_blind_sampling(tmp_path):
    import gzip

    p = tmp_path / "sample.csv.gz"
    p.write_bytes(
        gzip.compress(
            b"sequence,target,set,validation\nMACD,0,train,\nMVCD,1,train,\nMACV,2,train,True\nMACE,3,test,\nVACD,4,test,\n"
        )
    )
    ref = lambda s: {
        "sequence": s,
        "source": "synthetic fixture",
        "sha256": hashlib.sha256(s.encode()).hexdigest(),
    }
    spec = {
        "protein_id": "p",
        "assay_id": "a",
        "dataset": {"path": str(p), "sha256": checksum(p)},
        "reference": ref("ACD"),
        "full_reference": ref("MACD"),
        "crop": [1, 4],
        "adapter": "flip2_blank_validation_false",
    }
    rows, audit = collection_rows(spec, {"train": 1, "validation": 1, "test": 1})
    assert len(rows) == 3 and audit["outside_crop_exclusions"] == ["5"]
    assert all(r["id"].startswith("a:") and len(r["sequence"]) == 3 for r in rows)
    assert {r["partition"] for r in rows} == {"train", "validation", "test"}
    changed = gzip.decompress(p.read_bytes()).replace(b",0,", b",100,").replace(b",1,", b",200,")
    p.write_bytes(gzip.compress(changed))
    spec["dataset"]["sha256"] = checksum(p)
    again, _ = collection_rows(spec, {"train": 1, "validation": 1, "test": 1})
    assert [r["id"] for r in rows] == [r["id"] for r in again]


def test_config_pins_minimal_three_proteins():
    with open("configs/m4.json") as handle:
        c = json.load(handle)
    assert len(c["collection"]) == 3
    assert all(s["reference"]["source"] and s["dataset"].get("license") for s in c["collection"])


def test_history_protection_and_reference_alias_guard():
    from proteinmod.multiprotein_data import collection_output, model_inputs

    for p in ["benchmarks", "benchmarks/amylase_m3", "benchmarks/structures/new"]:
        with pytest.raises(ValueError, match="historical"):
            collection_output({"experiment": {"output": p}})
    same = [
        {"protein_id": p, "assay_id": p, "reference": {"sequence": "ACD"}} for p in ["a", "b", "c"]
    ]
    with pytest.raises(ValueError, match="distinct"):
        model_inputs({"collection": same})
