"""Offline thermodynamic semantics, source coordinates and leakage regression checks."""

import copy
import hashlib

import numpy as np
import pytest

from proteinmod.data import AMINO_ACIDS
from proteinmod.stability_data import (
    audit_stability_roles,
    ddg_target,
    merge_measurements,
    mutant_sequence,
    output_directory,
    pair_key,
    rows_from_table,
)
from proteinmod.stability_experiments import ridge_fit, validate_task


def assay():
    return {
        "id": "a",
        "unit": "kcal/mol",
        "source_convention": "positive_stabilizing",
        "task_type": "ddg",
        "conditions": {"pH": 7.4, "temperature_k": 298.0, "replicates": None},
    }


def test_sign_units_and_censoring_are_not_proxies():
    assert ddg_target(2.0, "kcal/mol", "positive_destabilizing") == 2
    assert ddg_target(4.184, "kJ/mol", "positive_stabilizing") == -1
    for value in (None, "-", "NA", ">3", float("nan"), float("inf")):
        assert ddg_target(value, "kcal/mol", "positive_stabilizing") is None
    for task in ("dtm", "dg", "enrichment"):
        with pytest.raises(ValueError, match="separate"):
            ddg_target(2, "kcal/mol", "positive_stabilizing", task)
    with pytest.raises(ValueError, match="units"):
        ddg_target(2, "Celsius", "positive_destabilizing")


def test_wide_author_table_whitelist_masks_and_mutation_mapping():
    reference = "ACD"
    spec = {
        "protein_id": "p",
        "reference": {
            "sequence": reference,
            "source": "fixture",
            "sha256": hashlib.sha256(reference.encode()).hexdigest(),
        },
    }
    table = [
        {
            "pdb_name": "p.pdb",
            "pos": "1",
            "wt_aa": "A",
            **{"ddg_" + a: "0" if a == "A" else "1" for a in AMINO_ACIDS},
            "meanzero_dg_A": "999999",
            "pc1": "-9999",
        }
    ]
    a, audit = rows_from_table(table, [spec], assay(), 600)
    assert len(a) == 18 and all(r["target"] == -1 for r in a)
    assert all(
        r["target_mask"] and not r["label_ci_available"] and not r["replicate_quality_available"]
        for r in a
    )
    assert all(not r["conditions_mask"]["replicates"] for r in a)
    other = copy.deepcopy(table)
    other[0]["pc1"] = "fake label derived feature"
    assert rows_from_table(other, [spec], assay(), 600) == (a, audit)
    assert (
        mutant_sequence(reference, ["D3V", "A1L"])
        == mutant_sequence(reference, ["A1L", "D3V"])
        == "LCV"
    )
    with pytest.raises(ValueError, match="WT"):
        mutant_sequence(reference, ["L1V"])
    other[0]["wt_aa"] = "V"
    with pytest.raises(ValueError, match="mismatch"):
        rows_from_table(other, [spec], assay(), 600)
    with pytest.raises(ValueError, match="Duplicate"):
        rows_from_table(table + table, [spec], assay(), 600)


def test_repeated_measurements_keep_distinct_conditions_and_assays():
    base = {
        "id": "1",
        "reference": "AC",
        "sequence": "VC",
        "assay_id": "a",
        "conditions": {"pH": 7.4},
        "target": 1.0,
        "target_mask": True,
    }
    rows = [
        base,
        {**base, "id": "2", "target": 3.0},
        {**base, "id": "3", "conditions": {"pH": 5.0}},
        {**base, "id": "4", "assay_id": "b"},
    ]
    result = merge_measurements(rows + [base])
    assert result[0]["duplicate_records_removed"] == 1
    assert len(result) == 3 and result[0]["target"] == 2 and result[0]["measurement_count"] == 2
    assert result[0]["measurement_sd_kcal_mol"] == 1
    assert pair_key("AC", "VC") == pair_key("VC", "AC")


def test_reverse_pair_and_shared_construct_leakage():
    rows = [
        {"protein_id": p, "sequence": s, "pair_key": pair_key(a, s)}
        for p, a, s in [("p", "AC", "VC"), ("q", "VC", "AC"), ("r", "DE", "FE")]
    ]
    roles = {"train": np.array([0]), "test": np.array([1]), "validation": np.array([2])}
    with pytest.raises(ValueError, match="reverse"):
        audit_stability_roles(rows, roles, True, [])
    rows[1]["pair_key"] = "other"
    specs = [
        {"protein_id": p, "structure_member": "same", "reference": {"sequence": s}}
        for p, s in [("p", "AC"), ("q", "VC"), ("r", "DE")]
    ]
    with pytest.raises(ValueError, match="construct"):
        audit_stability_roles(rows, roles, True, specs)
    for spec in specs:
        spec["structure_member"] = spec["protein_id"]
    specs[1]["reference"]["sequence"] = "AC"
    with pytest.raises(ValueError, match="Homolog"):
        audit_stability_roles(rows, roles, True, specs)


def task_rows():
    return [
        {
            "id": str(i),
            "target": float(i) / 10,
            "target_mask": True,
            "task_type": "ddg",
            "target_unit": "kcal/mol",
            "target_convention": "positive_destabilizing",
            "assay_id": "a",
            "conditions": {"pH": 7.4},
            "protein_id": "p",
        }
        for i in range(10)
    ]


def test_physical_ridge_ignores_test_targets_and_task_masks():
    rows = task_rows()
    x = np.arange(40, dtype=float).reshape(10, 4)
    roles = {"train": np.arange(6), "validation": np.array([6, 7]), "test": np.array([8, 9])}
    a, sel = ridge_fit(x, rows, roles, [0.1, 10.0])
    changed = copy.deepcopy(rows)
    for i in roles["test"]:
        changed[i]["target"] = -1000
    b, other = ridge_fit(x, changed, roles, [0.1, 10.0])
    np.testing.assert_array_equal(a, b)
    assert sel == other
    changed[0]["assay_id"] = "different"
    with pytest.raises(ValueError, match="separate"):
        validate_task(changed)
    changed = copy.deepcopy(rows)
    changed[0]["target_mask"] = False
    with pytest.raises(ValueError, match="canonical"):
        validate_task(changed)
    for path in ("benchmarks/m5/new", "benchmarks/m4", "benchmarks", "."):
        with pytest.raises(ValueError, match="historical"):
            output_directory(path)
