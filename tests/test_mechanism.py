"""Outcome-blind roles and exact graph-null constraints."""

import json
from pathlib import Path

import numpy as np
import pytest

from proteinmod.mechanism import balanced, degree_null, study_roles


def test_fixed_roles_and_nested_labels():
    rows = json.loads(Path("benchmarks/m12/rows.json").read_text())
    families = json.loads(Path("configs/m12.json").read_text())["family_order"]
    cases = study_roles(rows, families)
    for study, fold, protocol, roles in cases:
        assert not (set(roles["train"]) & set(roles["test"]))
        assert not (set(roles["validation"]) & set(roles["test"]))
        if study == "H1":
            train_family = {rows[i]["family"] for i in roles["train"]}
            assert not train_family & {rows[i]["family"] for i in roles["test"]}
            assert len(roles["test"]) == 32
            previous = set()
            for f in (0.1, 0.25, 0.5, 1):
                ix = set(balanced(rows, roles["train"], f, 3))
                assert previous <= ix
                previous = ix
        if study == "H2":
            assert len(roles["train"]) == 128 and len(roles["test"]) == 16
            if protocol != "G1":
                assert fold not in {rows[i]["protein_id"] for i in roles["train"]}
            if protocol == "G3":
                assert not {rows[i]["family"] for i in roles["train"]} & {
                    rows[i]["family"] for i in roles["test"]
                }
    scrambled = [{**r, "target": float("nan")} for r in rows]
    for a, b in zip(cases, study_roles(scrambled, families), strict=True):
        assert a[:3] == b[:3]
        assert all(np.array_equal(a[3][k], b[3][k]) for k in a[3])


def test_degree_null():
    edges = np.array([(i, (i + j) % 12) for i in range(12) for j in (1, 11, 2, 10)]).T
    null, n = degree_null(edges, 3)
    assert n > 0 and not np.array_equal(edges, null)
    assert np.array_equal(np.bincount(edges[0]), np.bincount(null[0]))
    assert len({tuple(e) for e in null.T}) == null.shape[1]
    assert all(a != b for a, b in null.T)
    assert {tuple(e) for e in null.T} == {tuple(e[::-1]) for e in null.T}
    assert np.array_equal(null, degree_null(edges, 3)[0])
    with pytest.raises(ValueError):
        degree_null(np.array([[0, 0], [0, 0]]), 0)


def test_reporting_keeps_undefined_correlations():
    import importlib.util

    spec = importlib.util.spec_from_file_location("m17report", "scripts/report_m17.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.mean([0.2, 0.4]) == pytest.approx(0.3)
    assert module.mean([0.2, None]) is None
    assert module.mean([]) is None
    assert module.rho([1, 2, 3], [1, 1, 1]) is None
    assert module.rho([1, 2, 3], [3, 2, 1]) == -1


def test_committed_split_manifest_matches_label_blind_generator():
    rows = json.loads(Path("benchmarks/m12/rows.json").read_text())
    config = json.loads(Path("configs/m17.json").read_text())
    families = json.loads(Path("configs/m12.json").read_text())["family_order"]
    expected = []
    for study, fold, protocol, roles in study_roles(rows, families):
        for seed in config["seeds"] if study == "H1" else [0]:
            for fraction in config["fractions"] if study == "H1" else [1.0]:
                selected = {**roles, "train": balanced(rows, roles["train"], fraction, seed)}
                expected.append(
                    {
                        "study": study,
                        "fold": fold,
                        "protocol": protocol,
                        "seed": seed,
                        "fraction": fraction,
                        "roles": {k: [rows[i]["id"] for i in ix] for k, ix in selected.items()},
                    }
                )
    actual = json.loads(Path("benchmarks/m17/split-manifest.json").read_text())["folds"]
    assert expected == actual
