"""Offline low-label role safety and label-free cache identity."""

import copy

import numpy as np
import pytest

from proteinmod.lora_data import prefix_contract, subset


def test_nested_training_subset_does_not_touch_validation_test_or_labels():
    rows = [{"id": str(i), "protein_id": str(i // 8), "target": i} for i in range(24)]
    roles = {
        "train": np.r_[np.arange(6), np.arange(8, 14)],
        "validation": np.array([6, 7, 14, 15]),
        "test": np.arange(16, 24),
    }
    small = subset(rows, roles, 0.25, "fixed")
    full = subset(rows, roles, 1.0, "fixed")
    assert set(small["train"]) <= set(full["train"]) == set(roles["train"])
    assert {rows[i]["protein_id"] for i in small["train"]} == {"0", "1"}
    np.testing.assert_array_equal(small["validation"], roles["validation"])
    np.testing.assert_array_equal(small["test"], roles["test"])
    changed = copy.deepcopy(rows)
    for r in changed:
        r["target"] = 999
    np.testing.assert_array_equal(subset(changed, roles, 0.25, "fixed")["train"], small["train"])
    with pytest.raises(ValueError):
        subset(rows, roles, 0, "fixed")


def test_cache_contract_changes_with_weights_sequence_boundary_and_revision():
    a = prefix_contract({"revision": "a"}, {"weights": "hash"}, ["one"], ["AAA"], 11)
    assert a != prefix_contract({"revision": "b"}, {"weights": "hash"}, ["one"], ["AAA"], 11)
    assert a != prefix_contract({"revision": "a"}, {"weights": "new"}, ["one"], ["AAA"], 11)
    assert a != prefix_contract({"revision": "a"}, {"weights": "hash"}, ["one"], ["AVA"], 11)
    assert a != prefix_contract({"revision": "a"}, {"weights": "hash"}, ["one"], ["AAA"], 10)


def test_history_protects_old_files_and_excludes_future_milestones(tmp_path, monkeypatch):
    from pathlib import Path

    from proteinmod.lora_data import history

    monkeypatch.chdir(tmp_path)
    for name in ("benchmarks/m14", "benchmarks/m15", "benchmarks/m16", "configs"):
        Path(name).mkdir(parents=True, exist_ok=True)
    protected = Path("benchmarks/m14/metrics.json")
    protected.write_text("historical")
    Path("configs/m14.json").write_text("old config")
    before = history()
    assert set(before) == {"benchmarks/m14/metrics.json", "configs/m14.json"}
    Path("benchmarks/m15/metrics.json").write_text("new")
    Path("benchmarks/m16/metrics.json").write_text("future")
    Path("configs/m15.json").write_text("new config")
    assert history() == before
    protected.write_text("changed")
    assert history() != before
