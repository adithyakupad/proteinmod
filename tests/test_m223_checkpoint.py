"""Synthetic numeric contracts only. No historical trainer is imported."""

import copy
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from proteinmod.synthetic_checkpoint import (
    CheckpointError,
    Contract,
    canonical,
    digest,
    evaluate,
    initial,
    load,
    numeric_data,
    pack,
    save,
    train,
    unpack,
    update,
)


def contract(seed=0):
    x, y, groups, _, _ = numeric_data(seed)
    return Contract(x, y, groups, seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_120_equals_disk_12_plus_108(seed, tmp_path):
    c = contract(seed)
    full = train(c, 120)
    prefix = train(c, 12)
    path = tmp_path / "state.json"
    trusted = save(path, prefix, c)
    resumed = train(c, 120, load(path, c, trusted))
    assert canonical(full) == canonical(resumed)
    assert canonical(prefix) == canonical(train(c, 12))
    assert canonical(full) == canonical(train(c, 120))
    assert full["updates"] == 120 * 10
    assert prefix["updates"] == 12 * 10
    with pytest.raises(FileExistsError):
        save(path, prefix, c)


def test_schedule_complete_grouped_ragged_and_immutable():
    c = contract()
    for epoch in c.schedule:
        assert sorted(i for batch in epoch for i in batch) == list(range(59))
        assert sorted(map(len, epoch)) == [1, 1, 1, 8, 8, 8, 8, 8, 8, 8]
        assert all(len({c.groups[i] for i in batch}) == 1 for batch in epoch)
    next(batch for batch in c.schedule[0] if len(batch) > 1).reverse()
    with pytest.raises(CheckpointError, match="changed"):
        train(c, 1)


def test_optimizer_reset_is_rejected_and_moments_matter():
    c = contract()
    prefix = train(c, 12)
    invalid = copy.deepcopy(prefix)
    invalid["optimizer_step"] = 0
    for kind in ("m", "v"):
        invalid[kind] = {key: pack(np.zeros(shape)) for key, shape in c.shapes.items()}
    with pytest.raises(CheckpointError, match="counters"):
        train(c, 120, invalid)
    # Deliberately bypass the loader for a labeled negative control, never a valid resume.
    params = {key: unpack(value, c.shapes[key]) for key, value in prefix["parameters"].items()}
    moments = {key: np.zeros(shape) for key, shape in c.shapes.items()}
    squares = copy.deepcopy(moments)
    x = (c.x - c.mean) / c.scale
    step = 0
    for batches in c.schedule[12:]:
        for batch in batches:
            step += 1
            update(params, moments, squares, step, x[batch], c.y[batch])
    proper = train(c, 120, prefix)
    assert any(
        not np.array_equal(params[key], unpack(value, c.shapes[key]))
        for key, value in proper["parameters"].items()
    )


def test_heldout_noninterference():
    x, y, groups, held, target = numeric_data(0)
    c = Contract(x, y, groups, 0)
    before = train(c, 12)
    evaluate(before, c, held, target)
    held[:] = 1e5
    target[:] = -1e5
    evaluate(before, c, held, target)
    assert canonical(before) == canonical(train(c, 12))
    assert canonical(train(c, 120, before)) == canonical(train(c, 120))


@pytest.mark.parametrize("change", ["value", "row_order", "group", "seed", "dtype", "shape"])
def test_changed_input_rejected(change, tmp_path):
    c = contract()
    path = tmp_path / "checkpoint.json"
    trusted = save(path, train(c, 12), c)
    x, y, groups, _, _ = numeric_data(0)
    seed = 0
    if change == "value":
        y[0] += 1
    elif change == "row_order":
        x = x[::-1].copy()
        y = y[::-1].copy()
        groups = groups[::-1]
    elif change == "group":
        groups[0] = "d"
    elif change == "seed":
        seed = 1
    elif change == "dtype":
        x = x.astype("float32")
    else:
        x = x[:, :-1]
    with pytest.raises(CheckpointError):
        load(path, Contract(x, y, groups, seed), trusted)


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "unknown",
        "missing",
        "dtype",
        "shape",
        "nan",
        "negative_v",
        "rng",
        "updates",
        "epoch",
        "config",
        "preprocessing",
        "implementation",
        "environment",
    ],
)
def test_semantic_rejection_even_with_recomputed_checksums(change, tmp_path):
    c = contract()
    state = train(c, 12)
    if change == "version":
        state["version"] = 2
    elif change == "unknown":
        state["unknown"] = 1
    elif change == "missing":
        del state["m"]
    elif change == "dtype":
        state["parameters"]["w1"]["dtype"] = "<f4"
    elif change == "shape":
        state["parameters"]["w1"]["shape"] = [72]
    elif change == "nan":
        state["parameters"]["w1"]["values"][0][0] = "NaN"
    elif change == "negative_v":
        state["v"]["w1"]["values"][0][0] = -1
    elif change == "rng":
        state["rng"]["state"]["state"] += 1
    elif change == "updates":
        state["updates"] += 1
    elif change == "epoch":
        state["epoch"] = True
    elif change == "config":
        state["identity"]["config"]["learning_rate"] = 0.1
    elif change == "preprocessing":
        state["identity"]["preprocessing"]["mean"]["values"][0] += 1
    elif change == "implementation":
        state["identity"]["implementation_sha256"] = "0" * 64
    else:
        state["identity"]["environment"]["numpy"] = "different"
    data = canonical({"payload": state, "payload_sha256": digest(state)})
    path = tmp_path / "bad.json"
    path.write_bytes(data)
    with pytest.raises(CheckpointError):
        load(path, c, hashlib.sha256(data).hexdigest())


def test_corruption_truncation_duplicate_keys_and_checksum(tmp_path):
    c = contract()
    path = tmp_path / "state.json"
    trusted = save(path, train(c, 12), c)
    original = path.read_bytes()
    path.write_bytes(original[:-8])
    with pytest.raises(CheckpointError, match="SHA256"):
        load(path, c, trusted)
    for data in (
        b"{",
        b'{"payload":1,"payload":2,"payload_sha256":"x"}',
        b'{"payload":{},"payload_sha256":"wrong"}',
        b'{"payload":NaN,"payload_sha256":"x"}',
    ):
        path.write_bytes(data)
        with pytest.raises(CheckpointError):
            load(path, c, hashlib.sha256(data).hexdigest())


def test_rejections_survive_optimized_python(tmp_path):
    code = """
from proteinmod.synthetic_checkpoint import Contract, numeric_data, initial, train, CheckpointError
x,y,g,_,_=numeric_data(0)
c=Contract(x,y,g,0)
s=initial(c)
s['version']=999
try:
    train(c,120,s)
except CheckpointError:
    pass
else:
    raise RuntimeError('fail-closed validation disabled by -O')
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_snapshot_isolation_and_changed_contract():
    c = contract()
    state = train(c, 12)
    original = canonical(state)
    train(c, 13, state)
    assert canonical(state) == original
    c.x[0, 0] += 1
    with pytest.raises(CheckpointError):
        train(c, 13, state)


def test_json_result_serializable():
    c = contract()
    _, _, _, x, y = numeric_data(0)
    metrics = evaluate(train(c, 1), c, x, y)
    assert set(json.loads(canonical(metrics))) == {"mse", "spearman"}


def test_ci_collection_guard_rejects_historical_edits():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "m223_root", Path(__file__).parents[1] / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_synthetic_changes(["A\tsrc/proteinmod/synthetic_checkpoint.py"])
    for lines in (
        [],
        ["M\tsrc/proteinmod/edge.py"],
        ["A\tunexpected.py"],
        ["D\tbenchmarks/m223/PROTOCOL.md"],
    ):
        with pytest.raises(ValueError):
            module.validate_synthetic_changes(lines)


def test_checkpoint_size_limit_and_numeric_coercion(tmp_path):
    c = contract()
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * 2_000_001)
    with pytest.raises(CheckpointError, match="size"):
        load(path, c, hashlib.sha256(path.read_bytes()).hexdigest())
    for invalid in ("0.5", True, None):
        value = pack(np.ones((1,)))
        value["values"][0] = invalid
        with pytest.raises(CheckpointError):
            unpack(value, (1,))


def test_heldout_metrics_are_not_a_resume_condition():
    c = contract()
    prefix = train(c, 12)
    _, _, _, x, y = numeric_data(0)
    before = canonical(prefix)
    evaluate(prefix, c, x, y)
    assert canonical(prefix) == before
    c.mean[0] += 1
    with pytest.raises(CheckpointError):
        train(c, 13, prefix)


def test_cold_process_disk_continuation(tmp_path):
    c = contract()
    path = tmp_path / "state.json"
    trusted = save(path, train(c, 12), c)
    code = """
import sys
from proteinmod.synthetic_checkpoint import Contract,numeric_data,load,train,digest
x,y,g,_,_=numeric_data(0)
c=Contract(x,y,g,0)
print(digest(train(c,120,load(sys.argv[1],c,sys.argv[2]))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), trusted], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == digest(train(c, 120))


def test_portable_rounding_is_separate_from_checkpoint_identity(monkeypatch):
    import importlib.util
    from pathlib import Path

    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(variable, os.environ.get(variable, "1"))
    spec = importlib.util.spec_from_file_location(
        "m223_runner", Path(__file__).parents[1] / "scripts/run_m223.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    c = contract()
    _, _, groups, held, target = numeric_data(0)
    metrics = {
        h: {s: {"mse": 1.0, "spearman": 0.5} for s in ("train", "validation", "test")}
        for h in ("epoch12", "full120", "resumed120")
    }
    case = {
        "seed": 0,
        "identity": c.identity,
        "metrics": metrics,
        "fixtures": {
            "x": pack(c.x),
            "y": pack(c.y),
            "held": pack(held),
            "target": pack(target),
            "groups": groups,
        },
    }
    actual = {"seeds": [case]}
    rounded = copy.deepcopy(actual)
    rounded["seeds"][0]["fixtures"]["y"]["values"][0] += 1e-15
    rounded_case = rounded["seeds"][0]
    rounded_case["identity"]["input_sha256"] = digest(
        {k: rounded_case["fixtures"][k] for k in ("x", "y", "groups")}
    )
    assert runner.compare(actual, rounded) == 0
    # The same numeric alteration remains forbidden for checkpoint continuation.
    with pytest.raises(CheckpointError):
        bad = initial(c)
        bad["identity"] = rounded_case["identity"]
        train(c, 1, bad)
    rounded_case["fixtures"]["y"]["values"][0] += 0.1
    rounded_case["identity"]["input_sha256"] = digest(
        {k: rounded_case["fixtures"][k] for k in ("x", "y", "groups")}
    )
    with pytest.raises(ValueError, match="rounding"):
        runner.compare(actual, rounded)
    changed = copy.deepcopy(actual)
    changed["seeds"][0]["identity"]["config"]["learning_rate"] = 1.0
    with pytest.raises(ValueError, match="identity"):
        runner.compare(actual, changed)
