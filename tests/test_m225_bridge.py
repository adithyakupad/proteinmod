"""Prespecified synthetic-only serialization bridge and adversarial controls."""

import copy
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from proteinmod import synthetic_bridge as bridge
from proteinmod import synthetic_regression as original
from proteinmod.synthetic_checkpoint import CheckpointError, canonical, digest


def fixture(scenario_id="scarce_noisy", seed=0):
    config = bridge.registered_config()
    scenario = next(s for s in config["scenarios"] if s["id"] == scenario_id)
    data = original.generate(config, scenario, seed)
    contract = bridge.Contract(*data["train"], scenario_id, seed)
    prefix = original.fit(config, *data["train"], seed, stop=12)[12]
    return contract, prefix, data


@pytest.mark.parametrize("scenario", ["ample_clean", "scarce_noisy"])
def test_exact_substitution_and_terminal_roundtrip(scenario, tmp_path):
    c, prefix, data = fixture(scenario)
    full = original.fit(c.config, *data["train"], c.seed)
    bridge.require_equal(prefix, full[12])
    path = tmp_path / "prefix.json"
    trusted = bridge.save(path, prefix, c)
    restored = bridge.load(path, c, trusted)
    bridge.require_equal(prefix, restored)
    resumed = bridge.resume(path, c, trusted)
    bridge.require_equal(full[120], resumed)
    for split, arrays in data.items():
        assert original.metrics(full[120], *arrays) == original.metrics(resumed, *arrays), split
    terminal = tmp_path / "terminal.json"
    end_hash = bridge.save(terminal, resumed, c)
    bridge.require_equal(resumed, bridge.load(terminal, c, end_hash))
    with pytest.raises(CheckpointError, match="terminal"):
        bridge.resume(terminal, c, end_hash)
    with pytest.raises(FileExistsError):
        bridge.save(path, prefix, c)


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "schema",
        "unknown",
        "missing",
        "dtype",
        "shape",
        "nan",
        "coercion",
        "negative_v",
        "rng",
        "counter",
        "mean",
        "trace",
        "identity",
        "schedule",
        "environment",
        "source",
    ],
)
def test_rejection_recomputed_checksums(change, tmp_path):
    c, prefix, _ = fixture()
    payload = bridge.encode(prefix, c)
    state = payload["state"]
    if change == "version":
        payload["version"] = True
    elif change == "schema":
        payload["schema"] = "unsupported"
    elif change == "unknown":
        payload["extra"] = 0
    elif change == "missing":
        del state["m"]
    elif change == "dtype":
        state["parameters"]["w1"]["dtype"] = "<f4"
    elif change == "shape":
        state["parameters"]["w1"]["shape"] = [1024]
    elif change == "nan":
        state["parameters"]["w1"]["values"][0][0] = "NaN"
    elif change == "coercion":
        state["parameters"]["w1"]["values"][0][0] = True
    elif change == "negative_v":
        state["v"]["w1"]["values"][0][0] = -1
    elif change == "rng":
        state["rng"]["state"]["state"] += 1
    elif change == "counter":
        state["step"] = 0
    elif change == "mean":
        state["mean"][0] += 1
    elif change == "trace":
        state["train_mse_trace"] = []
    elif change == "identity":
        payload["identity"]["config_sha256"] = "0" * 64
    elif change == "schedule":
        payload["identity"]["schedule"][0][0].reverse()
    elif change == "environment":
        payload["identity"]["environment"]["numpy"] = "other"
    else:
        payload["identity"]["source_sha256"]["m221"] = "0" * 64
    data = canonical({"payload": payload, "payload_sha256": digest(payload)})
    path = tmp_path / "bad.json"
    path.write_bytes(data)
    before = original.snapshot_digest(prefix)
    with pytest.raises(CheckpointError):
        bridge.resume(path, c, hashlib.sha256(data).hexdigest())
    assert original.snapshot_digest(prefix) == before


@pytest.mark.parametrize("change", ["value", "row_order", "seed", "scenario"])
def test_changed_identity_rejected(change, tmp_path):
    c, prefix, data = fixture()
    path = tmp_path / "prefix.json"
    trusted = bridge.save(path, prefix, c)
    x, y = (a.copy() for a in data["train"])
    scenario, seed = "scarce_noisy", 0
    if change == "value":
        y[0] += 1
    elif change == "row_order":
        x, y = x[::-1].copy(), y[::-1].copy()
    elif change == "seed":
        seed = 1
    else:
        scenario = "ample_clean"
        config = bridge.registered_config()
        x, y = original.generate(config, config["scenarios"][0], 0)["train"]
    with pytest.raises(CheckpointError):
        bridge.resume(path, bridge.Contract(x, y, scenario, seed), trusted)


def test_corruption_duplicate_keys_truncation_and_size(tmp_path):
    c, prefix, _ = fixture()
    path = tmp_path / "prefix.json"
    trusted = bridge.save(path, prefix, c)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(CheckpointError):
        bridge.load(path, c, trusted)
    for raw in (
        b"{",
        b'{"payload":0,"payload":1,"payload_sha256":"x"}',
        b'{"payload":{},"payload_sha256":"wrong"}',
        b'{"payload":NaN}',
    ):
        path.write_bytes(raw)
        with pytest.raises(CheckpointError):
            bridge.load(path, c, hashlib.sha256(raw).hexdigest())
    path.write_bytes(b" " * 2_000_001)
    with pytest.raises(CheckpointError, match="size"):
        bridge.load(path, c, "0" * 64)


def test_optimizer_reset_rejected(tmp_path):
    c, prefix, _ = fixture()
    reset = copy.deepcopy(prefix)
    reset["step"] = 0
    for kind in ("m", "v"):
        for value in reset[kind].values():
            value.fill(0)
    with pytest.raises(CheckpointError, match="Counter"):
        bridge.save(tmp_path / "reset.json", reset, c)


def test_heldout_isolation_and_contract_mutation(tmp_path):
    c, prefix, data = fixture()
    path = tmp_path / "prefix.json"
    trusted = bridge.save(path, prefix, c)
    before = original.snapshot_digest(prefix)
    first = bridge.resume(path, c, trusted)
    for name in ("validation", "test"):
        x, y = data[name]
        x[:] = 1e5
        y[:] = -1e5
        original.metrics(prefix, x, y)
    bridge.require_equal(first, bridge.resume(path, c, trusted))
    assert original.snapshot_digest(prefix) == before
    c.x[0, 0] += 1
    with pytest.raises(CheckpointError):
        bridge.resume(path, c, trusted)


def test_cold_process_and_optimized_rejection(tmp_path):
    c, prefix, data = fixture()
    path = tmp_path / "prefix.json"
    trusted = bridge.save(path, prefix, c)
    code = """
import sys
from proteinmod import synthetic_bridge as b, synthetic_regression as o
c=b.registered_config()
d=o.generate(c,c['scenarios'][1],0)
contract=b.Contract(*d['train'],'scarce_noisy',0)
print(o.snapshot_digest(b.resume(sys.argv[1],contract,sys.argv[2])))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), trusted], capture_output=True, text=True, check=True
    )
    expected = original.fit(c.config, *data["train"], 0, stop=120, resume=prefix)[120]
    assert result.stdout.strip() == original.snapshot_digest(expected)
    code += """
try:
 b.resume(sys.argv[1],contract,'0'*64)
except b.CheckpointError:
 pass
else:
 raise RuntimeError('Optimized validation failed')
"""
    subprocess.run(
        [sys.executable, "-O", "-c", code, str(path), trusted],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )


def test_schedule_is_the_original_permutation_not_grouped():
    c, _, _ = fixture("ample_clean")
    rng = original.generator(c.config, 300, 0)
    for batches in c.schedule:
        expected = rng.permutation(512)
        np.testing.assert_array_equal(np.concatenate(batches), expected)
        assert [len(batch) for batch in batches] == [32] * 16
    assert canonical(rng.bit_generator.state) == canonical(c.rng_states[120])


def test_collection_guard_preserves_parent():
    from proteinmod.m225_collection import validate_changes

    validate_changes(["A\tsrc/proteinmod/synthetic_bridge.py"])
    for changes in ([], ["M\tconftest.py"], ["A\tunrelated.py"]):
        with pytest.raises(ValueError):
            validate_changes(changes)


def test_changed_contract_configuration_rejected(tmp_path):
    c, prefix, _ = fixture()
    path = tmp_path / "prefix.json"
    trusted = bridge.save(path, prefix, c)
    c.config["batch_size"] = 16
    with pytest.raises(CheckpointError, match="contract"):
        bridge.resume(path, c, trusted)
