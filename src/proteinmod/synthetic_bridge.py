"""M22.5: strict serialization boundary around the unchanged M22.1 numeric trainer."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from proteinmod import synthetic_regression as original
from proteinmod.synthetic_checkpoint import (
    CheckpointError,
    canonical,
    digest,
    environment,
    keys,
    pack,
    unique_pairs,
    unpack,
)

CONFIG_SHA256 = "8c5a97563a8f13e5dbd80c88095ced96af28f8fc9ec5fce6e2cdd2670d66c231"
ROOT = Path(__file__).resolve().parents[2]
STATE_KEYS = {
    "parameters",
    "m",
    "v",
    "epoch",
    "step",
    "mean",
    "scale",
    "seed",
    "config_sha256",
    "training_sha256",
    "train_mse_trace",
    "rng",
}


def registered_config():
    data = (ROOT / "configs/m221_synthetic.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != CONFIG_SHA256:
        raise CheckpointError("Registered config bytes changed")
    return json.loads(data)


class Contract:
    """Trusted expected identity, containing training-only arrays; no evaluation API."""

    def __init__(self, x, y, scenario, seed):
        self.config = registered_config()
        scenarios = {s["id"]: s for s in self.config["scenarios"]}
        if (
            scenario not in scenarios
            or type(seed) is not int
            or seed not in range(10)
            or not isinstance(x, np.ndarray)
            or not isinstance(y, np.ndarray)
            or x.dtype.str != "<f8"
            or y.dtype.str != "<f8"
            or x.shape != (scenarios[scenario]["n_train"], 16)
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise CheckpointError("Invalid registered training input/scenario/seed")
        self.x, self.y, self.seed = x.copy(), y.copy(), seed
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        rng = original.generator(self.config, 300, seed)
        self.schedule, self.rng_states = [], [copy.deepcopy(rng.bit_generator.state)]
        for _ in range(120):
            order = rng.permutation(len(x))
            self.schedule.append([order[i : i + 32].tolist() for i in range(0, len(x), 32)])
            self.rng_states.append(copy.deepcopy(rng.bit_generator.state))
        self.identity = {
            "scenario": scenario,
            "seed": seed,
            "config_sha256": CONFIG_SHA256,
            "config_fingerprint": original.fingerprint(self.config),
            "training_sha256": original.array_digest(x, y),
            "input_shape": [list(x.shape), list(y.shape)],
            "dtype": "<f8",
            "preprocessing": {"mean": pack(self.mean), "scale": pack(self.scale)},
            "schedule": copy.deepcopy(self.schedule),
            "schedule_sha256": digest(self.schedule),
            "sampler": "original M22.1 PCG64.permutation then contiguous chunks32; no group shuffle",
            "source_sha256": {
                name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in [("m221", original)]
            },
            "bridge_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "m223_sha256": hashlib.sha256(
                (ROOT / "src/proteinmod/synthetic_checkpoint.py").read_bytes()
            ).hexdigest(),
            "environment": environment(),
            "initial_parameters_sha256": original.array_digest(
                *original.initialize(self.config, seed).values()
            ),
        }

    @property
    def shapes(self):
        return {"w1": (16, 64), "b1": (64,), "w2": (64,), "b2": (1,)}

    def check(self):
        if (
            original.array_digest(self.x, self.y) != self.identity["training_sha256"]
            or original.fingerprint(self.config) != self.identity["config_fingerprint"]
            or digest(self.schedule) != self.identity["schedule_sha256"]
            or pack(self.mean) != self.identity["preprocessing"]["mean"]
            or pack(self.scale) != self.identity["preprocessing"]["scale"]
        ):
            raise CheckpointError("Trusted contract changed")


def validate(state, contract):
    contract.check()
    keys(state, STATE_KEYS)
    epoch = state["epoch"]
    if type(epoch) is not int or epoch not in (12, 120):
        raise CheckpointError("Only registered completed checkpoint epochs supported")
    if (
        type(state["step"]) is not int
        or state["step"] != epoch * len(contract.schedule[0])
        or type(state["seed"]) is not int
        or state["seed"] != contract.seed
        or state["config_sha256"] != contract.identity["config_fingerprint"]
        or state["training_sha256"] != contract.identity["training_sha256"]
    ):
        raise CheckpointError("Counter/config/input/seed mismatch")
    if canonical(state["rng"]) != canonical(contract.rng_states[epoch]):
        raise CheckpointError("RNG/schedule boundary mismatch")
    for kind in ("parameters", "m", "v"):
        keys(state[kind], set(contract.shapes))
        for key, shape in contract.shapes.items():
            value = state[kind][key]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype.str != "<f8"
                or value.shape != shape
                or not np.isfinite(value).all()
                or (kind == "v" and (value < 0).any())
            ):
                raise CheckpointError("Invalid model/optimizer dtype/shape/value")
    for key, expected in (("mean", contract.mean), ("scale", contract.scale)):
        values = state[key]
        if (
            not isinstance(values, list)
            or len(values) != 16
            or any(type(v) is not float or not np.isfinite(v) for v in values)
            or canonical(values) != canonical(expected.tolist())
        ):
            raise CheckpointError("Preprocessing mismatch")
    trace = state["train_mse_trace"]
    if (
        not isinstance(trace, list)
        or len(trace) != epoch
        or any(type(v) is not float or not np.isfinite(v) or v < 0 for v in trace)
    ):
        raise CheckpointError("Invalid training trace")


def encode(state, contract):
    validate(state, contract)
    encoded = copy.deepcopy(state)
    for kind in ("parameters", "m", "v"):
        encoded[kind] = {key: pack(value) for key, value in state[kind].items()}
    return {
        "schema": "proteinmod.m225.m221-envelope",
        "version": 1,
        "identity": copy.deepcopy(contract.identity),
        "state": encoded,
    }


def decode(payload, contract):
    keys(payload, {"schema", "version", "identity", "state"})
    if (
        payload["schema"] != "proteinmod.m225.m221-envelope"
        or type(payload["version"]) is not int
        or payload["version"] != 1
        or canonical(payload["identity"]) != canonical(contract.identity)
    ):
        raise CheckpointError("Schema/version/trusted identity mismatch")
    state = copy.deepcopy(payload["state"])
    keys(state, STATE_KEYS)
    for kind in ("parameters", "m", "v"):
        keys(state[kind], set(contract.shapes))
        state[kind] = {
            key: unpack(value, contract.shapes[key]) for key, value in state[kind].items()
        }
    validate(state, contract)
    return state


def save(path, state, contract):
    payload = encode(state, contract)
    data = canonical({"payload": payload, "payload_sha256": digest(payload)})
    with Path(path).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def load(path, contract, expected_sha256):
    if Path(path).stat().st_size > 2_000_000:
        raise CheckpointError("Checkpoint size limit exceeded")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise CheckpointError("Trusted file SHA256 mismatch")
    try:
        envelope = json.loads(
            data,
            object_pairs_hook=unique_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(CheckpointError("Nonfinite JSON")),
        )
        keys(envelope, {"payload", "payload_sha256"})
        if digest(envelope["payload"]) != envelope["payload_sha256"]:
            raise CheckpointError("Payload checksum mismatch")
        return decode(envelope["payload"], contract)
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError) as error:
        raise CheckpointError(f"Invalid checkpoint: {error}") from error


def resume(path, contract, expected_sha256):
    state = load(path, contract, expected_sha256)
    if state["epoch"] != 12:
        raise CheckpointError("Continuation requires epoch 12; terminal checkpoint cannot advance")
    result = original.fit(
        contract.config, contract.x, contract.y, contract.seed, stop=120, resume=state
    )[120]
    validate(result, contract)
    return result


def require_equal(first, second):
    if canonical(
        {k: v for k, v in first.items() if k not in ("parameters", "m", "v")}
    ) != canonical({k: v for k, v in second.items() if k not in ("parameters", "m", "v")}):
        raise CheckpointError("Metadata/RNG/trace mismatch")
    for kind in ("parameters", "m", "v"):
        for key in original.PARAMETERS:
            if (
                first[kind][key].dtype != second[kind][key].dtype
                or first[kind][key].shape != second[kind][key].shape
                or first[kind][key].tobytes() != second[kind][key].tobytes()
            ):
                raise CheckpointError("Bitwise model/optimizer mismatch")
