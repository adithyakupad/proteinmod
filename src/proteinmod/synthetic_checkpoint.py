"""M22.3: numeric-only, epoch-boundary Adam replay; no historical trainer imports."""

import copy
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

VERSION = 1
CONFIG = {
    "features": 6,
    "hidden": 12,
    "batch_size": 8,
    "epochs": 120,
    "learning_rate": 0.003,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "dtype": "<f8",
    "activation": "tanh",
    "optimizer": "Adam",
    "dropout": None,
    "scheduler": None,
    "early_stopping": None,
}
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class CheckpointError(ValueError):
    """Untrusted or incompatible checkpoint; no training may proceed."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_build": str(np.show_config(mode="dicts")),
        "threads": {key: os.environ.get(key) for key in THREADS},
    }


def pack(array):
    return {"dtype": array.dtype.str, "shape": list(array.shape), "values": array.tolist()}


def unpack(value, shape):
    keys(value, {"dtype", "shape", "values"})
    if (
        value["dtype"] != "<f8"
        or value["shape"] != list(shape)
        or not isinstance(value["shape"], list)
        or any(type(n) is not int for n in value["shape"])
    ):
        raise CheckpointError("Array dtype/shape mismatch")

    def numeric_tree(item):
        if isinstance(item, list):
            return all(numeric_tree(child) for child in item)
        return type(item) in (int, float)

    if not numeric_tree(value["values"]):
        raise CheckpointError("Array values must be numeric, without coercion")
    try:
        array = np.array(value["values"], dtype="<f8")
    except (ValueError, TypeError, OverflowError) as error:
        raise CheckpointError("Invalid numeric array") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise CheckpointError("Array nonfinite or incorrectly shaped")
    return array


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != expected:
        raise CheckpointError("Unknown or missing schema fields")


def numeric_data(seed):
    """Artificial normal numbers only; groups deliberately have ragged remainders."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([223, seed, 1])))
    x = rng.normal(size=(59, 6)).astype("<f8")
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] + rng.normal(0, 0.1, 59)).astype("<f8")
    groups = ["a"] * 17 + ["b"] * 33 + ["c"] * 9
    held = rng.normal(size=(100, 6)).astype("<f8")
    target = (np.sin(held[:, 0]) + 0.5 * held[:, 1]).astype("<f8")
    return x, y, groups, held, target


def order_rng(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([223, seed, 3])))


def epoch_batches(rng, groups):
    batches = []
    for group in sorted(set(groups)):
        indices = np.flatnonzero(np.array(groups) == group)
        indices = rng.permutation(indices)
        batches.extend(
            indices[i : i + CONFIG["batch_size"]].tolist()
            for i in range(0, len(indices), CONFIG["batch_size"])
        )
    rng.shuffle(batches)
    return batches


class Contract:
    """Trusted caller-side identity. Held-out arrays are intentionally absent."""

    def __init__(self, x, y, groups, seed):
        if (
            x.dtype.str != "<f8"
            or y.dtype.str != "<f8"
            or x.ndim != 2
            or x.shape[1] != CONFIG["features"]
            or y.shape != (len(x),)
            or not len(x)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or len(groups) != len(x)
            or any(type(g) is not str or not g for g in groups)
            or type(seed) is not int
            or not 0 <= seed < 2**32
        ):
            raise CheckpointError("Invalid training input")
        self.x, self.y, self.groups, self.seed = x.copy(), y.copy(), list(groups), seed
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1
        rng = order_rng(seed)
        self.rng_states = [copy.deepcopy(rng.bit_generator.state)]
        self.schedule = []
        for _ in range(CONFIG["epochs"]):
            self.schedule.append(epoch_batches(rng, groups))
            self.rng_states.append(copy.deepcopy(rng.bit_generator.state))
        self.identity = {
            "config": CONFIG.copy(),
            "seed": seed,
            "input_sha256": digest({"x": pack(x), "y": pack(y), "groups": groups}),
            "schedule_sha256": digest(self.schedule),
            "batch_schedule": copy.deepcopy(self.schedule),
            "sampler_version": "group-sorted/per-group-permute/chunk8/shuffle-batches/v1",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": environment(),
            "preprocessing": {"mean": pack(self.mean), "scale": pack(self.scale)},
            "group_order": sorted(set(groups)),
            "row_identity": "ordered input rows",
            "checkpoint_boundary": "after update, epoch complete, before next permutation",
        }

    def verify_input(self):
        current = digest({"x": pack(self.x), "y": pack(self.y), "groups": self.groups})
        if (
            current != self.identity["input_sha256"]
            or digest(self.schedule) != self.identity["schedule_sha256"]
            or pack(self.mean) != self.identity["preprocessing"]["mean"]
            or pack(self.scale) != self.identity["preprocessing"]["scale"]
            or CONFIG != self.identity["config"]
        ):
            raise CheckpointError("Contract input/schedule changed")

    @property
    def shapes(self):
        return {"w1": (6, 12), "b1": (12,), "w2": (12,), "b2": (1,)}


def initial(contract):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([223, contract.seed, 2])))
    params = {
        key: rng.normal(0, 0.2, shape).astype("<f8")
        if key.startswith("w")
        else np.zeros(shape, dtype="<f8")
        for key, shape in contract.shapes.items()
    }
    return {
        "schema": "proteinmod.synthetic.checkpoint",
        "version": VERSION,
        "identity": copy.deepcopy(contract.identity),
        "epoch": 0,
        "updates": 0,
        "optimizer_step": 0,
        "rng": copy.deepcopy(contract.rng_states[0]),
        "initialization_rng": copy.deepcopy(rng.bit_generator.state),
        "parameters": {key: pack(value) for key, value in params.items()},
        "m": {key: pack(np.zeros_like(value)) for key, value in params.items()},
        "v": {key: pack(np.zeros_like(value)) for key, value in params.items()},
    }


def validate(state, contract):
    contract.verify_input()
    keys(
        state,
        {
            "schema",
            "version",
            "identity",
            "epoch",
            "updates",
            "optimizer_step",
            "rng",
            "initialization_rng",
            "parameters",
            "m",
            "v",
        },
    )
    if (
        state["schema"] != "proteinmod.synthetic.checkpoint"
        or type(state["version"]) is not int
        or state["version"] != VERSION
        or canonical(state["identity"]) != canonical(contract.identity)
    ):
        raise CheckpointError("Schema/version or trusted identity mismatch")
    epoch = state["epoch"]
    if type(epoch) is not int or not 0 <= epoch <= CONFIG["epochs"]:
        raise CheckpointError("Invalid epoch boundary")
    updates = sum(len(batch) for batch in contract.schedule[:epoch])
    if (
        type(state["updates"]) is not int
        or state["updates"] != updates
        or type(state["optimizer_step"]) is not int
        or state["optimizer_step"] != updates
    ):
        raise CheckpointError("Inconsistent update/optimizer counters")
    if state["rng"] != contract.rng_states[epoch]:
        raise CheckpointError("RNG/schedule cursor mismatch")
    if state["initialization_rng"] != initial(contract)["initialization_rng"]:
        raise CheckpointError("Initialization RNG mismatch")
    for kind in ("parameters", "m", "v"):
        keys(state[kind], set(contract.shapes))
        for key, shape in contract.shapes.items():
            value = unpack(state[kind][key], shape)
            if kind == "v" and (value < 0).any():
                raise CheckpointError("Negative second moment")


def prediction(parameters, x):
    hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
    return hidden @ parameters["w2"] + parameters["b2"][0]


def update(parameters, moments, squares, step, x, y):
    hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
    residual = hidden @ parameters["w2"] + parameters["b2"][0] - y
    output_gradient = 2 * residual / len(y)
    hidden_gradient = output_gradient[:, None] * parameters["w2"] * (1 - hidden**2)
    gradients = {
        "w1": x.T @ hidden_gradient,
        "b1": hidden_gradient.sum(axis=0),
        "w2": hidden.T @ output_gradient,
        "b2": np.array([output_gradient.sum()]),
    }
    for key in parameters:
        moments[key] = 0.9 * moments[key] + 0.1 * gradients[key]
        squares[key] = 0.999 * squares[key] + 0.001 * gradients[key] ** 2
        parameters[key] -= (
            0.003
            * (moments[key] / (1 - 0.9**step))
            / (np.sqrt(squares[key] / (1 - 0.999**step)) + 1e-8)
        )


def train(contract, until, state=None):
    """No checkpoint selection, evaluation callback, or held-out input accepted."""
    state = initial(contract) if state is None else copy.deepcopy(state)
    validate(state, contract)
    if type(until) is not int or not state["epoch"] <= until <= CONFIG["epochs"]:
        raise CheckpointError("Invalid fixed horizon")
    arrays = {
        kind: {key: unpack(value, contract.shapes[key]) for key, value in state[kind].items()}
        for kind in ("parameters", "m", "v")
    }
    x = (contract.x - contract.mean) / contract.scale
    rng = order_rng(contract.seed)
    rng.bit_generator.state = copy.deepcopy(state["rng"])
    for epoch in range(state["epoch"], until):
        batches = epoch_batches(rng, contract.groups)
        if batches != contract.schedule[epoch]:
            raise CheckpointError("Schedule mismatch")
        for indices in batches:
            state["updates"] += 1
            update(
                arrays["parameters"],
                arrays["m"],
                arrays["v"],
                state["updates"],
                x[indices],
                contract.y[indices],
            )
        state["epoch"] = epoch + 1
        state["optimizer_step"] = state["updates"]
        state["rng"] = copy.deepcopy(rng.bit_generator.state)
    for kind, values in arrays.items():
        state[kind] = {key: pack(value) for key, value in values.items()}
    validate(state, contract)
    return state


def save(path, state, contract):
    validate(state, contract)
    data = canonical({"payload": state, "payload_sha256": digest(state)})
    with Path(path).open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("Duplicate JSON key")
        result[key] = value
    return result


def load(path, contract, expected_sha256):
    """The expected file hash must come from a trusted record, not the file itself."""
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
        validate(envelope["payload"], contract)
    except (
        TypeError,
        KeyError,
        UnicodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        raise CheckpointError("Malformed checkpoint") from error
    return envelope["payload"]


def evaluate(state, contract, x, y):
    from scipy.stats import spearmanr

    validate(state, contract)
    params = {
        key: unpack(value, contract.shapes[key]) for key, value in state["parameters"].items()
    }
    predicted = prediction(params, (x - contract.mean) / contract.scale)
    return {
        "mse": float(np.mean((predicted - y) ** 2)),
        "spearman": (
            float(spearmanr(predicted, y).statistic)
            if np.ptp(predicted) > 0 and np.ptp(y) > 0
            else None
        ),
    }
