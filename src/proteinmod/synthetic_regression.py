"""M22.1: fixed synthetic arrays and a small NumPy MLP; no biological imports."""

import copy
import hashlib
import json

import numpy as np
from scipy.stats import rankdata, t

PARAMETERS = ("w1", "b1", "w2", "b2")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def generator(config, namespace, seed, *extra):
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([config["root_seed"], namespace, seed, *extra]))
    )


def generate(config, scenario, seed):
    """Independent split generation; no dataset files, sequences or model outputs."""
    scenario_id = [s["id"] for s in config["scenarios"]].index(scenario["id"])
    result = {}
    for split_id, name in enumerate(("train", "validation", "test")):
        rng = generator(config, config["seed_namespaces"]["data"], seed, scenario_id, split_id)
        x = rng.normal(size=(scenario["n_" + name], config["input_width"]))
        y = np.sin(x[:, 0]) + 0.5 * x[:, 1]
        y += scenario["noise_sd"] * rng.normal(size=len(x))
        result[name] = (x, y)
    return result


def initialize(config, seed):
    rng = generator(config, config["seed_namespaces"]["initialization"], seed)
    width, hidden = config["input_width"], config["hidden_width"]
    first, second = np.sqrt(6 / (width + hidden)), np.sqrt(6 / (hidden + 1))
    return {
        "w1": rng.uniform(-first, first, (width, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.uniform(-second, second, hidden),
        "b2": np.zeros(1),
    }


def predict(parameters, x):
    return np.tanh(x @ parameters["w1"] + parameters["b1"]) @ parameters["w2"] + parameters["b2"][0]


def loss_gradient(parameters, x, y):
    hidden = np.tanh(x @ parameters["w1"] + parameters["b1"])
    residual = hidden @ parameters["w2"] + parameters["b2"][0] - y
    derivative = 2 * residual / len(x)
    local = derivative[:, None] * parameters["w2"] * (1 - hidden**2)
    return float(np.mean(residual**2)), {
        "w1": x.T @ local,
        "b1": local.sum(axis=0),
        "w2": hidden.T @ derivative,
        "b2": np.array([derivative.sum()]),
    }


def snapshot_digest(state):
    metadata = {key: value for key, value in state.items() if key not in {"parameters", "m", "v"}}
    return fingerprint(
        {
            "metadata": metadata,
            "arrays": {
                group: array_digest(*(state[group][name] for name in PARAMETERS))
                for group in ("parameters", "m", "v")
            },
        }
    )


def fit(config, x_train, y_train, seed, stop=120, resume=None):
    """Training accepts only training arrays; evaluation cannot affect its state."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if (
        x_train.ndim != 2
        or x_train.shape[1] != config["input_width"]
        or y_train.shape != (len(x_train),)
        or len(x_train) < 2
        or not np.isfinite(x_train).all()
        or not np.isfinite(y_train).all()
    ):
        raise ValueError("Finite synthetic training matrix/vector required")
    identity = array_digest(x_train, y_train)
    if resume is None:
        parameters = initialize(config, seed)
        mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
        scale[scale == 0] = 1.0
        state = {
            "parameters": parameters,
            "m": {key: np.zeros_like(value) for key, value in parameters.items()},
            "v": {key: np.zeros_like(value) for key, value in parameters.items()},
            "epoch": 0,
            "step": 0,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "seed": seed,
            "config_sha256": fingerprint(config),
            "training_sha256": identity,
            "train_mse_trace": [],
            "rng": generator(
                config, config["seed_namespaces"]["batch_order"], seed
            ).bit_generator.state,
        }
    else:
        state = copy.deepcopy(resume)
        if (
            state["config_sha256"] != fingerprint(config)
            or state["training_sha256"] != identity
            or state["seed"] != seed
        ):
            raise ValueError("Resume config/training/seed identity mismatch")
    if not state["epoch"] < stop <= config["epochs"][-1]:
        raise ValueError("Stop must advance within the registered horizon")
    x = (x_train - state["mean"]) / state["scale"]
    rng = generator(config, config["seed_namespaces"]["batch_order"], seed)
    rng.bit_generator.state = state["rng"]
    settings = config["optimizer"]
    beta1, beta2 = settings["beta1"], settings["beta2"]
    checkpoints = {}
    for epoch in range(state["epoch"] + 1, stop + 1):
        order = rng.permutation(len(x))
        for offset in range(0, len(x), config["batch_size"]):
            indices = order[offset : offset + config["batch_size"]]
            loss, gradients = loss_gradient(state["parameters"], x[indices], y_train[indices])
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
                raise ValueError("Nonfinite synthetic optimization; no replacement run permitted")
            state["step"] += 1
            for name in PARAMETERS:
                state["m"][name] *= beta1
                state["m"][name] += (1 - beta1) * gradients[name]
                state["v"][name] *= beta2
                state["v"][name] += (1 - beta2) * gradients[name] ** 2
                mean = state["m"][name] / (1 - beta1 ** state["step"])
                variance = state["v"][name] / (1 - beta2 ** state["step"])
                state["parameters"][name] -= (
                    settings["learning_rate"] * mean / (np.sqrt(variance) + settings["epsilon"])
                )
        state["epoch"] = epoch
        state["rng"] = copy.deepcopy(rng.bit_generator.state)
        fitted = predict(state["parameters"], x)
        state["train_mse_trace"].append(float(np.mean((fitted - y_train) ** 2)))
        if epoch in config["epochs"] or epoch == stop:
            checkpoints[epoch] = copy.deepcopy(state)
    return checkpoints


def metrics(state, x, y):
    fitted = predict(state["parameters"], (x - state["mean"]) / state["scale"])
    if not np.isfinite(fitted).all():
        raise ValueError("Nonfinite checkpoint prediction")
    rank_x, rank_y = rankdata(fitted), rankdata(y)
    rho = (
        None
        if np.ptp(rank_x) == 0 or np.ptp(rank_y) == 0
        else float(np.corrcoef(rank_x, rank_y)[0, 1])
    )
    return {"mse": float(np.mean((fitted - y) ** 2)), "spearman": rho}


def save_checkpoint(path, state):
    metadata = {key: value for key, value in state.items() if key not in {"parameters", "m", "v"}}
    arrays = {
        f"{group}_{name}": state[group][name]
        for group in ("parameters", "m", "v")
        for name in PARAMETERS
    }
    with path.open("xb") as handle:
        np.savez(handle, metadata=np.array(json.dumps(metadata, allow_nan=False)), **arrays)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        state = json.loads(str(archive["metadata"]))
        for group in ("parameters", "m", "v"):
            state[group] = {name: archive[f"{group}_{name}"].copy() for name in PARAMETERS}
    return state


def paired_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent seed pairs required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    radius = float(t.ppf(0.975, len(values) - 1) * se)
    return {
        "n": len(values),
        "mean": mean,
        "sample_sd": sd,
        "standard_error": float(se),
        "ci95": [mean - radius, mean + radius],
        "positive": int(np.sum(values > 0)),
        "negative": int(np.sum(values < 0)),
        "zero": int(np.sum(values == 0)),
    }


def summarize(records):
    summaries = {}
    for name in sorted({row["scenario"] for row in records}):
        selected = [row for row in records if row["scenario"] == name]
        effects = {}
        for split in ("train", "validation", "test"):
            effects[split] = {}
            for metric in ("mse", "spearman"):
                values = [row["paired_effects"][split][metric] for row in selected]
                effects[split][metric] = (
                    None if any(value is None for value in values) else paired_statistics(values)
                )
        test = effects["test"]["mse"]
        expected = (
            test["negative"] >= 8 and test["ci95"][1] < 0
            if name == "ample_clean"
            else test["positive"] >= 8
            and test["ci95"][0] > 0
            and effects["train"]["mse"]["mean"] < 0
        )
        summaries[name] = {"effects": effects, "prespecified_expectation_met": bool(expected)}
    return summaries
