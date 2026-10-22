"""M23 preregistered numeric-only geometry and selection primitives."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t

from proteinmod import synthetic_regression as m22
from proteinmod.synthetic_bridge import require_equal
from proteinmod.synthetic_checkpoint import digest, pack

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("unscaled", "standardized", "energy", "mlp")


def config():
    settings = json.loads((ROOT / "benchmarks/m23/config.json").read_text())
    if (
        hashlib.sha256((ROOT / "benchmarks/m23/PREREGISTRATION.md").read_bytes()).hexdigest()
        != settings["plan_sha256"]
    ):
        raise ValueError("Registered protocol changed")
    return settings


def rng(seed, namespace, role=0):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([23, namespace, seed, role])))


def correlation(p, rho):
    return rho ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])


def generate(seed, cfg):
    beta = np.asarray(cfg["beta"])
    scales = np.logspace(*cfg["scale_log10"], cfg["p"])
    data = {name: {} for name in ("A", "B", "C")}
    for role, (split, n) in enumerate(
        (
            ("train", cfg["n_train"]),
            ("validation", max(cfg["n_validation"])),
            ("test", cfg["n_test"]),
        )
    ):
        base = rng(seed, 100, role).normal(size=(n, cfg["p"]))
        noise = rng(seed, 101, role).normal(size=n)
        for name, scenario_data in data.items():
            rho = cfg["rho"][split] if name == "C" else 0.0
            latent = base @ np.linalg.cholesky(correlation(cfg["p"], rho)).T if rho else base.copy()
            scale = np.ones(cfg["p"]) if name == "A" else scales
            scenario_data[split] = (latent * scale, latent @ beta + noise)
    return data


def preprocessing(x):
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    energy = float(np.mean(np.var(x, axis=0)))
    if not energy > 0 or not np.isfinite(x).all():
        raise ValueError("Invalid training-only preprocessing input")
    return mean, sd, energy


def ridge(x, y, method, penalty):
    mean, sd, energy = preprocessing(x)
    scale = (
        sd
        if method == "standardized"
        else np.full(x.shape[1], np.sqrt(energy))
        if method == "energy"
        else np.ones(x.shape[1])
    )
    lam = penalty * energy if method == "unscaled" else penalty
    z = (x - mean) / scale
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    weight = vt.T @ ((s / (s * s + len(x) * lam)) * (u.T @ (y - y.mean())))
    coefficient = weight / scale
    return {
        "method": method,
        "penalty": float(penalty),
        "coefficient": coefficient,
        "intercept": float(y.mean() - mean @ coefficient),
    }


def mlp(x, y, penalty, seed, cfg):
    mean, sd, _ = preprocessing(x)
    z = (x - mean) / sd
    settings = {
        "root_seed": 23,
        "input_width": cfg["p"],
        "hidden_width": cfg["mlp"]["hidden"],
        "seed_namespaces": {"initialization": 200},
    }
    parameters = m22.initialize(settings, seed)
    # Preserve the M22 Glorot/zero-bias template, with M23's explicit role namespace.
    init_rng = rng(seed, 200)
    for key, fan_in, fan_out in (
        ("w1", cfg["p"], cfg["mlp"]["hidden"]),
        ("w2", cfg["mlp"]["hidden"], 1),
    ):
        bound = np.sqrt(6 / (fan_in + fan_out))
        parameters[key] = init_rng.uniform(-bound, bound, parameters[key].shape)
    moments = {key: np.zeros_like(value) for key, value in parameters.items()}
    squares = copy.deepcopy(moments)
    order_rng = rng(seed, 300)
    step = 0
    centered = y - y.mean()
    for _ in range(cfg["mlp"]["epochs"]):
        order = order_rng.permutation(len(x))
        for start in range(0, len(x), cfg["mlp"]["batch_size"]):
            indices = order[start : start + cfg["mlp"]["batch_size"]]
            _, gradients = m22.loss_gradient(parameters, z[indices], centered[indices])
            for key in ("w1", "w2"):
                gradients[key] += 2 * penalty * parameters[key]
            step += 1
            for key in parameters:
                moments[key] = 0.9 * moments[key] + 0.1 * gradients[key]
                squares[key] = 0.999 * squares[key] + 0.001 * gradients[key] ** 2
                parameters[key] -= (
                    cfg["mlp"]["lr"]
                    * (moments[key] / (1 - 0.9**step))
                    / (np.sqrt(squares[key] / (1 - 0.999**step)) + 1e-8)
                )
    if not all(np.isfinite(value).all() for value in parameters.values()):
        raise ValueError("Nonfinite fixed MLP; no replacement allowed")
    return {
        "method": "mlp",
        "penalty": float(penalty),
        "parameters": parameters,
        "m": moments,
        "v": squares,
        "rng": order_rng.bit_generator.state,
        "step": step,
        "epoch": cfg["mlp"]["epochs"],
        "mean": mean.tolist(),
        "scale": sd.tolist(),
        "y_mean": float(y.mean()),
        "seed": seed,
    }


def fit(x, y, method, penalty, seed, cfg):
    return mlp(x, y, penalty, seed, cfg) if method == "mlp" else ridge(x, y, method, penalty)


def state_digest(state):
    if state["method"] == "mlp":
        return m22.snapshot_digest(state)
    return digest({**state, "coefficient": pack(state["coefficient"])})


def equal(first, second):
    if first["method"] == "mlp":
        require_equal(first, second)
    elif state_digest(first) != state_digest(second):
        raise ValueError("Exact ridge replay failed")


def predict(state, x):
    if state["method"] == "mlp":
        return (
            m22.predict(state["parameters"], (x - state["mean"]) / state["scale"]) + state["y_mean"]
        )
    return x @ state["coefficient"] + state["intercept"]


def metrics(y, prediction):
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    return {
        "mse": float(np.mean((y - prediction) ** 2)),
        "spearman": float(spearmanr(y, prediction).statistic)
        if np.ptp(y) > 0 and np.ptp(prediction) > 0
        else None,
    }


def select(losses, indices):
    best = indices[0]
    for index in indices[1:]:
        if losses[index] < losses[best]:
            best = index
    return int(best)


def geometry(x):
    _, sd, energy = preprocessing(x)
    answer = {}
    for method, scale in (
        ("unscaled", np.ones(x.shape[1])),
        ("standardized", sd),
        ("energy", np.full(x.shape[1], np.sqrt(energy))),
    ):
        z = (x - x.mean(axis=0)) / scale
        singular = np.linalg.svd(z, compute_uv=False)
        probability = singular**2 / np.sum(singular**2)
        variance = z.var(axis=0)
        answer[method] = {
            "effective_rank": float(np.exp(-np.sum(probability * np.log(probability)))),
            "anisotropy": float(variance.max() / variance.min()),
            "condition_number": float(singular[0] / singular[-1]),
            "numerical_rank": int(
                np.sum(singular > np.finfo(float).eps * max(z.shape) * singular[0])
            ),
        }
    return answer


def population(state, scenario, role, cfg):
    if state["method"] == "mlp":
        return None
    scale = np.ones(cfg["p"]) if scenario == "A" else np.logspace(*cfg["scale_log10"], cfg["p"])
    delta = state["coefficient"] * scale - np.asarray(cfg["beta"])
    rho = cfg["rho"][role] if scenario == "C" else 0.0
    return float(1 + state["intercept"] ** 2 + delta @ correlation(cfg["p"], rho) @ delta)


def recovery(state, scenario, cfg):
    if state["method"] == "mlp":
        return None
    scale = np.ones(cfg["p"]) if scenario == "A" else np.logspace(*cfg["scale_log10"], cfg["p"])
    beta = np.asarray(cfg["beta"])
    return float(np.sum((scale * state["coefficient"] - beta) ** 2) / np.sum(beta**2))


def paired(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Finite independent paired units required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    radius = float(t.ppf(1 - alpha / 2, len(values) - 1) * se)
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "se": float(se),
        "ci": [mean - radius, mean + radius],
        "positive": int(np.sum(values > 0)),
        "negative": int(np.sum(values < 0)),
        "alpha": alpha,
    }


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(middle - radius), float(middle + radius)]


def numerical_controls(states, x, y, penalties, cfg):
    _, sd, _ = preprocessing(x)
    maximum = 0.0
    for index, penalty in enumerate(penalties):
        raw = predict(states["unscaled"][index], x)
        normalized = predict(states["energy"][index], x)
        if not np.allclose(raw, normalized, **cfg["numerical"]):
            raise ValueError("Scalar-rescaling numerical gate failed")
        centered = x - x.mean(axis=0)
        beta = np.linalg.solve(
            centered.T @ centered / len(x) + penalty * np.diag(sd**2),
            centered.T @ (y - y.mean()) / len(x),
        )
        generalized = (x - x.mean(axis=0)) @ beta + y.mean()
        standard = predict(states["standardized"][index], x)
        if not np.allclose(generalized, standard, **cfg["numerical"]):
            raise ValueError("Diagonal-penalty equivalence failed")
        maximum = max(
            maximum,
            float(np.max(np.abs(raw - normalized))),
            float(np.max(np.abs(generalized - standard))),
        )
    return maximum
