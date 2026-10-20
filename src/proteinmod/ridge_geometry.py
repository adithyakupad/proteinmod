"""Train-only ridge geometry and deterministic validation selection for M19."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from proteinmod.multiprotein_experiments import fit_targets, macro_score


def fit(matrix, rows, roles, penalties, scaled, normalized, shuffle_seed=None):
    train = roles["train"]
    target = fit_targets(rows, train)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for assay in sorted({rows[i]["assay_id"] for i in train}):
            indices = [i for i in train if rows[i]["assay_id"] == assay]
            target[indices] = rng.permutation(target[indices])
    scaler = StandardScaler().fit(matrix[train]) if scaled else None
    features = scaler.transform(matrix).astype(float) if scaled else matrix.astype(float)
    centered_train = features[train] - features[train].mean(axis=0)
    energy = float(np.sum(centered_train**2) / len(train))
    divisor = np.sqrt(energy) if normalized else 1.0
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Feature energy must be finite and positive")
    features = features / divisor
    candidates = []
    best = None
    for penalty in penalties:
        model = Ridge(alpha=penalty, solver="lsqr", tol=1e-8, max_iter=10000)
        model.fit(features[train], target[train])
        score = macro_score(rows, roles["validation"], model.predict(features[roles["validation"]]))
        candidates.append({"penalty": penalty, "validation_rho": score})
        if best is None or score > best[0]:
            best = score, penalty, model
    score, penalty, model = best
    state = {
        "mean": scaler.mean_ if scaled else np.zeros(matrix.shape[1]),
        "scale": scaler.scale_ if scaled else np.ones(matrix.shape[1]),
        "divisor": np.asarray(divisor),
        "coefficient": model.coef_,
        "intercept": np.asarray(model.intercept_),
    }
    selection = {
        "penalty": penalty,
        "effective_alpha": penalty * divisor**2,
        "energy": energy,
        "validation_rho": score,
        "candidates": candidates,
        "scaled": scaled,
        "normalized": normalized,
        "parameters": matrix.shape[1] + 1,
    }
    return state, selection


def predict(matrix, state):
    features = (matrix.astype(float) - state["mean"]) / state["scale"]
    features = features / state["divisor"]
    return features @ state["coefficient"] + state["intercept"]
