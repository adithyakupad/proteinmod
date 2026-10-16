"""Explicit epistemic sensitivity, finite-sample residual quantiles and UQ diagnostics."""

import hashlib

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def ensemble_summary(members, eligible=None):
    values = np.asarray(members, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("At least two aligned ensemble members required")
    mask = np.ones(values.shape[1], bool) if eligible is None else np.asarray(eligible)
    if mask.shape != (values.shape[1],) or mask.dtype != bool:
        raise ValueError("Boolean prediction eligibility mask required")
    if not np.isfinite(values[:, mask]).all():
        raise ValueError("Eligible predictions must be finite")
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    return {
        "point_prediction": [float(v) if m else None for v, m in zip(mean, mask, strict=True)],
        "uncertainty": [float(v) if m else None for v, m in zip(sd, mask, strict=True)],
        "eligible": mask.tolist(),
        "uncertainty_type": "epistemic_training_sensitivity_not_posterior",
        "members": len(values),
        "aleatoric_variance": None,
        "structure_confidence_is_predictive_uncertainty": False,
    }


def conformal_quantile(scores, level):
    scores = np.asarray(scores, dtype=float)
    if not 0 < level < 1 or scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Finite nonnegative scores and nominal level in (0,1) required")
    k = int(np.ceil((len(scores) + 1) * level))
    return float(np.sort(scores)[k - 1]) if len(scores) and k <= len(scores) else float("inf")


def interval(point, scale, quantile, unit, coverage_claim=False):
    point, scale = np.asarray(point, float), np.asarray(scale, float)
    if (
        point.shape != scale.shape
        or not np.isfinite(point).all()
        or not np.isfinite(scale).all()
        or (scale < 0).any()
        or quantile < 0
        or np.isnan(quantile)
    ):
        raise ValueError("Ordered finite point/scale inputs required")
    finite = np.isfinite(quantile)
    return {
        "lower": (point - quantile * scale).tolist() if finite else [None] * len(point),
        "upper": (point + quantile * scale).tolist() if finite else [None] * len(point),
        "status": "finite" if finite else "unbounded_insufficient_calibration_units",
        "unit": unit,
        "coverage_guarantee_claimed": bool(coverage_claim),
    }


def bootstrap_indices(rows, train, seed):
    """Within-protein genotype bootstrap, conditional on this fixed training cohort."""
    rng = np.random.default_rng(seed)
    train = np.asarray(train, dtype=int)
    if not len(train) or len(set(train)) != len(train):
        raise ValueError("Unique nonempty training IDs required")
    return np.concatenate(
        [
            rng.choice(ix, size=len(ix), replace=True)
            for p in sorted({rows[i]["protein_id"] for i in train})
            if len(ix := np.array([i for i in train if rows[i]["protein_id"] == p]))
        ]
    )


def training_distance(features, train):
    features = np.asarray(features, float)
    train = np.asarray(train, int)
    if features.ndim != 2 or not np.isfinite(features).all() or not len(train):
        raise ValueError("Finite features and training examples required")
    scaled = StandardScaler().fit(features[train]).transform(features)
    return NearestNeighbors(n_neighbors=1).fit(scaled[train]).kneighbors(
        scaled, return_distance=True
    )[0][:, 0] / np.sqrt(features.shape[1])


def shuffled(values, groups, seed):
    result = np.array(values, copy=True)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    if len(result) != len(groups):
        raise ValueError("Shuffle groups must align")
    for group in sorted(set(groups)):
        ix = np.flatnonzero(groups == group)
        result[ix] = result[rng.permutation(ix)]
    return result


def ordinal_error(measured, predicted):
    """Per-variant pairwise discordance; evaluation only, no test-fitted target scale."""
    y, p = np.asarray(measured, float), np.asarray(predicted, float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite paired observations required")
    truth = np.sign(y[:, None] - y[None, :])
    order = np.sign(p[:, None] - p[None, :])
    valid = truth != 0
    loss = (truth * order < 0) + 0.5 * (order == 0)
    return np.divide(
        (loss * valid).sum(1), valid.sum(1), out=np.full(len(y), np.nan), where=valid.sum(1) > 0
    )


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3 or np.ptp(a[finite]) == 0 or np.ptp(b[finite]) == 0:
        return None
    return float(spearmanr(a[finite], b[finite]).statistic)


def interval_metrics(y, lower, upper, level):
    y, lo, hi = np.asarray(y, float), np.asarray(lower, float), np.asarray(upper, float)
    if (
        not 0 < level < 1
        or y.shape != lo.shape
        or y.shape != hi.shape
        or not len(y)
        or not np.isfinite([y, lo, hi]).all()
        or (lo > hi).any()
    ):
        raise ValueError("Finite ordered intervals and aligned targets required")
    width = hi - lo
    alpha = 1 - level
    score = width + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0)
    return {
        "n": len(y),
        "coverage": float(np.mean((lo <= y) & (y <= hi))),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "interval_score": float(score.mean()),
    }


def selective_risk(error, ranking, ids, fractions, seed=0):
    error, ranking = np.asarray(error, float), np.asarray(ranking, float)
    if (
        error.shape != ranking.shape
        or len(ids) != len(error)
        or not np.isfinite(error).all()
        or not np.isfinite(ranking).all()
    ):
        raise ValueError("Finite paired risks and ranking required")
    ties = [hashlib.sha256((str(seed) + str(i)).encode()).hexdigest() for i in ids]
    order = sorted(range(len(ids)), key=lambda i: (ranking[i], ties[i]))
    result = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Retained fraction must be in (0,1]")
        n = max(1, int(np.ceil(len(error) * fraction)))
        ix = order[:n]
        result.append({"retained_fraction": fraction, "n": n, "risk": float(error[ix].mean())})
    return result
