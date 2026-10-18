"""Prespecified numeric screens; no cross-device bitwise guarantee."""

import numpy as np


def activation_comparison(a, b, depth=12, allowance=32):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need matched finite nonempty tensors")
    budget = allowance * depth * np.finfo(np.float32).eps * (1 + np.maximum(abs(a), abs(b)))
    error = abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "exact": bool(np.array_equal(a, b)),
        "maximum_absolute_difference": float(error.max()),
        "maximum_budget_ratio": float((error / budget).max()),
        "roundoff_screen_pass": bool((error <= budget).all()),
        "legacy_1e-6_pass": bool((error <= 1e-6).all()),
    }
