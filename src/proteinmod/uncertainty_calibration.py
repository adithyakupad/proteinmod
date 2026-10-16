"""Calibration-only scores; test observations are never inputs to interval fitting."""

import numpy as np
from scipy.stats import norm

from proteinmod.uncertainty import conformal_quantile, interval


def calibration_state(y, point, sd, distance, shuffle_sd, rows, roles, config, protocol):
    groups = (
        sorted({rows[i]["protein_id"] for i in roles["calibration"]})
        if protocol in ("G1", "S1")
        else ["pooled"]
    )
    state = {
        "grouping": "protein" if protocol in ("G1", "S1") else "pooled_compatible_native_task",
        "groups": {},
        "unit": "kcal/mol",
        "convention": "positive_destabilizing",
        "coverage_guarantee_claimed": False,
        "interpretation": "Empirical residual calibration; family/protein shift invalidates or leaves exchangeability unverified",
    }
    for group in groups:
        select = lambda role, group=group: np.array(
            [
                i
                for i in roles[role]
                if np.isfinite(y[i]) and (group == "pooled" or rows[i]["protein_id"] == group)
            ],
            int,
        )
        cal, val = select("calibration"), select("validation")
        residual = np.abs(y[cal] - point[cal])
        sd_floor = config["native_scale_floor_kcal_mol"]
        distance_floor = config["distance_floor"]
        scores = {
            "validation_residual": np.abs(y[val] - point[val]),
            "calibration_residual": residual,
            "ensemble_scaled": residual / np.maximum(sd[cal], sd_floor),
            "distance_scaled": residual / np.maximum(distance[cal], distance_floor),
            "shuffled_scaled": residual / np.maximum(shuffle_sd[cal], sd_floor),
        }
        quantiles = {}
        for level in config["nominal_levels"]:
            quantiles[str(level)] = {}
            for method in config["methods"]:
                q = (
                    float(norm.ppf((1 + level) / 2))
                    if method == "ensemble_normal"
                    else conformal_quantile(scores[method], level)
                )
                quantiles[str(level)][method] = q if np.isfinite(q) else None
        state["groups"][group] = {
            "calibration_n": len(cal),
            "validation_n": len(val),
            "quantiles": quantiles,
            "calibration_ids": [rows[i]["id"] for i in cal],
        }
    if protocol == "G3":
        cal = np.array([i for i in roles["calibration"] if np.isfinite(y[i])], int)
        families = sorted({rows[i]["family"] for i in cal})
        maxima = [
            float(
                np.max(
                    np.abs(
                        y[[i for i in cal if rows[i]["family"] == f]]
                        - point[[i for i in cal if rows[i]["family"] == f]]
                    )
                )
            )
            for f in families
        ]
        state["family_block_audit"] = {
            "independent_calibration_families": len(families),
            "quantiles": {
                str(level): (q if np.isfinite(q := conformal_quantile(maxima, level)) else None)
                for level in config["nominal_levels"]
            },
            "status": "No finite 80/90% group-conformal interval from one calibration family; no mutation-as-family replication",
        }
    return state


def prediction_intervals(point, sd, distance, shuffle_sd, rows, state, config):
    fields = {}
    for level in config["nominal_levels"]:
        for method in config["methods"]:
            key = method + "_" + str(round(100 * level))
            lower = [None] * len(rows)
            upper = [None] * len(rows)
            status = ["no_calibration_group"] * len(rows)
            for group, record in state["groups"].items():
                ix = np.array(
                    [
                        i
                        for i, r in enumerate(rows)
                        if group == "pooled" or r["protein_id"] == group
                    ],
                    int,
                )
                scale = (
                    np.maximum(sd[ix], config["native_scale_floor_kcal_mol"])
                    if method == "ensemble_scaled"
                    else np.maximum(shuffle_sd[ix], config["native_scale_floor_kcal_mol"])
                    if method == "shuffled_scaled"
                    else np.maximum(distance[ix], config["distance_floor"])
                    if method == "distance_scaled"
                    else sd[ix]
                    if method == "ensemble_normal"
                    else np.ones(len(ix))
                )
                q = record["quantiles"][str(level)][method]
                result = interval(point[ix], scale, float("inf") if q is None else q, "kcal/mol")
                for j, i in enumerate(ix):
                    lower[i], upper[i], status[i] = (
                        result["lower"][j],
                        result["upper"][j],
                        result["status"],
                    )
            fields[key] = {"lower": lower, "upper": upper, "status": status}
    return fields
