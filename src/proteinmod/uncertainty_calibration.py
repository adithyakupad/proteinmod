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
        calibration_indices, validation_indices = select("calibration"), select("validation")
        residual = np.abs(y[calibration_indices] - point[calibration_indices])
        sd_floor = config["native_scale_floor_kcal_mol"]
        distance_floor = config["distance_floor"]
        scores = {
            "validation_residual": np.abs(y[validation_indices] - point[validation_indices]),
            "calibration_residual": residual,
            "ensemble_scaled": residual / np.maximum(sd[calibration_indices], sd_floor),
            "distance_scaled": residual / np.maximum(distance[calibration_indices], distance_floor),
            "shuffled_scaled": residual / np.maximum(shuffle_sd[calibration_indices], sd_floor),
        }
        quantiles = {}
        for level in config["nominal_levels"]:
            quantiles[str(level)] = {}
            for method in config["methods"]:
                quantile = (
                    float(norm.ppf((1 + level) / 2))
                    if method == "ensemble_normal"
                    else conformal_quantile(scores[method], level)
                )
                quantiles[str(level)][method] = quantile if np.isfinite(quantile) else None
        state["groups"][group] = {
            "calibration_n": len(calibration_indices),
            "validation_n": len(validation_indices),
            "quantiles": quantiles,
            "calibration_ids": [rows[i]["id"] for i in calibration_indices],
        }
    if protocol == "G3":
        calibration_indices = np.array([i for i in roles["calibration"] if np.isfinite(y[i])], int)
        families = sorted({rows[i]["family"] for i in calibration_indices})
        maxima = [
            float(
                np.max(
                    np.abs(
                        y[[i for i in calibration_indices if rows[i]["family"] == f]]
                        - point[[i for i in calibration_indices if rows[i]["family"] == f]]
                    )
                )
            )
            for f in families
        ]
        state["family_block_audit"] = {
            "independent_calibration_families": len(families),
            "quantiles": {
                str(level): (
                    quantile if np.isfinite(quantile := conformal_quantile(maxima, level)) else None
                )
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
                group_indices = np.array(
                    [
                        row_index
                        for row_index, r in enumerate(rows)
                        if group == "pooled" or r["protein_id"] == group
                    ],
                    int,
                )
                scale = (
                    np.maximum(sd[group_indices], config["native_scale_floor_kcal_mol"])
                    if method == "ensemble_scaled"
                    else np.maximum(
                        shuffle_sd[group_indices], config["native_scale_floor_kcal_mol"]
                    )
                    if method == "shuffled_scaled"
                    else np.maximum(distance[group_indices], config["distance_floor"])
                    if method == "distance_scaled"
                    else sd[group_indices]
                    if method == "ensemble_normal"
                    else np.ones(len(group_indices))
                )
                quantile = record["quantiles"][str(level)][method]
                result = interval(
                    point[group_indices],
                    scale,
                    float("inf") if quantile is None else quantile,
                    "kcal/mol",
                )
                for group_offset, row_index in enumerate(group_indices):
                    lower[row_index], upper[row_index], status[row_index] = (
                        result["lower"][group_offset],
                        result["upper"][group_offset],
                        result["status"],
                    )
            fields[key] = {"lower": lower, "upper": upper, "status": status}
    return fields
