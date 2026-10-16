"""Recompute uncertainty, calibration and selective-risk metrics from saved examples."""

import csv
import json
from collections import defaultdict

import numpy as np

from proteinmod.data import write_json
from proteinmod.structure_pipeline import write_csv
from proteinmod.uncertainty import correlation, interval_metrics, ordinal_error, selective_risk


def number(value):
    return None if value in ("", None) else float(value)


def analyze(predictions, config):
    grouped = defaultdict(list)
    for p in predictions:
        grouped[
            tuple(p[k] for k in ("dataset", "protocol", "fold", "model", "repeat", "protein_id"))
        ].append(p)
    points = []
    intervals = []
    curves = []
    for key, part in sorted(grouped.items()):
        base = dict(
            zip(("dataset", "protocol", "fold", "model", "repeat", "protein_id"), key, strict=True)
        )
        base["family"] = part[0]["family"] or None
        native = [p for p in part if number(p["measured_native"]) is not None]
        y = np.array([float(p["measured_native"]) for p in native])
        p = np.array([float(p["native_point"]) for p in native])
        sd = np.array([float(p["native_sd"]) for p in native])
        distance = np.array([float(p["distance_to_training"]) for p in native])
        shuffle = np.array([float(p["native_sd_shuffled"]) for p in native])
        error = np.abs(y - p)
        point = {
            **base,
            "n_all": len(part),
            "n_native": len(native),
            "native_mae": float(error.mean()) if len(native) else None,
            "native_rmse": float(np.sqrt(np.mean((y - p) ** 2))) if len(native) else None,
            "native_spearman": correlation(y, p),
            "native_uncertainty_error_rho": correlation(sd, error),
            "distance_error_rho": correlation(distance, error),
            "shuffled_error_rho": correlation(shuffle, error),
            "native_mean_sd": float(sd.mean()) if len(native) else None,
            "nearest_training_reference_identity": number(
                part[0]["nearest_training_reference_identity"]
            ),
            "structure_coverage": float(part[0]["structure_coverage"]),
            "n_single": sum(int(p["mutation_count"]) == 1 for p in part),
            "n_multiple": sum(int(p["mutation_count"]) > 1 for p in part),
        }
        if native:
            for ranker, ranking in [
                ("ensemble", sd),
                ("shuffled", shuffle),
                ("distance", distance),
                ("constant", np.zeros(len(native))),
            ]:
                for r in selective_risk(
                    error, ranking, [p["id"] for p in native], config["retained_fractions"]
                ):
                    curves.append({**base, "risk_type": "native_mae", "ranker": ranker, **r})
            for level in config["nominal_levels"]:
                for method in config["methods"]:
                    name = method + "_" + str(round(100 * level))
                    finite = np.array(
                        [
                            number(r[name + "_lower"]) is not None
                            and number(r[name + "_upper"]) is not None
                            for r in native
                        ]
                    )
                    result = {
                        "n": len(native),
                        "finite_n": int(finite.sum()),
                        "unbounded_n": int((~finite).sum()),
                    }
                    if finite.any():
                        lo = np.array(
                            [
                                float(r[name + "_lower"])
                                for r in native
                                if number(r[name + "_lower"]) is not None
                            ]
                        )
                        hi = np.array(
                            [
                                float(r[name + "_upper"])
                                for r in native
                                if number(r[name + "_upper"]) is not None
                            ]
                        )
                        metrics = interval_metrics(y[finite], lo, hi, level)
                        result.update(
                            {
                                "coverage": float(
                                    (metrics["coverage"] * finite.sum() + (~finite).sum())
                                    / len(native)
                                ),
                                "mean_width": metrics["mean_width"] if finite.all() else None,
                                "median_width": metrics["median_width"] if finite.all() else None,
                                "interval_score": metrics["interval_score"]
                                if finite.all()
                                else None,
                                "finite_coverage": metrics["coverage"],
                                "finite_mean_width": metrics["mean_width"],
                            }
                        )
                    else:
                        result.update(
                            {
                                "coverage": 1.0,
                                "mean_width": None,
                                "median_width": None,
                                "interval_score": None,
                                "finite_coverage": None,
                                "finite_mean_width": None,
                            }
                        )
                    intervals.append({**base, "method": method, "nominal": level, **result})
        if base["dataset"] == "M12":
            measured = np.array([float(p["measured_original"]) for p in part])
            pred = np.array([float(p["rank_point"]) for p in part])
            error_rank = ordinal_error(measured, pred)
            rank_sd = np.array([float(p["rank_sd"]) for p in part])
            rank_shuffle = np.array([float(p["rank_sd_shuffled"]) for p in part])
            d = np.array([float(p["distance_to_training"]) for p in part])
            valid = np.isfinite(error_rank)
            point.update(
                {
                    "rank_spearman": correlation(measured, pred),
                    "rank_outside_unit_fraction": float(np.mean((pred < 0) | (pred > 1))),
                    "deterministic_rank_spearman": correlation(
                        measured, [float(p["deterministic_rank"]) for p in part]
                    ),
                    "ordinal_error": float(error_rank[valid].mean()) if valid.any() else None,
                    "ordinal_uncertainty_error_rho": correlation(rank_sd, error_rank),
                    "single_ordinal_error": float(
                        np.mean(
                            [
                                e
                                for e, r in zip(error_rank, part, strict=True)
                                if int(r["mutation_count"]) == 1
                            ]
                        )
                    )
                    if any(int(r["mutation_count"]) == 1 for r in part)
                    else None,
                    "multiple_ordinal_error": float(
                        np.mean(
                            [
                                e
                                for e, r in zip(error_rank, part, strict=True)
                                if int(r["mutation_count"]) > 1
                            ]
                        )
                    )
                    if any(int(r["mutation_count"]) > 1 for r in part)
                    else None,
                }
            )
            for ranker, ranking in [
                ("ensemble", rank_sd),
                ("shuffled", rank_shuffle),
                ("distance", d),
                ("constant", np.zeros(len(part))),
            ]:
                for r in selective_risk(
                    error_rank[valid],
                    ranking[valid],
                    np.array([p["id"] for p in part])[valid],
                    config["retained_fractions"],
                ):
                    curves.append(
                        {**base, "risk_type": "ordinal_discordance", "ranker": ranker, **r}
                    )
        else:
            point.update(
                {
                    k: None
                    for k in (
                        "rank_spearman",
                        "rank_outside_unit_fraction",
                        "deterministic_rank_spearman",
                        "ordinal_error",
                        "ordinal_uncertainty_error_rho",
                        "single_ordinal_error",
                        "multiple_ordinal_error",
                    )
                }
            )
        points.append(point)
    return points, intervals, curves


def macro(records, keys, metrics, strict=()):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda x: str(x[0])):
        families = defaultdict(list)
        for r in rows:
            families[r["family"] or r["protein_id"]].append(r)
        item = dict(zip(keys, key, strict=True))
        item["replication_groups"] = len(families)
        for metric in metrics:
            values = []
            for group in families.values():
                v = [r[metric] for r in group if r[metric] is not None]
                values.append(
                    float(np.mean(v))
                    if v and (metric not in strict or len(v) == len(group))
                    else None
                )
            valid = [v for v in values if v is not None]
            item[metric] = (
                float(np.mean(valid))
                if valid and (metric not in strict or len(valid) == len(values))
                else None
            )
            item[metric + "_group_sd"] = float(np.std(valid)) if valid else None
            item[metric + "_defined_groups"] = len(valid)
        result.append(item)
    return result


def report(output, config):
    with (output / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    points, intervals, curves = analyze(predictions, config)
    write_csv(output / "per_protein.csv", points)
    write_csv(output / "interval_metrics.csv", intervals)
    write_csv(output / "selective_risk.csv", curves)
    intervals_summary = macro(
        intervals,
        ("dataset", "protocol", "model", "method", "nominal"),
        ("coverage", "mean_width", "median_width", "interval_score", "finite_n", "unbounded_n"),
        strict=("mean_width", "median_width", "interval_score"),
    )
    point_summary = macro(
        points,
        ("dataset", "protocol", "model"),
        (
            "native_mae",
            "native_uncertainty_error_rho",
            "distance_error_rho",
            "shuffled_error_rho",
            "rank_spearman",
            "deterministic_rank_spearman",
            "ordinal_error",
            "ordinal_uncertainty_error_rho",
            "single_ordinal_error",
            "multiple_ordinal_error",
        ),
    )
    curve_summary = macro(
        curves,
        ("dataset", "protocol", "model", "risk_type", "ranker", "retained_fraction"),
        ("risk",),
    )
    per_family = macro(
        intervals,
        ("dataset", "protocol", "model", "family", "method", "nominal"),
        ("coverage", "mean_width", "interval_score"),
        strict=("mean_width", "interval_score"),
    )
    write_json(
        output / "summary.json",
        {"points": point_summary, "intervals": intervals_summary, "selective_risk": curve_summary},
    )
    write_csv(output / "per_family.csv", per_family)
    repeat_summary = macro(
        intervals,
        ("dataset", "protocol", "model", "repeat", "method", "nominal"),
        ("coverage", "mean_width", "interval_score"),
        strict=("mean_width", "interval_score"),
    )
    write_csv(output / "replicate_metrics.csv", repeat_summary)
    bins = []
    for a, b in zip(config["similarity_bins"][:-1], config["similarity_bins"][1:], strict=True):
        selected = [
            p
            for p in points
            if p["dataset"] == "M12"
            and p["nearest_training_reference_identity"] is not None
            and a <= p["nearest_training_reference_identity"] < b
        ]
        for row in macro(
            selected, ("dataset", "model"), ("native_mae", "native_mean_sd", "ordinal_error")
        ):
            bins.append({"identity_low": a, "identity_high": b, **row})
    write_csv(output / "ood_similarity_bins.csv", bins)

    def find(protocol, model, method="calibration_residual", level=0.9, dataset="M12"):
        return next(
            r
            for r in intervals_summary
            if (r["dataset"], r["protocol"], r["model"], r["method"], r["nominal"])
            == (dataset, protocol, model, method, level)
        )

    def fmt(value):
        return "unbounded" if value is None else f"{value:.3f}"

    models = sorted({p["model"] for p in points if p["dataset"] == "M12"})
    environment = json.loads((output / "environment.json").read_text())
    runs = json.loads((output / "runs.json").read_text())
    fit_count = sum(
        len(r["selection"])
        if isinstance(r["selection"], list)
        else len(r["selection"]["candidates"]) + r["selection"]["bootstrap_members"]
        for r in runs
    )
    lines = [
        "# M13 predictive uncertainty and calibration audit",
        "",
        "Actual local fits on unchanged M12 test genotypes. Original assay-local DMS labels and M0–M12 evidence remain untouched. Native intervals use only 347 verified primary-table single-mutant labels, with explicit positive-destabilizing folding ΔΔG in kcal/mol. No rank output or unrelated assay score is silently called a thermodynamic measurement.",
        "",
        f"Execution: {fit_count} individual fits, {len(runs)} uncertainty evaluations and {len(predictions)} saved held-out predictions. Final-run wall time before reporting: {environment['elapsed_seconds']:.2f} s; CPU process peak RSS {environment['max_rss_bytes']:,} bytes. The separate preliminary control run is retained in control_audit.json; these costs are not one combined run. Frozen ESM features were reused from existing caches.",
        "",
        "## Independent calibration under family shift",
        "",
        "G1 reserves 12 native-eligible training variants/protein. G2 reserves two complete non-test proteins. G3 trains, validates, calibrates and tests on four distinct families. **Only one family calibrates each G3 fold.** Finite variant-residual intervals below are empirical diagnostics under shift; they do not guarantee coverage on unseen families. Correct finite-sample family-block quantiles are unbounded at both 80% and 90%, not estimates based on hundreds of supposedly independent families.",
        "",
        "Native labels come from the existing M6 primary Figure 3 data release, crosswalked by pinned reference, substitution and complete mutant sequence. The processed M12 DMS labels and primary-table estimates can differ. These are not independent experimental replicates and are never used to estimate aleatoric variance. All native affine heads fit only training labels; calibration labels enter residual quantiles only.",
        "",
        "## Native 90% interval results",
        "",
        "Equal proteins within each M12 family, then equal families; two uncertainty replicates averaged. Widths are kcal/mol. M6 controls are a separate compatible native task and average proteins. Native test counts differ from rank-cohort counts because doubles/censored/missing primary labels are ineligible.",
        "",
        "| Model | G1 coverage / width | G2 coverage / width | G3 coverage / width | G3 ensemble-scaled coverage / width |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        r = [find(p, model) for p in ("G1", "G2", "G3")]
        s = find("G3", model, "ensemble_scaled")
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{fmt(x['coverage'])} / {fmt(x['mean_width'])}" for x in [*r, s])
            + " |"
        )
    families = ["PF00018", "PF00385", "PF00046", "PF00397"]
    lines += [
        "",
        "## G3 per-family 90% residual coverage",
        "",
        "| Model | " + " | ".join(families) + " | Replicate SD of macro coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        values = [
            next(
                r["coverage"]
                for r in per_family
                if (r["dataset"], r["protocol"], r["model"], r["family"], r["method"], r["nominal"])
                == ("M12", "G3", model, f, "calibration_residual", 0.9)
            )
            for f in families
        ]
        replicas = [
            r["coverage"]
            for r in repeat_summary
            if (r["dataset"], r["protocol"], r["model"], r["method"], r["nominal"])
            == ("M12", "G3", model, "calibration_residual", 0.9)
        ]
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(fmt(v) for v in [*values, float(np.std(replicas))])
            + " |"
        )
    lines += [
        "",
        "## G3 controls and uncertainty ranking",
        "",
        "| Model | Gaussian coverage | Distance-scaled coverage | Shuffled-scaled coverage | SD–absolute-error ρ | Distance–error ρ | Shuffled–error ρ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        r = next(
            x
            for x in point_summary
            if (x["dataset"], x["protocol"], x["model"]) == ("M12", "G3", model)
        )
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(
                fmt(v)
                for v in [
                    find("G3", model, "ensemble_normal")["coverage"],
                    find("G3", model, "distance_scaled")["coverage"],
                    find("G3", model, "shuffled_scaled")["coverage"],
                    r["native_uncertainty_error_rho"],
                    r["distance_error_rho"],
                    r["shuffled_error_rho"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## G3 selective native risk",
        "",
        "Mean absolute error (kcal/mol); lower is better. Retention is performed independently within each protein, then equally aggregated across proteins/families. Constant ranking breaks ties with a prespecified ID hash. These are descriptive curves, not statistical evidence from independent mutations.",
        "",
        "| Model | Full MAE | Ensemble 25% MAE | Distance 25% MAE | Shuffled 25% MAE | Constant 25% MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:

        def risk(ranker, fraction, model=model):
            return next(
                r["risk"]
                for r in curve_summary
                if (
                    r["dataset"],
                    r["protocol"],
                    r["model"],
                    r["risk_type"],
                    r["ranker"],
                    r["retained_fraction"],
                )
                == ("M12", "G3", model, "native_mae", ranker, fraction)
            )

        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(
                fmt(v)
                for v in [
                    risk("ensemble", 1),
                    risk("ensemble", 0.25),
                    risk("distance", 0.25),
                    risk("shuffled", 0.25),
                    risk("constant", 0.25),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Native coverage at nominal 80% on G3",
        "",
        "| Model | Residual coverage / width | Ensemble-scaled coverage / width |",
        "|---|---:|---:|",
    ]
    for model in models:
        r = [
            find("G3", model, method, level=0.8)
            for method in ("calibration_residual", "ensemble_scaled")
        ]
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{fmt(x['coverage'])} / {fmt(x['mean_width'])}" for x in r)
            + " |"
        )
    lines += [
        "",
        "## M6 native-unit sanity control",
        "",
        "| Model | S1 90% coverage / width | S2 90% coverage / width |",
        "|---|---:|---:|",
    ]
    for model in config["m6_native_control"]["models"]:
        r = [find(p, model, dataset="M6") for p in ("S1", "S2")]
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{fmt(x['coverage'])} / {fmt(x['mean_width'])}" for x in r)
            + " |"
        )
    lines += [
        "",
        "S2 calibration variants come from its training protein, with a separate validation protein and held-out test protein. This is a native-unit sanity check under protein shift, not a family-calibration guarantee. No labels from the three-protein M6 control are mixed into M12 fitting; the same-variant primary-label crosswalk is the explicit secondary task above.",
        "",
        "## Interpretation and failure analysis",
        "",
        "The largest residual-interval failure is the held-out SH3 family: sequence ridge covers only about 13% at nominal 90%; all models lose coverage there. High aggregate coverage on homeodomains/WW can conceal that family-specific failure. Replicate stability does not repair a systematic cross-family shift.",
        "Every SH3 sequence-ridge ensemble mean rank output lies outside the training target range [0,1]. Its training-only affine native head consequently extrapolates strongly: this is a predictor/head transfer failure, not a reason to refit a test-family scale. No clipping, test-derived normalization or post-hoc model repair was applied. Per-protein extrapolation fractions are retained.",
        "Ensemble/bootstrap dispersion measures sensitivity to this training procedure and cohort, not a Bayesian posterior or all epistemic uncertainty. The shared feature extractor, fixed transforms, limited bootstrap members and affine native head can agree while remaining wrong on a new family. A Gaussian band built solely from that dispersion omits residual/model/measurement error and can under-cover severely.",
        "",
        "Residual intervals include combined error without an aleatoric/epistemic decomposition. Ensemble-scaled or distance-scaled conformal intervals can widen dramatically where scales are small or shift changes residual distributions; inspect interval scores and widths rather than coverage alone. Shuffling provides a null ranking/control of scale assignment, not a fitted uncertainty method. The validation-residual baseline reuses model-selection validation and is explicitly not split conformal.",
        "",
        "Only four independent M12 families are available; no mutation-level significance or population-wide family-coverage CI is justified. Group dispersion and each family are retained in per_family.csv. Native eligibility selects singles and available primary-table positions; ordinal diagnostics retain all single/double variants. Structural coverage is complete in this cohort, so no robustness claim for missing structures follows. AlphaFold confidence in the separate M6 inputs and unknown experimental confidence in M12 are not predictive SDs.",
        "",
        "Report tables are regenerated from saved per-example means, members and bounds. Test labels are used only by evaluation. No test-fitted CDF, outcome scale, residual quantile, family calibration or hyperparameter change is used. Different train sizes after reserving calibration prevent direct attribution of M12/M13 point-performance differences solely to uncertainty estimation.",
        "",
        "See DATA_PROVENANCE.md, PREREGISTRATION.md, audit.json, runs.json, environment.json, interval_metrics.csv, per_protein.csv and selective_risk.csv for exclusions, actual member fits, per-family failures and costs. Exact reproduction is in docs/M13.md. M14/M15 are excluded.",
        "",
        "![Coverage](coverage.png)",
        "",
        "![Selective risk](selective_risk.png)",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, protocol in zip(axes, ("G1", "G2", "G3"), strict=True):
        for method in ("calibration_residual", "ensemble_scaled", "ensemble_normal"):
            ax.plot(
                range(len(models)),
                [find(protocol, m, method)["coverage"] for m in models],
                marker="o",
                label=method,
            )
        ax.axhline(0.9, color="black", ls="--", lw=0.8)
        ax.set(
            xticks=range(len(models)),
            xticklabels=models,
            ylim=(0, 1.03),
            title=protocol,
            ylabel="Empirical native 90% coverage",
        )
        ax.tick_params(axis="x", labelrotation=45)
    axes[-1].legend(fontsize=7)
    fig.savefig(output / "coverage.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for model in models:
        r = [
            x
            for x in curve_summary
            if (x["dataset"], x["protocol"], x["model"], x["risk_type"], x["ranker"])
            == ("M12", "G3", model, "native_mae", "ensemble")
        ]
        ax.plot(
            [x["retained_fraction"] for x in r], [x["risk"] for x in r], marker="o", label=model
        )
    ax.set(
        xlabel="Retained fraction within protein",
        ylabel="Equal-family native MAE (kcal/mol)",
        title="G3 selective prediction; ensemble ranking",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "selective_risk.png", dpi=160)
    plt.close(fig)
