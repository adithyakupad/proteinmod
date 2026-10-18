"""Paired development contrasts, seed spread and native-only calibration reanalysis."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from proteinmod.data import write_json


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def rho(a, b):
    return None if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(spearmanr(a, b).statistic)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report():
    root = Path("runs/m17")
    runs = json.loads((root / "runs.json").read_text())
    with (root / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    scores = []
    for r in runs:
        values = r["per_protein"]
        scores.append(
            {k: r[k] for k in ("study", "fold", "protocol", "seed", "fraction", "model")}
            | {
                "rho": mean(v["spearman_rho"] for v in values.values()),
                "defined_proteins": sum(v["spearman_rho"] is not None for v in values.values()),
                "total_proteins": len(values),
                "train_rho": r["selection"]["train_score"],
                "validation_rho": r["selection"]["validation_score"],
                "gap": r["selection"]["train_score"] - r["selection"]["validation_score"],
                "seconds": r["seconds"],
                "parameters": r["selection"]["parameters"],
            }
        )
    key = lambda r: tuple(r[k] for k in ("study", "fold", "protocol", "seed", "fraction"))
    baselines = {key(r): r["rho"] for r in scores if r["model"] == "structural_ridge"}
    for r in scores:
        baseline = baselines[key(r)]
        r["paired_minus_structural"] = (
            None if baseline is None or r["rho"] is None else r["rho"] - baseline
        )
    write_csv(root / "paired-scores.csv", scores)
    aggregates = []
    groups = defaultdict(list)
    for r in scores:
        groups[(r["study"], r["fold"], r["protocol"], r["fraction"], r["model"])].append(r)
    for (study, fold, protocol, fraction, model), rr in sorted(groups.items()):
        aggregates.append(
            {
                "study": study,
                "fold": fold,
                "protocol": protocol,
                "fraction": fraction,
                "model": model,
                "seeds": len(rr),
                "rho": mean(r["rho"] for r in rr),
                "rho_seed_sd": float(np.std([r["rho"] for r in rr], ddof=1))
                if len(rr) > 1 and all(r["rho"] is not None for r in rr)
                else None,
                "paired_minus_structural": mean(r["paired_minus_structural"] for r in rr),
                "gap": mean(r["gap"] for r in rr),
                "seconds": sum(r["seconds"] for r in rr),
                "parameters": rr[0]["parameters"],
            }
        )
    write_csv(root / "per-family.csv", aggregates)
    # Seed spread mixes initialization and training-subsample variation at low labels.
    pp = defaultdict(list)
    for p in predictions:
        if p["study"] == "H1":
            pp[(p["fold"], p["fraction"], p["model"], p["protein_id"], p["id"])].append(p)
    uncertainty = defaultdict(list)
    for (family, fraction, model, pid, vid), rr in pp.items():
        assert len(rr) == 5 and len({r["seed"] for r in rr}) == 5
        target = {r["target"] for r in rr}
        assert len(target) == 1
        uncertainty[(family, fraction, model, pid)].append(
            (
                vid,
                float(rr[0]["target"]),
                np.mean([float(r["prediction"]) for r in rr]),
                np.std([float(r["prediction"]) for r in rr], ddof=1),
            )
        )
    diagnostics = []
    for (family, fraction, model, pid), rr in sorted(uncertainty.items()):
        rr = sorted(rr)
        a = np.array([r[1:] for r in rr])
        error = abs(rankdata(a[:, 0]) - rankdata(a[:, 1])) / len(rr)
        spread = a[:, 2]
        order = np.argsort(spread, kind="stable")
        null = np.random.default_rng(314159).permutation(order)
        informative = np.std(spread) >= 1e-12
        diagnostics.append(
            {
                "family": family,
                "fraction": fraction,
                "model": model,
                "protein_id": pid,
                "n": len(rr),
                "spread_error_rho": rho(spread, error),
                "mean_ordinal_error": float(error.mean()),
                "retain_half_error": float(error[order[: len(rr) // 2]].mean())
                if informative
                else None,
                "shuffled_retain_half_error": float(error[null[: len(rr) // 2]].mean())
                if informative
                else None,
                "mean_spread": float(spread.mean()),
                "uninformative_spread": not informative,
            }
        )
    write_csv(root / "uncertainty.csv", diagnostics)
    # Actual native outcomes and bounds from prior separately calibrated M13 models.
    calibration = defaultdict(list)
    with Path("benchmarks/m13/predictions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["native_eligible"] != "True" or r["role"] != "test" or not r["measured_native"]:
                continue
            y = float(r["measured_native"])
            for level in (80, 90):
                prefix = f"calibration_residual_{level}"
                lo, hi = float(r[prefix + "_lower"]), float(r[prefix + "_upper"])
                calibration[(r["dataset"], r["protocol"], r["model"], r["family"], level)].append(
                    (lo <= y <= hi, hi - lo)
                )
    native = []
    for (dataset, protocol, model, family, level), rr in sorted(calibration.items()):
        widths = np.array([r[1] for r in rr])
        native.append(
            {
                "dataset": dataset,
                "protocol": protocol,
                "model": model,
                "family": family,
                "nominal": level / 100,
                "n_repeat_observations": len(rr),
                "coverage": mean(r[0] for r in rr),
                "mean_width_kcal_mol": float(widths.mean()) if np.isfinite(widths).all() else None,
                "unbounded_intervals": int((~np.isfinite(widths)).sum()),
                "evidence": "M13 reanalysis, not new M17 model calibration",
            }
        )
    write_csv(root / "native-calibration-reanalysis.csv", native)
    block_records = []
    for run in json.loads(Path("benchmarks/m13/runs.json").read_text()):
        if "family_block_audit" in run["calibration"]:
            block = run["calibration"]["family_block_audit"]
            assert block["independent_calibration_families"] == 1
            assert all(v is None for v in block["quantiles"].values())
            block_records.append(
                {k: run[k] for k in ("dataset", "protocol", "fold", "model", "repeat")} | block
            )
    write_json(root / "family-block-calibration.json", block_records)

    write_json(
        root / "summary.json",
        {
            "fits": len(runs),
            "predictions": len(predictions),
            "per_family": aggregates,
            "native_calibration_groups": len(native),
            "external_confirmation": "BLOCKED label-level lineage unresolved",
            "family_ci": "Not estimated: only four development families",
            "uncertainty": "Seed/subsample spread is not measured aleatoric uncertainty or structure confidence",
        },
    )
    plot(aggregates, root)


def plot(rows, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    names = [
        "structural_ridge",
        "esm_local_ridge",
        "mpnn_local_ridge",
        "structural_mlp",
        "edge_full",
        "zero_message",
    ]
    for model in names:
        rr = [r for r in rows if r["study"] == "H1" and r["model"] == model]
        fractions = sorted({r["fraction"] for r in rr})
        values = [mean(r["rho"] for r in rr if r["fraction"] == fraction) for fraction in fractions]
        axes[0].plot(fractions, values, marker="o", label=model)
    axes[0].set(
        xlabel="Fraction of 64 training labels per protein",
        ylabel="Equal-family Spearman",
        title="Development G3 learning curves",
    )
    axes[0].legend(fontsize=7)
    full = [r for r in rows if r["study"] == "H1" and r["fraction"] == 1]
    families = sorted({r["fold"] for r in full})
    for model in names[1:]:
        rr = [next(r for r in full if r["model"] == model and r["fold"] == f) for f in families]
        axes[1].plot(families, [r["paired_minus_structural"] for r in rr], marker="o", label=model)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(
        ylabel="Paired rho minus structural ridge", title="Four family effects; no population CI"
    )
    axes[1].tick_params(axis="x", labelrotation=30)
    fig.savefig(root / "mechanisms.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    report()
