"""M8 paired comparisons and negative controls, generated only after fixed experiments."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proteinmod.conditioning_experiments import history, output_path
from proteinmod.data import checksum, write_json
from proteinmod.structure_pipeline import write_csv


def report(source, destination):
    before = history()
    output = output_path(destination)
    runs = json.loads((source / "runs.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    with (source / "predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    id_groups = defaultdict(list)
    for r in predictions:
        id_groups[
            r["protocol"], r["fold"], r["seed"], r["source"], r["model"], r["control"]
        ].append(r["id"])
    for key, ids in id_groups.items():
        reference = id_groups[key[:3] + ("none", "m7_function", "none")]
        assert len(ids) == len(set(ids)) and set(ids) == set(reference), "Unmatched test IDs"
    per = [
        {
            "protocol": r["protocol"],
            "fold": r["fold"],
            "source": r["source"],
            "model": r["model"],
            "control": r["control"],
            "seed": r["seed"],
            "protein_id": p,
            "n": v["n"],
            "spearman": v["spearman_rho"],
        }
        for r in runs
        for p, v in r["per_protein"].items()
    ]
    write_csv(output / "per_protein.csv", per)
    grouped = defaultdict(list)
    for r in per:
        grouped[r["protocol"], r["source"], r["model"], r["control"], r["seed"]].append(
            r["spearman"] or 0
        )
    seed_scores = {k: float(np.mean(v)) for k, v in grouped.items()}
    summary, paired = [], []
    base_model = {
        "stability_only": "m7_function",
        "residual": "m7_function",
        "feature_ridge": "feature_ridge",
        "neural_head": "neural_head",
    }
    cases = sorted({k[:4] for k in grouped})
    for protocol, origin, model, control in cases:
        values = np.array(
            [seed_scores[protocol, origin, model, control, s] for s in manifest["config"]["seeds"]]
        )
        record = {
            "protocol": protocol,
            "source": origin,
            "model": model,
            "control": control,
            "macro_spearman": float(values.mean()),
            "seed_sd": float(values.std(ddof=1)),
        }
        if origin != "none":
            baseline = np.array(
                [
                    seed_scores[protocol, "none", base_model[model], "none", s]
                    for s in manifest["config"]["seeds"]
                ]
            )
            delta = values - baseline
            record.update(
                {
                    "baseline": base_model[model],
                    "delta_mean": float(delta.mean()),
                    "delta_seed_sd": float(delta.std(ddof=1)),
                    "improved_seeds": int((delta > 0).sum()),
                }
            )
            for seed, change in zip(manifest["config"]["seeds"], delta, strict=True):
                paired.append(
                    {
                        "protocol": protocol,
                        "source": origin,
                        "model": model,
                        "control": control,
                        "seed": seed,
                        "baseline": base_model[model],
                        "macro_delta": float(change),
                    }
                )
        else:
            record.update(
                {"baseline": model, "delta_mean": 0.0, "delta_seed_sd": 0.0, "improved_seeds": 0}
            )
        summary.append(record)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_changes.csv", paired)
    control_comparisons = []
    for r in summary:
        if r["control"] != "predicted":
            continue
        for control in ("shuffled", "dummy"):
            values = [
                seed_scores[r["protocol"], r["source"], r["model"], "predicted", seed]
                - seed_scores[r["protocol"], r["source"], r["model"], control, seed]
                for seed in manifest["config"]["seeds"]
            ]
            control_comparisons.append(
                {k: r[k] for k in ("protocol", "source", "model")}
                | {
                    "control": control,
                    "predicted_minus_control_mean": float(np.mean(values)),
                    "paired_seed_sd": float(np.std(values, ddof=1)),
                    "positive_seeds": int(sum(v > 0 for v in values)),
                }
            )
    write_csv(output / "control_comparisons.csv", control_comparisons)
    sources = json.loads((source / "source_models.json").read_text())
    source_rows = [
        {
            "protocol": "S1" if r["function_fold"] == "within" else "S2",
            "function_fold": r["function_fold"],
            "stability_fold": r["stability_fold"],
            "source": r["source"],
            "alpha": r["selection"]["alpha"],
            "stability_macro_rmse_kcal_mol": r["stability_test"]["macro"]["rmse"],
            "stability_macro_spearman": r["stability_test"]["macro"]["spearman_rho"],
            **r["ood"],
        }
        for r in sources
    ]
    write_csv(output / "source_quality.csv", source_rows)
    # Held-out ranks are computed solely for posthoc error analysis, never conditioning inputs.
    error_groups = defaultdict(list)
    for r in predictions:
        if r["protocol"] == "G2" and r["model"] == "neural_head" and r["control"] == "predicted":
            error_groups[r["source"], r["seed"], r["protein_id"]].append(r)
    errors = []
    for values in error_groups.values():
        measured = rankdata([float(r["measured"]) for r in values]) / len(values)
        predicted = rankdata([float(r["prediction"]) for r in values]) / len(values)
        errors.extend(
            {**r, "posthoc_test_rank_error": float(abs(a - b))}
            for r, a, b in zip(values, measured, predicted, strict=True)
        )
    write_csv(
        output / "largest_rank_errors.csv",
        sorted(errors, key=lambda r: r["posthoc_test_rank_error"], reverse=True)[:30],
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    names = ("stability_only", "feature_ridge", "residual", "neural_head")
    for ax, origin in zip(axes, manifest["config"]["stability_sources"], strict=True):
        for j, control in enumerate(("predicted", "shuffled", "dummy")):
            cells = [
                next(
                    r
                    for r in summary
                    if r["protocol"] == "G2"
                    and r["source"] == origin
                    and r["model"] == m
                    and r["control"] == control
                )
                for m in names
            ]
            ax.errorbar(
                np.arange(4) + (j - 1) * 0.13,
                [r["delta_mean"] for r in cells],
                yerr=[r["delta_seed_sd"] for r in cells],
                fmt="o",
                capsize=3,
                label=control,
            )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(
            range(4), ("Scalar only", "Feature ridge", "Residual", "Neural head"), rotation=20
        )
        ax.set_title(origin)
        ax.set_ylabel("Δ macro Spearman vs baseline")
        ax.legend(fontsize=8)
    fig.suptitle(
        "G2 held-out proteins: paired mean ± seed SD (3 seeds)\nScalar only uses M7 baseline; other models use their matched baseline"
    )
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    with (source / "predicted_stability.csv").open() as handle:
        stability = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, origin in zip(axes, manifest["config"]["stability_sources"], strict=True):
        values = [
            float(r["predicted_ddg_kcal_mol"])
            for r in stability
            if r["source"] == origin and r["fold"] != "within" and r["function_role"] == "test"
        ]
        ax.hist(values, bins=35)
        ax.set_title(origin)
        ax.set_xlabel("Predicted folding ΔΔG kcal/mol, positive destabilization")
        ax.set_ylabel("Held-out function variants")
    fig.suptitle("Unverified OOD stability values; source trained on single mutants")
    fig.savefig(output / "stability_ood.png", dpi=160)
    plt.close(fig)
    lines = [
        "# M8 actual stability-informed function results",
        "",
        "G2 protein-held-out, equal protein macro, three fixed seeds. Δ compares the appropriate independent baseline; ± is paired sample SD across seeds, not a confidence interval.",
        "",
        "| Stability source | Function model | Spearman | Δ vs baseline ± SD | Shuffled Δ | Dummy Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for origin in manifest["config"]["stability_sources"]:
        for model in names:
            cells = {
                c: next(
                    r
                    for r in summary
                    if r["protocol"] == "G2"
                    and r["source"] == origin
                    and r["model"] == model
                    and r["control"] == c
                )
                for c in ("predicted", "shuffled", "dummy")
            }
            r = cells["predicted"]
            lines.append(
                f"| {origin} | {model} | {r['macro_spearman']:.6f} | {r['delta_mean']:+.6f} ± {r['delta_seed_sd']:.6f} | {cells['shuffled']['delta_mean']:+.6f} | {cells['dummy']['delta_mean']:+.6f} |"
            )
    lines += ["", "Independent G2 references:", ""]
    for r in summary:
        if r["protocol"] == "G2" and r["source"] == "none":
            lines.append(f"- {r['model']}: {r['macro_spearman']:.6f} ± {r['seed_sd']:.6f}.")
    lines += [
        "",
        f"Actual run: {manifest['elapsed_seconds']:.2f} s CPU, peak RSS {manifest['peak_rss_bytes'] / 2**20:.1f} MiB; {len(runs)} function model/seed/fold records and {len(predictions):,} held-out predictions. Cached ESM/representations exclude prior extraction/training costs.",
        "",
        "Eight independently fitted M6 ridge source models (two sources × four folds). Function fits: 4 independent feature ridges, 12 independent neural heads, 216 conditioned/control ridges, 72 conditioned/control neural heads = 304; source fits bring total to 312. Twelve fixed M7 baselines are replayed, not retrained when caches exist.",
        "",
        "## Interpretation limits",
        "",
        "All function stability values are predicted and OOD; no paired experimental stability measurements are invented. Sources were trained on permitted M6 training proteins and selected on stability validation only. Function model selection uses function validation only. All candidates and negative controls retained. Shuffling preserves protein/role marginals; dummy matches training mean/SD, not all distributional properties.",
        "",
        "Only three function proteins and three short stability constructs; source single-mutant training does not validate multi-mutant thermodynamics. Reference geometry and ESM pretraining limits remain. No calibrated uncertainty or mechanistic stability–function claim; no M9. See docs/m8_protocol.md and interpretation.md for design and failure analysis.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    assert history() == before
    write_json(
        output / "report_manifest.json",
        {
            "script_sha256": checksum(Path(__file__)),
            "source_predictions_sha256": checksum(source / "predictions.csv"),
            "files": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "report_manifest.json"
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m8"))
    parser.add_argument("--output", type=Path, default=Path("runs/m8"))
    args = parser.parse_args()
    report(args.input, args.output)
