"""Render actual M3 predictions, error slices and compute evidence without fitting models."""

import argparse
import csv
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, load_dataset, verified_reference, write_json
from proteinmod.experiments import summarize
from proteinmod.models import metrics
from proteinmod.structure_pipeline import write_csv


def report(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["artifacts_sha256"].items():
        if checksum(output / name) != expected:
            raise ValueError(f"Experiment artifact changed: {name}")
    runs = json.loads((output / "runs.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    with open(manifest["config"]["m3"]["sequence_config"], "rb") as f:
        config = tomllib.load(f)
    rows, _ = load_dataset(config["dataset"]["path"], verified_reference(config["reference"]))
    by_id = {r["id"]: r for r in rows}
    with (output / "test_predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    summary = summarize(runs)
    expected_ids = set(audit["valid_partition_ids"]["test"])
    for run in runs:
        group = [
            p
            for p in predictions
            if p["model"] == run["model"]
            and int(p["seed"]) == run["seed"]
            and float(p["fraction"]) == run["fraction"]
        ]
        if len(group) != len(expected_ids) or {p["id"] for p in group} != expected_ids:
            raise ValueError("Models do not have identical held-out variant IDs")
        recomputed = metrics(
            [float(p["measured"]) for p in group], [float(p["prediction"]) for p in group]
        )
        for key, value in recomputed.items():
            if value is None and run["test"][key] is None:
                continue
            if (
                value is None
                or run["test"][key] is None
                or not np.isclose(value, run["test"][key], rtol=1e-10, atol=1e-12)
            ):
                raise ValueError("Saved predictions do not reproduce reported metrics")
    full = [r for r in summary if r["fraction"] == 1]
    failures, slices = [], []
    for r in runs:
        if r["fraction"] != 1:
            continue
        group = [
            p
            for p in predictions
            if p["model"] == r["model"]
            and int(p["seed"]) == r["seed"]
            and float(p["fraction"]) == 1
        ]
        for label, select in [
            ("single", lambda n: n == 1),
            ("2_to_5", lambda n: 2 <= n <= 5),
            ("6_plus", lambda n: n >= 6),
        ]:
            selected = [p for p in group if select(len(by_id[p["id"]]["mutations"]))]
            if selected:
                slices.append(
                    {
                        "model": r["model"],
                        "seed": r["seed"],
                        "burden": label,
                        "n_test": len(selected),
                        **metrics(
                            [float(p["measured"]) for p in selected],
                            [float(p["prediction"]) for p in selected],
                        ),
                    }
                )
        for p in sorted(
            group, key=lambda p: abs(float(p["measured"]) - float(p["prediction"])), reverse=True
        )[:5]:
            failures.append(
                {
                    "model": r["model"],
                    "seed": r["seed"],
                    "id": p["id"],
                    "mutations": list(by_id[p["id"]]["mutations"]),
                    "measured": float(p["measured"]),
                    "prediction": float(p["prediction"]),
                    "absolute_error": abs(float(p["measured"]) - float(p["prediction"])),
                }
            )
    write_json(output / "largest_errors.json", failures)
    write_csv(output / "error_slices.csv", slices)
    exclusions = Counter(reason["reason"] for r in audit["exclusions"] for reason in r["reasons"])
    costs = [
        {
            "model": r["model"],
            "seed": r["seed"],
            "fraction": r["fraction"],
            "seconds": r["elapsed_seconds"],
            "parameters": r.get("parameters"),
            "train_rmse": r["train"]["rmse"],
            "validation_rmse": r["validation"]["rmse"],
            "test_rmse": r["test"]["rmse"],
            "epoch": r["selection"].get("epoch"),
            "weight_decay": r["selection"].get("weight_decay"),
        }
        for r in runs
    ]
    write_csv(output / "costs_and_fit_gaps.csv", costs)
    lines = [
        "# M3: matched structural prediction results",
        "",
        "Actual local experiments; full results are means ± sample SD over seeds 0/1/2.",
        "Seed SD is training variation, not uncertainty over future proteins.",
        "",
        f"Original counts: {audit['original_counts']}; valid: {audit['valid_counts']}; excluded: {audit['excluded_counts']}.",
        f"Exclusion site reasons (a variant can have several): {dict(exclusions)}.",
        "All models were refitted on identical valid training subsets; official split membership is unchanged.",
        "",
        "| Model | Spearman rho | Pearson r | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in full:
        cells = [
            f"{r[k + '_mean']:.6f} ± {r[k + '_std'] or 0:.6f}"
            for k in ("spearman_rho", "pearson_r", "mae", "rmse")
        ]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Learning curves",
        "",
        "| Model | Fraction | Train n | Spearman rho | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['model']} | {r['fraction']} | {r['n_train']} | {r['spearman_rho_mean']:.6f} ± {r['spearman_rho_std'] or 0:.6f} | {r['rmse_mean']:.6f} ± {r['rmse_std'] or 0:.6f} |"
        )
    lines += [
        "",
        "![Matched learning curves](learning_curves.png)",
        "",
        "## Failure analysis and fit gaps",
        "",
        "| Model | Seed | Train RMSE | Validation RMSE | Test RMSE | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in costs:
        if r["fraction"] == 1:
            lines.append(
                f"| {r['model']} | {r['seed']} | {r['train_rmse']:.6f} | {r['validation_rmse']:.6f} | {r['test_rmse']:.6f} | {r['epoch'] or '—'} |"
            )
    scores = {r["model"]: r for r in full}
    if "structural_gnn" in scores:
        difference = scores["structural_gnn"]["rmse_mean"] - scores["structural_mlp"]["rmse_mean"]
        lines += [
            "",
            f"Two-layer GNN minus zero-layer control RMSE: {difference:+.6f} (negative favors message passing).",
            "The mean message-passing gain must be read alongside the seed SD; three initializations do not establish statistical significance or generalization to other proteins.",
            "The zero-layer control already sees RSA, contacts, residue chemistry and mutation identity; this ablation tests learned propagation.",
            f"Same-seed repeated GNN run: {audit['gnn_reproducibility']}.",
        ]
    lines += [
        "",
        "Largest individual residuals are retained in largest_errors.json; mutation-burden slice metrics are in error_slices.csv.",
        "The test subset has the official mutation-burden distribution, which differs from training; ranking and squared-error metrics can disagree.",
        "Training/validation gaps and seed spread limit conclusions about overfitting. Validation stopping controls selection but does not prove generalization.",
        "This compact network has limited propagation range and receives reference-construct geometry, not true mutant conformational changes.",
        "All results, including low-data and high-mutation-burden failures, are retained without test-driven architecture changes. No model is promoted based on one assay.",
        "",
        "## Mapping, confidence and leakage",
        "",
        f"Graph: {audit['graph']['nodes']} nodes, {audit['graph']['undirected_edges']} edges, {audit['graph']['connected_components']} component(s), {audit['graph']['isolated_nodes']} isolated nodes.",
        f"Missing descriptor counts: {audit['graph']['descriptor_missing_nodes']}.",
        "Experimental B-factors are not pLDDT; all missing pLDDT values have explicit masks. DSSP secondary structure was unavailable and is omitted.",
        f"Complete duplicate sequences: {audit['leakage']['duplicate_sequences']}; shared train/test substitutions: {len(audit['leakage']['train_test_shared_substitutions'])}.",
        "This is same-protein interpolation. Shared substitutions and reference graph context prevent claims of novel-site/cross-protein generalization.",
        "Eligibility/features are label-free. Scaling and target normalization fit training only; validation RMSE selects alpha/epoch/weight decay; test labels do not enter selection.",
        "See audit.json for all exclusions, source hashes, historical checksums and actual valid IDs.",
        "",
        "## Compute",
        "",
        f"Device: {manifest['config']['gnn']['device']}; CPU threads: {manifest['config']['gnn']['cpu_threads']}; total wall time: {manifest['elapsed_seconds']:.3f} seconds.",
        f"Peak process RSS bytes: {manifest.get('peak_process_rss_bytes', 'not recorded in this run')}.",
        f"Historical files verified unchanged: {manifest['historical_files_unchanged']}.",
        "Per-run tuning/training/prediction durations and parameter counts are in costs_and_fit_gaps.csv. Feature extraction uses cached M2/ESM artifacts; it is not included as new encoder compute.",
        "Weights and dense feature arrays remain in the ignored cache; predictions, traces, hashes and numerical evidence are committed.",
        "",
        "Reproduce with docs/m3_protocol.md and the exact config/environment recorded in manifest.json.",
    ]
    lines += [
        "",
        "## Mutation-burden slices (mean RMSE over seeds)",
        "",
        "| Model | Burden | Test n | RMSE |",
        "|---|---|---:|---:|",
    ]
    for model in sorted(scores):
        for burden in ("single", "2_to_5", "6_plus"):
            group = [r for r in slices if r["model"] == model and r["burden"] == burden]
            if group:
                lines.append(
                    f"| {model} | {burden} | {group[0]['n_test']} | {np.mean([r['rmse'] for r in group]):.6f} |"
                )
    lines += [
        "",
        "Only 35 single mutants and 50 variants with ≥6 mutations are held out; small slice sizes limit interpretation.",
        "The low-data GNN trails structural ridge, and its ≥6-mutation RMSE exceeds the zero-layer control. Limited labels, propagation range and reference/mutant geometry differences are plausible limitations, not established causal explanations.",
        "",
        "## Full-data compute (mean per seed, including hyperparameter selection)",
        "",
        "| Model | Seconds | Neural parameters |",
        "|---|---:|---:|",
    ]
    for model in sorted(scores):
        group = [r for r in runs if r["model"] == model and r["fraction"] == 1]
        lines.append(
            f"| {model} | {np.mean([r['elapsed_seconds'] for r in group]):.3f} | {group[0].get('parameters', '—')} |"
        )
    lines += [
        "",
        "Historical M1 encoder extraction took 224.123 seconds on MPS, separate from these cached-feature probe fits. The final experiment duration excludes development runs, tests and the failed MPS smoke test.",
        "The deterministic MPS smoke test failed on index aggregation; M3 therefore requires CPU. See local_verification.json for the recorded failure and full-dataset mutation-order checks.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model in sorted({r["model"] for r in summary}):
        group = [r for r in summary if r["model"] == model]
        for ax, metric in zip(axes, ("spearman_rho", "rmse"), strict=True):
            ax.errorbar(
                [r["n_train"] for r in group],
                [r[metric + "_mean"] for r in group],
                yerr=[r[metric + "_std"] or 0 for r in group],
                marker="o",
                capsize=3,
                label=model,
            )
            ax.set(xscale="log", xlabel="Labeled training variants", ylabel=metric)
    axes[1].legend(fontsize=8)
    fig.suptitle("Matched FLIP2 Amylase: mean ± seed SD; fixed validation/test")
    fig.savefig(output / "learning_curves.png", dpi=160)
    fig.savefig(output / "learning_curves.svg")
    plt.close(fig)
    write_json(
        output / "report_manifest.json",
        {
            "script_sha256": checksum(__file__),
            "experiment_manifest_sha256": checksum(output / "manifest.json"),
            "artifacts_sha256": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "report_manifest.json"
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/amylase_m3"))
    report(parser.parse_args().output)
