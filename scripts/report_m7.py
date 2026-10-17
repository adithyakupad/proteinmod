"""Report only completed M7 predictions; never feed test results back into training."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proteinmod.data import checksum, write_json
from proteinmod.models import metrics
from proteinmod.multitask_experiments import history, output_path
from proteinmod.structure_pipeline import write_csv


def report(source, destination):
    before = history()
    output = output_path(destination)
    with (source / "predictions.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    matched = defaultdict(dict)
    for r in raw:
        key = (r["function_fold"], r["stability_fold"], r["task"], r["seed"])
        candidate = (r["model"], r["stability_weight"])
        matched[key].setdefault(candidate, []).append(r["id"])
    for candidates in matched.values():
        baseline = next(ids for (model, _), ids in candidates.items() if model == "independent")
        assert all(
            len(ids) == len(set(ids)) and set(ids) == set(baseline) for ids in candidates.values()
        ), "Model comparison IDs differ"
    groups = defaultdict(list)
    for r in raw:
        model = r["model"]
        if model == "joint" and float(r["stability_weight"]) != 1:
            model += "_weight" + r["stability_weight"]
        protocol = "within" if r["function_fold"] == "within" else "held_out"
        groups[protocol, r["task"], model, int(r["seed"]), r["protein_id"]].append(r)
    per = []
    for (protocol, task, model, seed, protein), values in groups.items():
        scores = metrics(
            [float(r["measured"]) for r in values], [float(r["prediction"]) for r in values]
        )
        per.append(
            {
                "protocol": protocol,
                "task": task,
                "model": model,
                "seed": seed,
                "protein_id": protein,
                "n": len(values),
                "spearman": scores["spearman_rho"],
                "rmse_kcal_mol": scores["rmse"] if task == "stability" else None,
                "mae_kcal_mol": scores["mae"] if task == "stability" else None,
            }
        )
    write_csv(output / "per_protein.csv", per)
    by_seed = defaultdict(list)
    for r in per:
        by_seed[r["protocol"], r["task"], r["model"], r["seed"]].append(r)
    aggregates = defaultdict(list)
    for (protocol, task, model, seed), values in by_seed.items():
        aggregates[protocol, task, model].append(
            {
                "seed": seed,
                "spearman": float(np.mean([r["spearman"] or 0 for r in values])),
                "rmse": float(np.mean([r["rmse_kcal_mol"] for r in values]))
                if task == "stability"
                else None,
            }
        )
    summary = []
    for (protocol, task, model), values in aggregates.items():
        r = {"protocol": protocol, "task": task, "model": model, "seeds": len(values)}
        for metric in ("spearman", "rmse"):
            v = [x[metric] for x in values if x[metric] is not None]
            r[metric + "_mean"] = float(np.mean(v)) if v else None
            r[metric + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        summary.append(r)
    write_csv(output / "summary.csv", summary)
    changes = []
    for r in per:
        if r["model"] not in ("joint", "selected_joint", "frozen"):
            continue
        old = next(
            x
            for x in per
            if x["model"] == "independent"
            and all(x[k] == r[k] for k in ("protocol", "task", "seed", "protein_id"))
        )
        delta = (r["spearman"] or 0) - (old["spearman"] or 0)
        changes.append(
            {k: r[k] for k in ("protocol", "task", "model", "seed", "protein_id")}
            | {
                "spearman_delta": delta,
                "rmse_delta_kcal_mol": (r["rmse_kcal_mol"] - old["rmse_kcal_mol"])
                if r["task"] == "stability"
                else None,
                "negative_transfer": (r["rmse_kcal_mol"] > old["rmse_kcal_mol"])
                if r["task"] == "stability"
                else delta < 0,
            }
        )
    write_csv(output / "transfer.csv", changes)
    selected = [r for r in raw if r["model"] == "selected_joint" and r["task"] == "stability"]
    largest = sorted(
        selected, key=lambda r: abs(float(r["prediction"]) - float(r["measured"])), reverse=True
    )[:30]
    write_csv(output / "largest_stability_errors.csv", largest)
    runs = json.loads((source / "runs.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    for selected in (r for r in runs if r["mode"] == "selected_joint"):
        cohort = [
            r
            for r in runs
            if all(r[k] == selected[k] for k in ("function_fold", "stability_fold", "seed"))
        ]
        candidates = [r for r in cohort if r["mode"] == "joint"]
        best = min(candidates, key=lambda r: (r["selection"]["score"], r["stability_weight"]))
        assert selected["stability_weight"] == best["stability_weight"]
        equal = next(r for r in candidates if r["stability_weight"] == 1)
        for task in ("function", "stability"):
            independent = next(r for r in cohort if r["mode"] == task)
            assert (
                independent["selection"]["examples_processed"][task]
                == equal["selection"]["examples_processed"][task]
            )
    diagnostics = []
    for r in runs:
        selection = r["selection"]
        chosen = selection["trace"][selection["epoch"] - 1]
        for task in r["test"]:
            train = selection["train"][task]
            train_score = (
                float(np.mean([v["spearman_rho"] or 0 for v in train.values()]))
                if task == "function"
                else train["macro"]["rmse"]
            )
            diagnostics.append(
                {
                    "function_fold": r["function_fold"],
                    "stability_fold": r["stability_fold"],
                    "seed": r["seed"],
                    "model": r["mode"],
                    "stability_weight": r["stability_weight"],
                    "task": task,
                    "selected_epoch": selection["epoch"],
                    "selected_before_shared_updates": selection["epoch"] <= 2,
                    "train_macro_spearman_or_rmse": train_score,
                    "validation_criterion": chosen["validation"][task],
                    "examples_processed": selection["examples_processed"][task],
                    "fit_seconds": selection["elapsed_seconds"],
                }
            )
    write_csv(output / "fit_diagnostics.csv", diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = ("independent", "frozen", "joint", "selected_joint")
    for ax, task, metric, label in zip(
        axes,
        ("function", "stability"),
        ("spearman", "rmse"),
        ("Function macro Spearman (higher better)", "Stability macro RMSE kcal/mol (lower better)"),
        strict=True,
    ):
        values = [
            next(
                r
                for r in summary
                if r["protocol"] == "held_out" and r["task"] == task and r["model"] == n
            )
            for n in names
        ]
        ax.bar(
            range(4),
            [r[metric + "_mean"] for r in values],
            yerr=[r[metric + "_sd"] for r in values],
            capsize=4,
        )
        ax.set_xticks(
            range(4), ("Independent", "Frozen", "Equal joint", "Selected joint"), rotation=25
        )
        ax.set_ylabel(label)
    fig.suptitle("Protein-held-out evaluation: mean ± sample SD over 3 seeds")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for r in runs:
        if r["mode"] != "selected_joint" or r["function_fold"] == "within":
            continue
        for ax, task in zip(axes, ("function", "stability"), strict=True):
            ax.plot(
                [x["epoch"] for x in r["selection"]["trace"]],
                [x["validation"][task] for x in r["selection"]["trace"]],
                alpha=0.6,
                label=f"{r['function_fold']}/{r['stability_fold']} seed{r['seed']}",
            )
            ax.set_title(task + " validation criterion (lower better)")
            ax.set_xlabel("Epoch (first 2: heads only)")
    axes[1].legend(fontsize=6)
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)
    lines = [
        "# M7 actual local multitask results",
        "",
        "Primary protein-held-out results; equal protein macro; mean ± sample SD across three fixed seeds.",
        "",
        "| Task | Model | Macro Spearman | Macro RMSE kcal/mol |",
        "|---|---|---:|---:|",
    ]
    for r in summary:
        if r["protocol"] != "held_out" or r["model"] not in names:
            continue
        rho = f"{r['spearman_mean']:.6f} ± {r['spearman_sd']:.6f}"
        rmse = f"{r['rmse_mean']:.6f} ± {r['rmse_sd']:.6f}" if r["task"] == "stability" else "—"
        lines.append(f"| {r['task']} | {r['model']} | {rho} | {rmse} |")
    lines += [
        "",
        f"Actual runtime: {manifest['elapsed_seconds']:.2f} s CPU, peak process RSS {manifest['peak_rss_bytes'] / 2**20:.1f} MiB. ESM extraction reused from immutable caches, excluded from training runtime.",
        f"{len(raw)} held-out predictions (includes grid candidates and selected aliases); {len(runs) - 12} actual fits, plus 12 selected-result aliases.",
        "",
        "## Negative transfer",
        "",
    ]
    for task in ("function", "stability"):
        for model in ("joint", "selected_joint"):
            ix = [
                r
                for r in changes
                if r["protocol"] == "held_out" and r["task"] == task and r["model"] == model
            ]
            lines.append(
                f"{task}, {model}: degradation in {sum(r['negative_transfer'] for r in ix)}/{len(ix)} protein-seed cases versus matched independent training."
            )
    parameters = runs[0]["selection"]["parameters"]
    encoder_parameters = runs[0]["selection"]["encoder_parameters"]
    head_parameters = (parameters - encoder_parameters) // 2
    lines += [
        "",
        f"Parameter controls: encoder {encoder_parameters:,}; each task head {head_parameters:,}. Effective independent pathway {encoder_parameters + head_parameters:,} per task, two independent pathways {2 * (encoder_parameters + head_parameters):,}; joint {parameters:,}; frozen control trains only {2 * head_parameters:,}. Unused independent heads exist solely for seed-matched initialization and are excluded from effective deployed counts.",
        "",
        "Equal-joint and the independent pair use identical per-task example exposures and encoder forward-pass budgets. Grid selection costs three joint fits per fold/seed; it is not compute-matched to one independent pair.",
    ]
    gradients = [
        r["selection"]["gradient"]["cosine"]
        for r in runs
        if r["mode"] == "joint" and r["selection"]["gradient"] is not None
    ]
    lines += [
        "",
        f"First trainable-step task gradient cosine range {min(gradients):.4f} to {max(gradients):.4f}. These local diagnostics do not establish causal interference across training.",
        "",
        "Configuration/protocol was committed before the full experiments. See docs/m7_protocol.md for exact roles, validation-only selection and native-unit handling.",
        "",
        "## Limits",
        "",
        "Only three proteins per task; no paired outcomes. Auxiliary fold pairing is prespecified, not exhaustive. Function protein and assay shifts remain confounded. No verified unseen-family generalization. Source stability compact table lacks raw replicate/censoring metadata; 1TUD has 60/600 targets at its observed upper bound. Geometry is reference geometry. Neither head conditions on the other task prediction; no mechanistic stability–function claim or M8 implementation.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    assert history() == before
    write_json(
        output / "report_manifest.json",
        {
            "source_predictions_sha256": checksum(source / "predictions.csv"),
            "script_sha256": checksum(Path(__file__)),
            "files": {
                p.name: checksum(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "report_manifest.json"
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m7"))
    parser.add_argument("--output", type=Path, default=Path("runs/m7"))
    args = parser.parse_args()
    report(args.input, args.output)
