"""Fitness, measured epistasis, scale sensitivity and conditional paired-block evidence."""

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proteinmod.data import checksum, write_json
from proteinmod.models import metrics
from proteinmod.structure_pipeline import write_csv


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def paired_block_interval(truth, model, baseline, groups, loss, settings):
    unique = sorted(set(groups))
    if len(unique) < 3:
        return {
            "blocks": len(unique),
            "low": None,
            "high": None,
            "reason": "fewer than three position blocks",
        }
    rng = np.random.default_rng(settings["seed"])
    groups = np.asarray(groups)
    ix = [np.flatnonzero(groups == g) for g in unique]

    def score(p, local):
        error = truth[local] - p[local]
        return (
            float(np.sqrt(np.mean(error**2))) if loss == "rmse" else float(np.mean(np.abs(error)))
        )

    differences = []
    for _ in range(settings["resamples"]):
        local = np.concatenate([ix[j] for j in rng.integers(0, len(ix), len(ix))])
        differences.append(score(baseline, local) - score(model, local))
    return {
        "blocks": len(unique),
        "low": float(np.quantile(differences, 0.025)),
        "high": float(np.quantile(differences, 0.975)),
        "direction": "positive is improvement; baseline loss minus model loss",
        "conditional": "fixed models/observed constituent labels; not measurement-noise uncertainty",
    }


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    runs = json.loads((root / "runs.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    audit = json.loads((root / "audit.json").read_text())
    rows = json.loads((root / "rows.json").read_text())
    lookup = {r["id"]: r for r in rows}
    predictions = read_csv(root / "predictions.csv")
    eps = read_csv(root / "epistasis_predictions.csv")
    assert len(runs) == 54 and len(predictions) == 28080
    names = list(dict.fromkeys(r["model"] for r in runs))
    summary = []
    slices = []
    rare = []
    paired = []
    for protocol in ("C1", "C2"):
        training = set(next(r for r in audit["roles"] if r["protocol"] == protocol)["ids"]["train"])
        frequency = Counter(
            tuple(sorted(pair))
            for g in training
            for pair in itertools.combinations(lookup[g]["mutations"], 2)
        )
        rarity = {}
        for genotype, row in lookup.items():
            counts = [
                frequency[tuple(sorted(p))] for p in itertools.combinations(row["mutations"], 2)
            ]
            rarity[genotype] = (
                "unseen_pair"
                if any(v == 0 for v in counts)
                else "rare_pair_1_2"
                if counts and min(counts) <= 2
                else "repeated_pair"
            )
        for name in names:
            local = [r for r in runs if r["protocol"] == protocol and r["model"] == name]
            record = {
                "protocol": protocol,
                "model": name,
                "seeds": len(local),
                "rmse_mean": float(np.mean([r["test"]["rmse"] for r in local])),
                "rmse_seed_sd": float(np.std([r["test"]["rmse"] for r in local])),
                "rho_mean": None
                if all(r["test"]["spearman_rho"] is None for r in local)
                else float(np.mean([r["test"]["spearman_rho"] or 0 for r in local])),
                "rho_seed_sd": float(np.std([r["test"]["spearman_rho"] or 0 for r in local])),
                "fit_seconds": sum(r["elapsed_seconds"] for r in local),
            }
            for context in ("WT", "nonWT"):
                values = [r["epistasis_" + context] for r in local]
                record["epsilon_" + context] = {
                    "n": values[0]["n"],
                    "mae_mean": float(np.mean([r["mae"] for r in values]))
                    if values[0]["n"]
                    else None,
                    "rho_mean": None
                    if all(r["spearman"] is None for r in values)
                    else float(np.mean([r["spearman"] or 0 for r in values])),
                }
            summary.append(record)
            for run in local:
                seed = run["seed"]
                subset = [
                    r
                    for r in predictions
                    if (r["protocol"], r["model"], int(r["seed"])) == (protocol, name, seed)
                ]
                for count in (2, 3, 4):
                    selected = [r for r in subset if int(r["mutation_count"]) == count]
                    if selected:
                        slices.append(
                            {
                                "protocol": protocol,
                                "model": name,
                                "seed": seed,
                                "mutation_count": count,
                                "n": len(selected),
                                **metrics(
                                    [float(r["observed_log_fitness"]) for r in selected],
                                    [float(r["predicted_log_fitness"]) for r in selected],
                                ),
                                "censored_fraction": sum(r["censored"] == "True" for r in selected)
                                / len(selected),
                            }
                        )
                for label in ("unseen_pair", "rare_pair_1_2", "repeated_pair"):
                    selected = [r for r in subset if rarity[r["id"]] == label]
                    if selected:
                        rare.append(
                            {
                                "protocol": protocol,
                                "model": name,
                                "seed": seed,
                                "pair_slice": label,
                                "n": len(selected),
                                **metrics(
                                    [float(r["observed_log_fitness"]) for r in selected],
                                    [float(r["predicted_log_fitness"]) for r in selected],
                                ),
                            }
                        )
            if len(local) > 1:
                for base in (
                    "single_additive",
                    "hybrid_zero3d",
                    "hybrid_shuffled_distance",
                    "hybrid_shuffled_biochemical",
                ):
                    if name == base:
                        continue
                    baseline = [r for r in runs if r["protocol"] == protocol and r["model"] == base]
                    for run in local:
                        matched = next(
                            (r for r in baseline if r["seed"] == run["seed"]), baseline[0]
                        )
                        paired.append(
                            {
                                "protocol": protocol,
                                "model": name,
                                "baseline": base,
                                "seed": run["seed"],
                                "rmse_improvement": matched["test"]["rmse"] - run["test"]["rmse"],
                                "epsilon_WT_mae_improvement": matched["epistasis_WT"]["mae"]
                                - run["epistasis_WT"]["mae"],
                            }
                        )
    write_json(output / "metrics.json", summary)
    write_csv(output / "mutation_count_slices.csv", slices)
    write_csv(output / "rare_pair_slices.csv", rare)
    write_csv(output / "paired_seed_changes.csv", paired)
    lookup_metrics = {(s["protocol"], s["model"]): s for s in summary}
    bootstrap = []
    for protocol in ("C1", "C2"):

        def ensemble(name, column, epistasis=False, protocol=protocol):
            data = [
                r
                for r in (eps if epistasis else predictions)
                if r["protocol"] == protocol
                and r["model"] == name
                and (not epistasis or r["background_mutations"] == "0")
            ]
            keys = {}
            for r in data:
                key = r["target"] if epistasis else r["id"]
                keys.setdefault(key, []).append(float(r[column]))
            return {k: float(np.mean(v)) for k, v in keys.items()}

        for name, base in [
            ("hybrid_pairs", "single_additive"),
            ("hybrid_pairs", "hybrid_zero3d"),
            ("hybrid_pairs", "hybrid_shuffled_distance"),
            ("hybrid_pairs", "hybrid_shuffled_biochemical"),
            ("structural_pairs", "sequence_pairs"),
        ]:
            for scope in ("fitness", "WT_epistasis"):
                direct = scope != "fitness"
                column = "predicted_log_epistasis" if direct else "predicted_log_fitness"
                a, b = ensemble(name, column, direct), ensemble(base, column, direct)
                keys = sorted(a)
                assert set(a) == set(b)
                truth = ensemble(
                    name, "measured_log_epistasis" if direct else "observed_log_fitness", direct
                )
                groups = ["-".join(str(int(t[1:-1])) for t in lookup[k]["mutations"]) for k in keys]
                result = paired_block_interval(
                    np.array([truth[k] for k in keys]),
                    np.array([a[k] for k in keys]),
                    np.array([b[k] for k in keys]),
                    groups,
                    "mae" if direct else "rmse",
                    config["bootstrap"],
                )
                bootstrap.append(
                    {
                        "protocol": protocol,
                        "model": name,
                        "baseline": base,
                        "scope": scope,
                        **result,
                    }
                )
    write_json(output / "paired_bootstrap.json", bootstrap)
    unique = {}
    for r in eps:
        unique.setdefault((r["protocol"], r["target"], r["background"], r["position_pair"]), r)
    scale = []
    for protocol in ("C1", "C2"):
        for context in ("WT", "nonWT"):
            local = [
                r
                for r in unique.values()
                if r["protocol"] == protocol
                and (r["background_mutations"] == "0") == (context == "WT")
            ]
            raw = np.array([float(r["measured_linear_epistasis"]) for r in local])
            log = np.array([float(r["measured_log_epistasis"]) for r in local])
            scale.append(
                {
                    "protocol": protocol,
                    "background": context,
                    "n": len(local),
                    "opposite_sign_fraction": float(np.mean(raw * log < 0)) if len(local) else None,
                    "median_absolute_log_epistasis": float(np.median(np.abs(log)))
                    if len(local)
                    else None,
                    "median_absolute_linear_epistasis": float(np.median(np.abs(raw)))
                    if len(local)
                    else None,
                }
            )
    write_json(output / "scale_sensitivity.json", scale)
    # Representative large measured contrasts, chosen by truth magnitude, not model success.
    examples = []
    for protocol in ("C1", "C2"):
        local = [
            r
            for r in unique.values()
            if r["protocol"] == protocol and r["background_mutations"] == "0"
        ]
        for r in sorted(local, key=lambda v: -abs(float(v["measured_log_epistasis"])))[:10]:
            model = [
                e
                for e in eps
                if e["protocol"] == protocol
                and e["model"] == "hybrid_pairs"
                and e["target"] == r["target"]
                and e["background"] == r["background"]
                and e["position_pair"] == r["position_pair"]
            ]
            examples.append(
                {
                    "protocol": protocol,
                    "target": r["target"],
                    "position_pair": r["position_pair"],
                    "measured_log_epistasis": float(r["measured_log_epistasis"]),
                    "hybrid_predicted_epistasis": float(
                        np.mean([float(e["predicted_log_epistasis"]) for e in model])
                    ),
                    "additive_predicted_epistasis": 0,
                    "minimum_constituent_input_count": min(
                        lookup[r[k]]["count_input"] for k in ("target", "a", "b", "background")
                    ),
                }
            )
    write_csv(output / "nonadditive_examples.csv", examples)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, protocol in zip(axes, ("C1", "C2"), strict=True):
        ax.barh(
            np.arange(len(names)),
            [lookup_metrics[protocol, n]["rmse_mean"] for n in names],
            xerr=[lookup_metrics[protocol, n]["rmse_seed_sd"] for n in names],
            capsize=2,
            color=["#167d9a" if n == "hybrid_pairs" else "#809ba5" for n in names],
        )
        ax.set_yticks(np.arange(len(names)), names)
        ax.set(title=protocol, xlabel="Held-out log-fitness RMSE (lower is better)")
    axes[0].invert_yaxis()
    fig.suptitle("M11: identical variants, same-experiment count-derived targets; seed SD")
    fig.tight_layout()
    fig.savefig(output / "fitness_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, protocol in zip(axes, ("C1", "C2"), strict=True):
        local = [
            r
            for r in eps
            if r["protocol"] == protocol
            and r["model"] == "hybrid_pairs"
            and r["background_mutations"] == "0"
            and r["seed"] == "0"
        ]
        ax.scatter(
            [float(r["measured_log_epistasis"]) for r in local],
            [float(r["predicted_log_epistasis"]) for r in local],
            s=12,
            alpha=0.5,
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(
            title=protocol + " WT-background pairs, seed0",
            xlabel="Measured log epistasis",
            ylabel="Predicted four-point contrast",
        )
    fig.tight_layout()
    fig.savefig(output / "epistasis.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    local = [
        r for r in unique.values() if r["protocol"] == "C1" and r["background_mutations"] == "0"
    ]
    ax.scatter(
        [float(r["measured_linear_epistasis"]) for r in local],
        [float(r["measured_log_epistasis"]) for r in local],
        s=12,
        alpha=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(
        title="Same measured factorials, different effect scales",
        xlabel="Raw-enrichment four-point contrast",
        ylabel="Log-enrichment four-point contrast",
    )
    fig.tight_layout()
    fig.savefig(output / "scale_sensitivity.png", dpi=160)
    plt.close(fig)

    def number(v):
        return "undefined" if v is None else f"{v:.4f}"

    fitness_table = ["| Model | C1 rho | C1 RMSE | C2 rho | C2 RMSE |", "|---|---:|---:|---:|---:|"]
    epsilon_table = [
        "| Model | C1 WT epsilon rho | C1 MAE | C2 WT epsilon rho | C2 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        a, b = lookup_metrics["C1", name], lookup_metrics["C2", name]
        fitness_table.append(
            f"| {name} | {number(a['rho_mean'])} | {a['rmse_mean']:.4f} | {number(b['rho_mean'])} | {b['rmse_mean']:.4f} |"
        )
        epsilon_table.append(
            f"| {name} | {number(a['epsilon_WT']['rho_mean'])} | {a['epsilon_WT']['mae_mean']:.4f} | {number(b['epsilon_WT']['rho_mean'])} | {b['epsilon_WT']['mae_mean']:.4f} |"
        )
    background_table = [
        "| Model | C1 non-WT epsilon rho | C1 MAE | C2 non-WT epsilon rho | C2 MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("single_additive", "esm_ridge", "hybrid_pairs", "hybrid_zero3d"):
        a = lookup_metrics["C1", name]["epsilon_nonWT"]
        b = lookup_metrics["C2", name]["epsilon_nonWT"]
        background_table.append(
            f"| {name} | {number(a['rho_mean'])} | {a['mae_mean']:.4f} | {number(b['rho_mean'])} | {b['mae_mean']:.4f} |"
        )
    effects = []
    for protocol in ("C1", "C2"):
        for baseline in (
            "single_additive",
            "hybrid_zero3d",
            "hybrid_shuffled_distance",
            "hybrid_shuffled_biochemical",
        ):
            local = [
                p
                for p in paired
                if p["protocol"] == protocol
                and p["model"] == "hybrid_pairs"
                and p["baseline"] == baseline
            ]
            values = [p["rmse_improvement"] for p in local]
            evalues = [p["epsilon_WT_mae_improvement"] for p in local]
            effects.append(
                f"- {protocol} hybrid_pairs versus {baseline}: fitness RMSE improvement {np.mean(values):+.4f} (seed SD {np.std(values):.4f}); WT epistasis MAE improvement {np.mean(evalues):+.4f} (SD {np.std(evalues):.4f})."
            )
    bootlines = []
    for b in bootstrap:
        if b["model"] == "hybrid_pairs" and b["baseline"] == "hybrid_zero3d":
            interval = (
                "unavailable (one position-pair block)"
                if b["low"] is None
                else f"[{b['low']:+.4f}, {b['high']:+.4f}]"
            )
            bootlines.append(
                f"- {b['protocol']} {b['scope']} explicit3D improvement, seed-mean ensemble:95% paired block interval {interval}; {b['blocks']} blocks."
            )
    source = audit["data"]
    m = manifest
    lines = [
        "# M11: explicit mutation interactions and measured epistasis",
        "",
        "## Real benchmark and results",
        "",
        "Original measured Wu2016 GB1 landscape, not the fitness-biased FLIP sampled subset. Same-experiment input/selected counts define all fitness labels, including singles; source149,361 measurements, bounded cohort2,648 (WT1/singles76/doubles2,091/triples360/quads120).64 probe-only lower-order genotypes support factorial evaluation and never train. C1 roles1,877/385/386, C2 roles1,569/425/654. C2 excludes pair39–54 on every training/validation background and retains all alleles through training singles. No whole-protein or unseen-position/family claim.",
        "",
        "54 model/seed/protocol cases produce28,080 matched held-out predictions. Neural means average three seeds; deterministic baselines have one fit. Ranking and RMSE describe ln(max(count-normalized enrichment,0.01)); no prediction clipping. This is not a common-assay percentile or thermodynamic scale.",
        "",
        *fitness_table,
        "",
        "The findings are mixed. C1 hybrid pair modeling improves WT epistasis over additivity, while ESM has better fitness ranking but weaker WT epistasis correlation. One-hot ridge ranks fitness well while implying exactly zero epistasis: predictive accuracy alone is not evidence of interaction recovery. Explicit3D does not beat the zero3D or shuffled-distance controls. On C2, the real hybrid's epistasis MAE is worse than additivity, and its fitness ranking degrades strongly. No structural/causal benefit is established.",
        "",
        "## Direct factorial epistasis",
        "",
        "Measured and predicted epistasis each use their own four terms: f(AB)-f(A)-f(B)+f(BG). Predicted contrasts use model predictions of all four, not noisy observed constituent labels as an oracle offset. Require all four actually measured and >=0.01; never fabricate singles/doubles or exact epistasis for censored observations. WT-background and non-WT background-specific pair contrasts are separate.",
        "",
        *epsilon_table,
        "",
        "C1 tests172 WT-background and68 non-WT contrasts; C2 tests213 WT-background and275 non-WT contrasts. Non-WT pairs share target genotypes and lower-order terms, so these counts are not independent experimental replicates:",
        "",
        *background_table,
        "",
        "The explicit hybrid does not improve non-WT contrast MAE over additivity. Its C2 fitness RMSE increases with mutation burden (doubles1.9180, triples2.4933, quadruples3.5734). That failure combines unseen-pairing/depth shift with a model constrained to background-independent pair contributions; it is not repaired or retuned after inspecting tests.",
        "",
        f"Across the declared cohort,{source['eligible_measured_contrasts']} contrasts are eligible (1,125 WT,374 non-WT); exclusions: {source['contrast_exclusions']}. Per-test counts and non-WT metrics are in runs.json/metrics.json. Constant/additive/count/one-hot predictions have zero model epistasis and undefined correlation, not a successful zero correlation. Numerical four-point roundoff below1e-10 is zeroed.",
        "",
        *effects,
        "",
        *bootlines,
        "",
        "paired_bootstrap.json uses1,000 paired block resamples of seed-mean predictions, with mutation-position-set blocks for fitness and position-pair blocks for WT epistasis. C1 has few blocks and C2 WT only one, so that interval is unavailable. These conditional pilot intervals do not quantify shared single-measurement noise, training uncertainty or biological replicate variance. The original counts have no replicate-level error model here.",
        "",
        "## Data and scale limitations",
        "",
        "Primary source: [Wu et al.2016](https://doi.org/10.7554/eLife.16965), [measured supplement1](https://doi.org/10.7554/eLife.16965.024), [official FLIP source README](https://github.com/J-SNACKKB/FLIP/blob/62cace8735f5610e2743cf06ce0f944b37fffaa6/splits/gb1/README.md). Archive2.66 MB, extracted46.4 MB, commit/checksums/licensing pinned in configs/m11.json. Original CC-BY-4.0, FLIP derivative AFL-3.0. No imputed supplement2 labels or keep/fitness-biased source sampling.",
        "",
        "The published Fitness singles were imported from Olson2014 and scaled1.159. M11 instead computes every genotype from Wu selected/input counts divided by the WT ratio (WT input92,735/selected338,346). Non-single published values reproduce this formula; median absolute difference between count-derived and published singles is "
        + f"{source['single_published_vs_count_median_absolute_difference']:.4f}. This avoids mixing assay batches but uses noisier single counts (minimum input14, some selected0). Global log-normalization cancels in four-point contrasts, but shared constituent/WT-background measurements still induce dependence. No precision or causal-mechanism claim.",
        "",
        "The floor0.01 reflects an approximate reported detection limit. Fitness evaluation retains censored values, so RMSE evaluates a floored observation, not an exact latent fitness. Direct epistasis excludes censored constituents, selecting a more measurable subset and potentially biasing its distribution. scale_sensitivity.json/plot compare raw-enrichment and log-enrichment contrasts of the same four measurements; nonlinear transformation changes both magnitude and sometimes sign. Do not conflate these units or treat epistasis as scale independent.",
        "",
        "FLIP inferred a full reference from a structure; only the compatible56-residue GB1 core is modeled. The inferred fusion tail is not claimed as an experimentally verified assay construct. IgG-Fc mRNA-display affinity selection can reflect folding and binding, not isolated thermodynamic stability. Numerical selection pH/temperature were not invented from library-preparation conditions.",
        "",
        "## Leakage, structure and interpretation",
        "",
        "All alleles are known from permitted training singles. C1 permits pairs seen in different training backgrounds, so it tests whole-genotype holdout only. C2 guarantees the test position pairing is absent from both training and validation; all quadruples are also test, creating mutation-depth shift. mutation_count_slices.csv separates doubles/triples/quads; rare_pair_slices.csv reports unseen and rare exact mutation-pair occurrences in training. No test/probe outcome enters a feature, scaler, additive delta or selection. Test-label perturbations reproduce fitted models unchanged.",
        "",
        "Validated1PGA geometry has55 compatible nodes; known Q2T at residue2 is excluded, and all four mutated sites map exactly. Distances, contact neighborhoods, components/paths, source-confidence and missing masks remain distinct. B-factors are A², not pLDDT or predictive uncertainty. Only six position pairs exist; geometry can encode positional identity, so improvements over shuffled/zero controls cannot establish a physical mechanism. Reference geometry is not measured mutant rearrangement.",
        "",
        "Measured-single additivity plus symmetric ridge/MLP pair residuals supports any mutation set with known features. WT/singles have empty pairs and unchanged additive predictions. Pair contributions use only that unordered pair and frozen reference-single context, so these explicit predictors are pairwise sums; they do not estimate true higher-order coefficients. Background-specific measured pair contrasts can vary for noise or higher-order reasons. The full-mutant ESM/hybrid probes model implicit interactions, but fitness accuracy alone is not evidence of measured epistasis.",
        "",
        "Plain sequence_pairs and structural_pairs have identical420-input MLP capacity (explicit3D zeroed for sequence). Hybrid uses720 dimensions including symmetric frozen reference-single ESM/GNN/bidirectional context; zero3D and shuffled hybrid controls have identical capacity. The no-explicit3D hybrid still has implicit structure. Shuffled3D moves descriptor records among the six unordered position pairs; biochemical control permutes explicit destination identity/property records within each position. Frozen contexts stay correct, so controls isolate explicit increments rather than removing all biological information. Reference permutations are label-blind and consistent across roles.",
        "",
        "nonadditive_examples.csv selects large measured WT contrasts by absolute truth magnitude, including errors, rather than successful model examples. Generalization and control differences should be interpreted alongside scarce structural diversity, noisy singles, censoring and low-order sampling. No source/model was changed after inspecting held-out outcomes.",
        "",
        "## Reproducibility and compute",
        "",
        f"Run:{m['elapsed_seconds']:.2f}s on {m['machine']} CPU/four threads; peak RSS {m['peak_rss_bytes'] / 1024**3:.3f} GiB. Frozen preparation:{audit['features']['elapsed_seconds']:.2f}s including local MPS ESM token extraction/cache access; checkpoint/frozen pretraining costs excluded. Versions:"
        + ", ".join(f"{k} {v}" for k, v in m["versions"].items())
        + ".",
        "",
        "Frozen ESM t12 35M revision6fbf070e... and M7 function-only encoder trained old amylase/validated NucB, with GB1 held out. No historical GB1-supervised head is loaded; PLM pretraining exposure unknown. No M8 stability scalar. Source/reference/structure/model/config/code hashes and devices are archived; binary caches/weights/raw archive stay ignored. Protocol/config/code/tests committed before fitting.",
        "",
        "Exact reproduction follows docs/m11_protocol.md: download pinned archive, prepare, run epistasis_experiments, verify_m11 and report_m11. Report-only: `MPLCONFIGDIR=/tmp/proteinmod-m11-matplotlib python scripts/report_m11.py --input benchmarks/m11 --output runs/m11-report`. Full repeat changes only output. Replay/repeat results are archived separately. All238 M0–M10 benchmark artifacts and15 prior configurations are unchanged. Stacked on verified #20 → #18 → #16 → #14 → #12, all50 dependency checks passed; main unchanged.",
        "",
        "No M12 family-held-out expansion, new encoder training, formal uncertainty calibration, LoRA or epistasis-as-causality claim.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    files = [
        "metrics.json",
        "mutation_count_slices.csv",
        "rare_pair_slices.csv",
        "paired_seed_changes.csv",
        "paired_bootstrap.json",
        "scale_sensitivity.json",
        "nonadditive_examples.csv",
        "fitness_comparison.png",
        "epistasis.png",
        "scale_sensitivity.png",
        "REPORT.md",
    ]
    write_json(
        output / "report_manifest.json",
        {
            "prediction_sha256": checksum(root / "predictions.csv"),
            "epistasis_prediction_sha256": checksum(root / "epistasis_predictions.csv"),
            "files": {n: checksum(output / n) for n in files},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m11"))
    parser.add_argument("--output", type=Path, default=Path("runs/m11"))
    args = parser.parse_args()
    report(args.input, args.output)
