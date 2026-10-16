"""Recompute family-balanced results from actual held-out M12 predictions."""

import csv
import json
from collections import defaultdict

import numpy as np

from proteinmod.data import write_json
from proteinmod.family_experiments import aggregate
from proteinmod.structure_pipeline import write_csv


def summarize(rows, predictions, common_only=False):
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    groups = defaultdict(list)
    for p in predictions:
        if common_only and rows[lookup[p["id"]]]["partition"] != "test":
            continue
        groups[(p["cohort"], p["protocol"], p["model"], int(p["seed"]))].append(p)
    results = []
    for (cohort, protocol, model, seed), part in sorted(groups.items()):
        ids = [p["id"] for p in part]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated held-out genotype in protocol summary")
        values = aggregate(rows, [lookup[i] for i in ids], [float(p["prediction"]) for p in part])
        results.append(
            dict(cohort=cohort, protocol=protocol, model=model, seed=seed, n=len(part), **values)
        )
    return results


def report(config, output):
    rows = json.loads((output / "rows.json").read_text())
    with (output / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    runs = json.loads((output / "runs.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    results = summarize(rows, predictions)
    common = summarize(rows, predictions, True)
    write_json(
        output / "summary.json", {"all_test_variants": results, "common_G1_test_variants": common}
    )
    tables = []
    for scope, values in [("all", results), ("common_G1_test", common)]:
        grouped = defaultdict(list)
        for r in values:
            for f, value in r["per_family"].items():
                grouped[(r["cohort"], r["protocol"], r["model"], f)].append(value)
        for (cohort, protocol, model, family), v in sorted(grouped.items()):
            tables.append(
                {
                    "scope": scope,
                    "cohort": cohort,
                    "protocol": protocol,
                    "model": model,
                    "family": family,
                    "mean_rho": float(np.mean(v)),
                    "seed_sd": float(np.std(v)),
                    "seeds": len(v),
                }
            )
    write_csv(output / "per_family.csv", tables)
    lines = [
        "# M12: family-held-out domain mutation-effect benchmark",
        "",
        "Actual locally executed predictions. Eight experimentally measured Tsuboyama 2023 domains, four externally annotated Pfam families (two proteins each), 96 label-blind sampled variants per protein. This is **stability mutation-effect transfer across small domain families**, not biochemical function transfer or unseen-fold generalization. Assays retain native processed DMS scores; models predict training-assay ranks. No pooled raw RMSE or thermodynamic calibration.",
        "",
        "## Primary matched-cohort results",
        "",
        "Each number averages proteins equally within each family, then four families equally, then seeds. Family SD describes the four effective replication units; it is not a confidence interval. Ridge is deterministic; neural runs use three fixed seeds.",
        "",
        "| Model | G1 ρ | G2 ρ | G3 ρ | G3 family SD | G3 seed SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    models = sorted({r["model"] for r in results if r["cohort"] == "structure_matched"})
    means = {}
    for model in models:
        means[model] = {}
        for protocol in ("G1", "G2", "G3"):
            part = [
                r
                for r in results
                if r["cohort"] == "structure_matched"
                and r["model"] == model
                and r["protocol"] == protocol
            ]
            means[model][protocol] = float(np.mean([r["family_macro_spearman"] for r in part]))
        families = [
            r["mean_rho"]
            for r in tables
            if r["scope"] == "all"
            and r["cohort"] == "structure_matched"
            and r["model"] == model
            and r["protocol"] == "G3"
        ]
        seeds = [
            r["family_macro_spearman"]
            for r in results
            if r["cohort"] == "structure_matched" and r["model"] == model and r["protocol"] == "G3"
        ]
        lines.append(
            f"| {model} | {means[model]['G1']:.4f} | {means[model]['G2']:.4f} | {means[model]['G3']:.4f} | {np.std(families):.4f} | {np.std(seeds):.4f} |"
        )
    lines += [
        "",
        "G1 evaluates 16 variants/protein; G2/G3 evaluate all 96 variants of their held-out proteins. G2/G3 retain all original within-protein partition tags but assign complete proteins/families to roles. Sequence-all and structure-matched happen to coincide here: all 768 variants have usable mutation sites. Their sequence baseline predictions must agree exactly.",
        "",
        "## Generalization gaps on identical test variants",
        "",
        "Secondary diagnostic restricts all protocols to the same original G1 test IDs (16 per protein), using already-saved predictions without refitting or selection. Positive gaps mean worse held-out generalization.",
        "",
        "| Model | Common G1 ρ | Common G2 ρ | Common G3 ρ | G1−G2 | G2−G3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        v = [
            float(
                np.mean(
                    [
                        r["family_macro_spearman"]
                        for r in common
                        if r["cohort"] == "structure_matched"
                        and r["model"] == model
                        and r["protocol"] == protocol
                    ]
                )
            )
            for protocol in ("G1", "G2", "G3")
        ]
        lines.append(
            f"| {model} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[0] - v[1]:.4f} | {v[1] - v[2]:.4f} |"
        )
    lines += [
        "",
        "## Per-family G3 results",
        "",
        "| Model | " + " | ".join(config["family_order"]) + " |",
        "|---|" + "---:|" * len(config["family_order"]),
    ]
    for model in models:
        v = [
            next(
                r
                for r in tables
                if r["scope"] == "all"
                and r["cohort"] == "structure_matched"
                and r["model"] == model
                and r["protocol"] == "G3"
                and r["family"] == f
            )
            for f in config["family_order"]
        ]
        lines.append(
            "| "
            + model
            + " | "
            + " | ".join(f"{r['mean_rho']:.4f} ± {r['seed_sd']:.4f}" for r in v)
            + " |"
        )
    overfitting = [
        {
            "protocol": r["protocol"],
            "fold": r["fold"],
            "model": r["model"],
            "seed": r["seed"],
            "train_rho": r["selection"]["train_macro_spearman"],
            "validation_rho": r["selection"]["validation_macro_spearman"],
            "train_validation_gap": r["selection"]["train_macro_spearman"]
            - r["selection"]["validation_macro_spearman"],
            "selected_epoch": r["selection"].get("epoch"),
            "parameters": r["selection"].get("parameters"),
        }
        for r in runs
        if r["cohort"] == "structure_matched"
    ]
    write_csv(output / "overfitting.csv", overfitting)
    elapsed = sum(r["elapsed_seconds"] for r in runs)
    lines += [
        "",
        "## Audits and compute",
        "",
        f"{len(runs)} actual fits, {len(predictions)} held-out predictions; summed fit time {elapsed:.1f} s. Exact versions, peak process RSS and full wall time are in environment.json. Feature extraction/device times and tensor hashes are in audit.json; large frozen tokens and model checkpoints remain ignored local caches.",
        "",
        f"Preserved {len(audit['historical_sha256'])} M0–M11 benchmark/configuration file hashes. No original partition or historical result was overwritten.",
        "",
        "All G2 roles are protein-disjoint with test/validation families represented in training. All G3 roles are family-disjoint. Global/local identity and aligned-block audits, ambiguity flags, and 40/60/80% coverage-filtered sequence clusters are retained. No unresolved cross-family cluster/partial-domain match passed the declared screening rules. External family definitions are Pfam 34.0 domain features on the deposited polymer entities, not whole-protein annotation names. Full-length annotations may describe additional domains absent from these assay constructs.",
        "",
        "Graphs are reference geometry only. Mapping/construct insertions, connectivity, missing descriptors and removal masks are recorded per protein. PITX2 selects chain P; SAV1 uses the declared exact reference polymer span 7–49 to resolve a cloning-tag alignment ambiguity. Seven structures are NMR and BBC1/1TG0 is crystallographic. All confidence descriptors were conservatively masked in this fixed experiment; B columns are not interpreted as pLDDT or model uncertainty. See ERRATA.md for the preregistered BBC1 display-name typo. Mutation masks and substituted features follow the M3/M5 tested interfaces.",
        "",
        "Target midranks and feature scaling use training only. Ridge alpha and neural epochs use validation only. Balanced validation protein counts give equal family weights. Every G3 encoder is randomly initialized; no M0–M11 supervised representation is transferred. ESM-2 is frozen at an immutable revision; unsupervised pretraining exposure and structure-database exposure cannot be excluded.",
        "",
        "## Interpretation and failure analysis",
        "",
        "Transfer is model- and family-dependent; compare the per-family table before interpreting the macro average. Negative correlations are retained. The four family outcomes are descriptive evidence, not a population-wide family-transfer estimate. Seed variation measures optimization sensitivity, not additional independent families.",
        "",
        "G1 has fewer training variants per protein than G2/G3 have per training protein; G3 additionally trains on only two families. Thus gaps mix reduced family diversity, training-set size and task difficulty. The common-test diagnostic removes test-genotype composition differences, not these training differences. With one assay per construct, protein/family and construct-specific assay effects remain inseparable. Shared study methodology reduces but cannot eliminate that confound. The cap, short domains and 12-epoch budget favor speed and are not a performance ceiling.",
        "",
        "Validation traces and training correlations in runs.json expose early stopping and overfitting; no larger search was launched after inspecting tests. Constant/undefined correlations are explicitly null per protein and zero only for macro selection/aggregation. No CI is reported from four families and no mutation-level significance claim is made. Additional independent families, assay endpoints and paired homolog measurements are needed before broad functional or unseen-fold claims.",
        "",
        "## Reproduction",
        "",
        "See DATA_PROVENANCE.md and docs/M12.md for pinned acquisition, offline verification, tests and exact commands. The preregistration/config were committed before fitting. No M13 work was added.",
        "",
        "![Family-balanced results](family_results.png)",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    x = np.arange(len(models))
    for j, protocol in enumerate(("G1", "G2", "G3")):
        ax.bar(x + (j - 1) * 0.25, [means[m][protocol] for m in models], width=0.25, label=protocol)
    ax.set(
        xticks=x,
        xticklabels=models,
        ylabel="Equal-family macro Spearman",
        title="Four families, eight measured domains; matched structural cohort",
    )
    ax.tick_params(axis="x", labelrotation=20)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend()
    fig.savefig(output / "family_results.png", dpi=160)
    plt.close(fig)
