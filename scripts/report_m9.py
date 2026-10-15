"""Produce assay-local M9 tables/plots from archived predictions and measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proteinmod.data import checksum, write_json
from proteinmod.structure_pipeline import write_csv


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    runs = json.loads((root / "runs.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    with (root / "predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    models = list(dict.fromkeys(r["model"] for r in runs))
    summary, assays, differences = [], [], []
    for protocol in ("G1", "G2"):
        for model in models:
            local = [r for r in runs if r["protocol"] == protocol and r["model"] == model]
            seed_scores = {
                s: float(np.mean([r["test"]["macro"] for r in local if r["seed"] == s]))
                for s in sorted({r["seed"] for r in local})
            }
            summary.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "mean_spearman": float(np.mean(list(seed_scores.values()))),
                    "seed_sd": float(np.std(list(seed_scores.values()))),
                    "seed_scores": seed_scores,
                    "elapsed_fit_seconds": sum(r["elapsed_seconds"] for r in local),
                }
            )
            for assay in sorted({a for r in local for a in r["test"]["assays"]}):
                values = [r["test"]["assays"][assay] for r in local if assay in r["test"]["assays"]]
                assays.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "assay_id": assay,
                        "protein_id": values[0]["protein_id"],
                        "family": values[0]["family"],
                        "n": values[0]["n"],
                        "mean_spearman": float(np.mean([v["spearman"] or 0 for v in values])),
                        "seed_sd": float(np.std([v["spearman"] or 0 for v in values])),
                    }
                )
        for seed in manifest["config"]["seeds"]:
            for mode in (
                "structured",
                "shuffled_train",
                "with_assay_id",
                "protein_proxy",
                "structured_missing_at_test",
            ):
                a = next(s for s in summary if s["protocol"] == protocol and s["model"] == mode)
                b = next(s for s in summary if s["protocol"] == protocol and s["model"] == "zero")
                differences.append(
                    {
                        "protocol": protocol,
                        "model": mode,
                        "seed": seed,
                        "delta_spearman_vs_zero": a["seed_scores"][seed] - b["seed_scores"][seed],
                    }
                )
    write_json(output / "metrics.json", summary)
    write_csv(output / "per_assay.csv", assays)
    write_csv(output / "paired_seed_changes.csv", differences)
    lookup = {(r["protocol"], r["model"]): r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, protocol in zip(axes, ("G1", "G2"), strict=True):
        values = [lookup[protocol, m]["mean_spearman"] for m in models]
        errors = [lookup[protocol, m]["seed_sd"] for m in models]
        ax.barh(
            np.arange(len(models)),
            values,
            xerr=errors,
            color=["#167d9a" if m == "structured" else "#809ba5" for m in models],
            capsize=2,
        )
        ax.set_yticks(np.arange(len(models)), models)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(protocol + (": within protein" if protocol == "G1" else ": held-out protein"))
        ax.set_xlabel("Assay-local Spearman; equal-protein macro")
    axes[0].invert_yaxis()
    fig.suptitle("M9: crossed abundance/activity assays (three proteins)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, protocol in zip(axes, ("G1", "G2"), strict=True):
        for mode in ("zero", "structured", "shuffled_train"):
            local = [
                r
                for r in runs
                if r["protocol"] == protocol and r["model"] == mode and r["seed"] == 0
            ]
            for j, run in enumerate(local):
                trace = run["selection"]["trace"]
                ax.plot(
                    [x["epoch"] for x in trace],
                    [x["validation_macro"] for x in trace],
                    label=mode if j == 0 else None,
                    alpha=0.8 if protocol == "G1" else 0.45,
                )
        ax.set(
            title=protocol + " validation trajectories, seed0",
            xlabel="Epoch",
            ylabel="Validation macro Spearman",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", dpi=160)
    plt.close(fig)
    table = ["| Model | G1 rho | G2 rho |", "|---|---:|---:|"]
    for m in models:
        table.append(
            f"| {m} | {lookup['G1', m]['mean_spearman']:.4f} | {lookup['G2', m]['mean_spearman']:.4f} |"
        )
    per_protein = [
        "| Held-out protein | Endpoint | Zero | Structured | Shuffled |",
        "|---|---|---:|---:|---:|",
    ]
    for protein in manifest["config"]["protein_order"]:
        for family in ("abundance", "activity"):
            values = [
                next(
                    a["mean_spearman"]
                    for a in assays
                    if a["protocol"] == "G2"
                    and a["protein_id"] == protein
                    and a["family"] == family
                    and a["model"] == m
                )
                for m in ("zero", "structured", "shuffled_train")
            ]
            per_protein.append(
                f"| {protein} | {family} | " + " | ".join(f"{v:.4f}" for v in values) + " |"
            )
    data = audit["data"]["proteins"]
    provenance = [
        "| Protein | Common genotypes | Selected | Original rows by assay |",
        "|---|---:|---:|---|",
    ]
    for p, d in data.items():
        provenance.append(
            f"| {p} | {d['common_genotypes']} | {d['selected_genotypes']} | "
            + "; ".join(f"{a}: {n}" for a, n in d["source_counts"].items())
            + " |"
        )
    confidence = audit["features"]["proteins"]

    def effect(protocol, mode):
        values = [
            d["delta_spearman_vs_zero"]
            for d in differences
            if d["protocol"] == protocol and d["model"] == mode
        ]
        return f"{np.mean(values):+.4f} (seed SD {np.std(values):.4f}; range {min(values):+.4f} to {max(values):+.4f})"

    lines = [
        "# M9: assay-aware mutation-effect prediction",
        "",
        "## Verified comparison",
        "",
        "Real local experiments on six ProteinGym v1.3 assays: PTEN, CYP2C9 and VKOR1, each with abundance/activity measurements of the same180 genotypes. 1,080 observations; G1 has756/162/162 train/validation/test rows, each G2 fold360/360/360. All92 model/seed/fold cases use matched test IDs (28,566 predictions). Native outcomes remain assay specific. Table values are assay-local Spearman, then endpoint/protein macro; neural means average three seeds, ridge is deterministic.",
        "",
        *table,
        "",
        f"Primary structured-minus-zero change: G1 {effect('G1', 'structured')}; G2 {effect('G2', 'structured')}. Shuffled-training-minus-zero G2: {effect('G2', 'shuffled_train')}. These are seed variation, not population confidence intervals.",
        "",
        "## Held-out protein/endpoint failures",
        "",
        *per_protein,
        "",
        "Conditioning is not uniformly beneficial. Interpret changes alongside the shuffled control and the stronger classical baselines; an improvement over the same neural head alone does not establish a general benefit over existing representations. Individual assay IDs remain protein-specific and map to UNK in G2. The protein-proxy control is diagnostic, not a deployable input. Separate assay-ID ridge heads fall back to the global head for all unseen test assays.",
        "",
        "Shuffling swaps complete endpoint records within each training protein (two families, so every nonidentity permutation is a swap). Correct validation/test metadata stay unchanged. This is an adversarial mismatch control, not a null distribution from many independent random annotations. Structured metadata includes endpoint-specific technologies/substrates that can be unseen at test; explicit UNK handling limits transfer.",
        "",
        "## Data, mapping and provenance",
        "",
        *provenance,
        "",
        "Sampling is label-blind sequence SHA256 on the intersection of experimental genotypes. Unpaired counts and cap exclusions remain in audit.json/source counts; a small intersected cohort can select for variants measurable by both assays. All selected mutations are single substitutions. Experimental score is processed DMS_score with higher-is-better direction, not kcal/mol; raw phenotype names/directions and source hashes are in the configuration. The ProteinGym MIT curated archive and original paper citations are retained. Structures are AlphaFold DB v6, CC-BY-4.0.",
        "",
        "Every selected genotype maps to a full exact reference/structure alignment: zero mapping exclusions. Geometry is reference-predicted geometry, not observed mutant rearrangement. Fractions of all reference residues with pLDDT<50: "
        + "; ".join(f"{p} {d['plddt_below_50_fraction']:.3f}" for p, d in confidence.items())
        + ". Low confidence was not excluded after examining outcomes. Connectivity and mappings are in audit.json.",
        "",
        "Temperature and pH are unavailable in all six curated metadata records (0/6 observed each); unavailable is distinct from not_applicable and observed zero. No fabricated physiological defaults. Numerical condition effects cannot be estimated here. Activity is a heterogeneous proxy: PTEN yeast growth, CYP probe activity, VKOR carboxylation reporter. This collection supports known-endpoint-family protein holdout, not novel-assay or unseen-family generalization. Only three proteins limit certainty.",
        "",
        "## Leakage, representation and compute",
        "",
        "Both endpoint observations of a genotype stay together in G1. G2 proteins are entirely disjoint. Vocabularies, standardization and target midranks fit training only; alpha/epoch selection uses validation only. Primary metadata excludes protein/assay IDs and all outcomes. Per-assay native correlations are aggregated only after evaluation. Test-label perturbation tests reproduce both model fits and selections unchanged.",
        "",
        "Frozen ESM-2 t12 35M revision6fbf070e65b0b7291e7bbcd451118c216cff79d8 and M7 function-only GNN/bidirectional encoder are reused; only the new heads train. Fixed encoder supervision was old amylase training/NucB validation; similarity audits screen the new proteins against both. PLM pretraining exposure is unknown. No M6 stability prediction head or M8 OOD stability scalar is read; no thermodynamic inference. The pretrained frozen encoder may underfit new human proteins; end-to-end adaptation is outside this bounded first test.",
        "",
        f"Head experiment: {manifest['elapsed_seconds']:.2f}s, CPU {manifest['machine']}, four threads; peak process RSS {manifest['peak_rss_bytes'] / 1024**3:.3f} GiB. Feature preparation {audit['features']['elapsed_seconds']:.2f}s with graph/token cache hits (not a cold-start timing); original token extraction used the recorded encoder device. Environment: "
        + ", ".join(f"{k} {v}" for k, v in manifest["versions"].items())
        + ". Exact configuration/seeds/source hashes/checkpoints are in manifest.json. Dense embeddings/weights are ignored, not committed.",
        "",
        "## Reproduction and historical preservation",
        "",
        "Follow docs/m9_protocol.md and configs/m9.json: acquire pinned selected members/structures, prepare cached features, run assay_experiments, verify_m9, report_m9. For report-only reproduction: `MPLCONFIGDIR=/tmp/proteinmod-m9-matplotlib python scripts/report_m9.py --input benchmarks/m9 --output runs/m9-report`. All199 M0–M8 benchmark hashes remain unchanged. Repeat/replay evidence is archived separately. Training traces report overfitting; no test-led tuning or cohort revision was performed after the prespecified commit.",
        "",
        "Stacked on draft PR #16, then #14 and #12; all30 dependency CI checks passed, but drafts remain unmerged. Main is unchanged. No M10 work, new multitask training, PLM fine-tuning or uncertainty calibration.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    write_json(
        output / "report_manifest.json",
        {
            "input_prediction_sha256": checksum(root / "predictions.csv"),
            "cases": len(runs),
            "test_predictions": len(predictions),
            "files": {
                n: checksum(output / n)
                for n in (
                    "REPORT.md",
                    "metrics.json",
                    "per_assay.csv",
                    "paired_seed_changes.csv",
                    "comparison.png",
                    "validation_curves.png",
                )
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m9"))
    parser.add_argument("--output", type=Path, default=Path("runs/m9"))
    args = parser.parse_args()
    report(args.input, args.output)
