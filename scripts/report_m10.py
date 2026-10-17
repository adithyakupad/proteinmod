"""M10 assay-local metrics, confidence slices, gate diagnostics and negative controls."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from proteinmod.data import checksum, write_json
from proteinmod.structure_pipeline import write_csv


def rho(a, b):
    return (
        float(spearmanr(a, b).statistic)
        if len(a) >= 3 and len(set(a)) > 1 and len(set(b)) > 1
        else None
    )


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    runs = json.loads((root / "runs.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    with (root / "predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    assert len(runs) == 96 and len(predictions) == 29808
    modes = manifest["config"]["modes"]
    summary, assays, paired, slices, gates, gaps = [], [], [], [], [], []
    for protocol in ("G1", "G2"):
        for mode in modes:
            local = [r for r in runs if r["protocol"] == protocol and r["mode"] == mode]
            scores = {
                s: float(np.mean([r["test"]["macro"] for r in local if r["seed"] == s]))
                for s in manifest["config"]["seeds"]
            }
            missing = {
                s: float(
                    np.mean([r["missing_structure_test"]["macro"] for r in local if r["seed"] == s])
                )
                for s in manifest["config"]["seeds"]
            }
            summary.append(
                {
                    "protocol": protocol,
                    "mode": mode,
                    "mean_spearman": float(np.mean(list(scores.values()))),
                    "seed_sd": float(np.std(list(scores.values()))),
                    "seed_scores": scores,
                    "missing_structure_stress_spearman": float(np.mean(list(missing.values()))),
                    "fit_seconds": sum(r["elapsed_seconds"] for r in local),
                }
            )
            for assay in sorted({a for r in local for a in r["test"]["assays"]}):
                values = [r["test"]["assays"][assay] for r in local if assay in r["test"]["assays"]]
                assays.append(
                    {
                        "protocol": protocol,
                        "mode": mode,
                        "assay_id": assay,
                        "protein_id": values[0]["protein_id"],
                        "family": values[0]["family"],
                        "n": values[0]["n"],
                        "mean_spearman": float(np.mean([v["spearman"] or 0 for v in values])),
                        "seed_sd": float(np.std([v["spearman"] or 0 for v in values])),
                    }
                )
        for mode, baseline in [
            ("confidence_input", "unconditioned"),
            ("gate", "weighted_late"),
            ("gate", "gate_zero"),
            ("combined", "unconditioned"),
            ("combined", "gate_zero"),
            ("combined", "combined_shuffled"),
            ("confidence_input", "input_shuffled"),
        ]:
            a = next(s for s in summary if s["protocol"] == protocol and s["mode"] == mode)
            b = next(s for s in summary if s["protocol"] == protocol and s["mode"] == baseline)
            for seed in manifest["config"]["seeds"]:
                paired.append(
                    {
                        "protocol": protocol,
                        "mode": mode,
                        "baseline": baseline,
                        "seed": seed,
                        "delta": a["seed_scores"][seed] - b["seed_scores"][seed],
                    }
                )
    for run in runs:
        local = [
            p
            for p in predictions
            if (p["protocol"], p["fold"], p["mode"], int(p["seed"]))
            == (run["protocol"], run["fold"], run["mode"], run["seed"])
        ]
        assert len(local) == sum(a["n"] for a in run["test"]["assays"].values())
        key = {k: run[k] for k in ("protocol", "fold", "mode", "seed")}
        gaps.append(
            {
                **key,
                "train_macro": run["selection"]["train_macro"],
                "validation_macro": run["selection"]["validation_macro"],
                "test_macro": run["test"]["macro"],
                "epoch": run["selection"]["epoch"],
            }
        )
        for assay in sorted({p["assay_id"] for p in local}):
            subset = [p for p in local if p["assay_id"] == assay]
            for quality in ("low_plddt", "high_plddt", "missing_plddt"):
                selected = [
                    p
                    for p in subset
                    if (
                        p["site_plddt_available"] == "False"
                        if quality == "missing_plddt"
                        else p["site_plddt_available"] == "True"
                        and (
                            float(p["site_plddt"]) < 70
                            if quality == "low_plddt"
                            else float(p["site_plddt"]) >= 70
                        )
                    )
                ]
                slices.append(
                    {
                        **key,
                        "assay_id": assay,
                        "source": "alphafold_plddt",
                        "quality_slice": quality,
                        "n": len(selected),
                        "spearman": rho(
                            [float(p["measured_native_score"]) for p in selected],
                            [float(p["prediction"]) for p in selected],
                        ),
                    }
                )
            values = np.array([float(p["structural_gate"]) for p in subset])
            gates.append(
                {
                    **key,
                    "assay_id": assay,
                    "n": len(subset),
                    "mean_gate": float(values.mean()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "sd": float(values.std()),
                    "gate_plddt_spearman": rho(
                        values.tolist(), [float(p["site_plddt"]) for p in subset]
                    ),
                    "learned_reliability_gate": run["mode"]
                    in ("gate", "combined", "gate_zero", "combined_shuffled"),
                }
            )
    write_json(output / "metrics.json", summary)
    for name, data in [
        ("per_assay.csv", assays),
        ("paired_changes.csv", paired),
        ("quality_slices.csv", slices),
        ("gates.csv", gates),
        ("generalization_gaps.csv", gaps),
    ]:
        write_csv(output / name, data)
    lookup = {(s["protocol"], s["mode"]): s for s in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, protocol in zip(axes, ("G1", "G2"), strict=True):
        ax.barh(
            np.arange(len(modes)),
            [lookup[protocol, m]["mean_spearman"] for m in modes],
            xerr=[lookup[protocol, m]["seed_sd"] for m in modes],
            capsize=2,
            color=["#167d9a" if m == "combined" else "#809ba5" for m in modes],
        )
        ax.set_yticks(np.arange(len(modes)), modes)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set(title=protocol, xlabel="Assay-local Spearman; equal-protein macro")
    axes[0].invert_yaxis()
    fig.suptitle("M10: reference-confidence input and reliability fusion; seed SD")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, protein in zip(axes, ("PTEN", "CYP2C9", "VKOR1"), strict=True):
        local = [
            p
            for p in predictions
            if p["protocol"] == "G2"
            and p["protein_id"] == protein
            and p["mode"] == "combined"
            and p["seed"] == "0"
        ]
        # Two assays share features; display each genotype once.
        local = local[::2]
        ax.scatter(
            [float(p["site_plddt"]) for p in local],
            [float(p["structural_gate"]) for p in local],
            s=10,
            alpha=0.6,
        )
        ax.set(
            title=protein,
            xlabel="Reference pLDDT",
            ylabel="Structural-query mixing weight",
            ylim=(0, 1),
        )
    fig.suptitle("Learned weights are not calibrated predictive uncertainty or causality")
    fig.tight_layout()
    fig.savefig(output / "gate_behavior.png", dpi=160)
    plt.close(fig)
    table = ["| Model | G1 rho | G2 rho | Missing-structure G2 stress |", "|---|---:|---:|---:|"]
    for mode in modes:
        table.append(
            f"| {mode} | {lookup['G1', mode]['mean_spearman']:.4f} | {lookup['G2', mode]['mean_spearman']:.4f} | {lookup['G2', mode]['missing_structure_stress_spearman']:.4f} |"
        )
    changes = []
    for mode, base in [
        ("confidence_input", "unconditioned"),
        ("gate", "gate_zero"),
        ("combined", "gate_zero"),
        ("combined", "combined_shuffled"),
    ]:
        delta = [
            p["delta"]
            for p in paired
            if p["protocol"] == "G2" and p["mode"] == mode and p["baseline"] == base
        ]
        changes.append(
            f"- {mode} minus {base}: {np.mean(delta):+.4f}, seed SD {np.std(delta):.4f}, range {min(delta):+.4f} to {max(delta):+.4f}."
        )
    by_protein = [
        "| Held-out protein | Assay | Unconditioned | Input | Gate | Combined | Shuffled combined |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for protein in ("PTEN", "CYP2C9", "VKOR1"):
        for family in ("abundance", "activity"):
            values = [
                next(
                    a["mean_spearman"]
                    for a in assays
                    if a["protocol"] == "G2"
                    and a["protein_id"] == protein
                    and a["family"] == family
                    and a["mode"] == m
                )
                for m in (
                    "unconditioned",
                    "confidence_input",
                    "gate",
                    "combined",
                    "combined_shuffled",
                )
            ]
            by_protein.append(
                f"| {protein} | {family} | " + " | ".join(f"{v:.4f}" for v in values) + " |"
            )
    quality_table = [
        "| Held-out protein | Site pLDDT | Variants per assay | Unconditioned rho | Combined rho |",
        "|---|---|---:|---:|---:|",
    ]
    gate_table = [
        "| Held-out protein | Combined mean structural-query weight | Weight/pLDDT Spearman |",
        "|---|---:|---:|",
    ]
    for protein in ("PTEN", "CYP2C9", "VKOR1"):
        for quality in ("low_plddt", "high_plddt"):
            values = []
            for mode in ("unconditioned", "combined"):
                local = [
                    s
                    for s in slices
                    if s["protocol"] == "G2"
                    and s["fold"] == protein
                    and s["quality_slice"] == quality
                    and s["mode"] == mode
                ]
                valid = [s["spearman"] for s in local if s["spearman"] is not None]
                values.append(f"{np.mean(valid):.4f}" if valid else "unavailable")
            quality_table.append(
                f"| {protein} | {quality} | {local[0]['n']} | {' | '.join(values)} |"
            )
        local = [
            g
            for g in gates
            if g["protocol"] == "G2" and g["fold"] == protein and g["mode"] == "combined"
        ]
        gate_table.append(
            f"| {protein} | {np.mean([g['mean_gate'] for g in local]):.4f} | {np.mean([g['gate_plddt_spearman'] for g in local]):.4f} |"
        )
    lines = [
        "# M10: confidence-aware structure input and reliability fusion",
        "",
        "## Actual matched experiments",
        "",
        "The exact M9 six-assay cohort and G1/G2 role IDs are retained:1,080 observations from540 paired-endpoint genotypes across PTEN, CYP2C9 and VKOR1. 96 selected model/seed/fold fits,29,808 held-out predictions plus the same29,808 missing-structure stress predictions. All native correlations are assay-local before equal-endpoint/protein aggregation; three-seed means and variation are not population confidence intervals. All candidates and failures remain visible.",
        "",
        *table,
        "",
        "G2 paired seed changes:",
        "",
        *changes,
        "",
        "The result is negative: confidence input is worse than the matched neutral-input baseline in all three G2 seeds. The learned gate does not improve on its zero-confidence control, and combined fusion's small advantage over shuffled confidence changes sign across seeds. Do not infer incremental biological reliability from these comparisons. This tests a frozen pretrained encoder plus small heads, not every possible confidence-aware architecture.",
        "",
        "## Protein and quality failures",
        "",
        *by_protein,
        "",
        *quality_table,
        "",
        "Slice rho is averaged over the two native assays and three seeds, without pooling raw targets. PTEN/VKOR1 low-pLDDT gains coexist with high-pLDDT regressions; CYP2C9 has only four low-pLDDT genotypes, so its slice is extremely noisy. Local predicted geometry quality is not a functional-effect error estimate. Frozen encoder quality weights learned on different proteins may not transfer; global source and full-coverage constants cannot teach cross-source or missingness effects here.",
        "",
        "quality_slices.csv contains per-assay low(<70), high(>=70) and missing site-pLDDT strata for every fold/seed/model, including empty or too-small slices as null. All actual source strata are AlphaFold and every genotype is a single mutant; experimental-vs-predicted source contrast and mutation-burden performance differences are unidentifiable here. Do not average unlike-assay raw measurements or silently drop small slices. generalization_gaps.csv retains train/validation/test scores and selected epochs.",
        "",
        "Missing structure forces mixing weight exactly0 and recomputes sequence-query features with structural attention masked. Mixture uses its trained sequence expert; concatenation uses its same trained head on missing geometry. The stress is inference-only: no absent-structure examples were used in training, so these scores do not establish calibrated fallback quality.",
        "",
        "## Scientific semantics and architecture",
        "",
        "AlphaFold pLDDT describes predicted local reference geometry, not mutation-effect prediction uncertainty. Experimental B-factors have units A², while experimental resolution has units A; neither is converted into pLDDT. Values/observed fractions and source/mapping provenance remain distinct. No M8 stability extrapolation or thermodynamic scalar is consumed. Reference geometry does not establish mutant rearrangements.",
        "",
        "The existing frozen GNN already contains B-factor/pLDDT input columns. The neutral profile masks those values and their availability consistently; real input enables typed, physically normalized confidence BEFORE GNN message passing, using existing pretrained weights. The real-input encoder exactly reproduces M9 frozen features. The M10 head uses raw frozen LayerNorm representations and fixed physical descriptor scales for every comparison, not M9 within-protein StandardScaler. Historical M9 head scores therefore are not a matched architecture baseline.",
        "",
        "Gate input is both cross-attended pooled branches plus18 typed confidence descriptors (site/neighborhood quality, source, mapping, coverage, missingness and optional resolution). Gate mixes sequence-query and structure-query expert predictions. Both branches already include bidirectional cross-attention: mixing weight is NOT total structural influence. Plain weighted late fusion learns one scalar on these same experts. Registered capacity is identical across modes, but active capacity differs between concatenation, scalar mix and gate; gate_zero and shuffled comparisons isolate the incremental confidence channel.",
        "",
        "Shuffling moves whole confidence value/mask records within each reference protein for each fixed seed; geometry/sequence/labels stay unchanged and local summaries are recomputed. This preserves protein-level confidence marginals and can expose protein-proxy or nuisance effects; it does not constitute a broad randomization test. Physical structure-source type is explicit; no protein/assay identity or target statistics enter the reliability descriptors. Training-assay midranks and validation-only epoch selection remain unchanged.",
        "",
        "gates.csv reports per-assay weight distributions and correlations with real pLDDT. Constant0.5 values in concatenation modes are inert diagnostics, not learned reliability; weighted_late has a learned scalar; gate_zero receives no explicit quality. No monotonic confidence-gate relationship was imposed. Weights must not be presented as predictive probabilities, error bars or mechanistic explanations.",
        "",
        *gate_table,
        "",
        "## Reproducibility, costs and limits",
        "",
        f"Head run: {manifest['elapsed_seconds']:.2f}s on {manifest['machine']} CPU, four threads; peak RSS {manifest['peak_rss_bytes'] / 1024**3:.3f} GiB. Cache-hit feature preparation: {audit['features']['elapsed_seconds']:.2f}s; original ESM tokens were extracted with MPS in M9. Frozen backbone: ESM-2 t12 35M revision6fbf070e..., M7 function-only GNN/bidirectional encoder trained on old amylase/validated on NucB. M9 similarity/provenance audits remain valid; PLM pretraining exposure is unknown. Registered parameters per head: {runs[0]['selection']['parameters']}. Environment: "
        + ", ".join(f"{p} {v}" for p, v in manifest["versions"].items())
        + ".",
        "",
        "Protocol/config/tests were committed before head fitting; full source/config/checkpoint/feature fingerprints and role IDs are archived. Reproduction follows docs/m10_protocol.md: prepare frozen profiles, run reliability_experiments, verify_m10 and report_m10. Report-only reproduction needs only committed evidence: `MPLCONFIGDIR=/tmp/proteinmod-m10-matplotlib python scripts/report_m10.py --input benchmarks/m10 --output runs/m10-report`. Dense token arrays and weights remain ignored.",
        "",
        "The bounded all-AlphaFold three-protein, single-mutant sample limits identifiability and statistical power. Full structures make missingness/mapping/resolution effects engineering-tested rather than scientifically validated. No cross-source, unseen-family, calibrated uncertainty or mutant-geometry claims. No M11, new epistasis, PLM fine-tuning, uncertainty calibration or multitask training.",
        "",
        "All219 M0–M9 benchmark artifacts and old configurations remain unchanged. M10 is stacked on verified draft #18 → #16 → #14 → #12; all40 dependency checks passed, main unchanged.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    write_json(
        output / "report_manifest.json",
        {
            "prediction_sha256": checksum(root / "predictions.csv"),
            "files": {
                name: checksum(output / name)
                for name in (
                    "metrics.json",
                    "per_assay.csv",
                    "paired_changes.csv",
                    "quality_slices.csv",
                    "gates.csv",
                    "generalization_gaps.csv",
                    "REPORT.md",
                    "comparison.png",
                    "gate_behavior.png",
                )
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m10"))
    parser.add_argument("--output", type=Path, default=Path("runs/m10"))
    args = parser.parse_args()
    report(args.input, args.output)
