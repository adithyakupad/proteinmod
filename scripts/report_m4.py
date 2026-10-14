"""Verify prediction metrics and summarize assay-local ranks without pooling assay scales."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, write_json
from proteinmod.models import metrics
from proteinmod.structure_pipeline import write_csv


def report(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["artifacts_sha256"].items():
        if checksum(output / name) != digest:
            raise ValueError(f"Changed experiment artifact: {name}")
    runs = json.loads((output / "runs.json").read_text())
    audit = json.loads((output / "audit.json").read_text())
    with (output / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in runs:
        for p, measured in r["test"].items():
            values = [
                x
                for x in predictions
                if x["protocol"] == r["protocol"]
                and x["fold"] == r["fold"]
                and x["cohort"] == r["cohort"]
                and x["model"] == r["model"]
                and int(x["seed"]) == r["seed"]
                and x["protein_id"] == p
            ]
            score = metrics(
                [float(x["measured"]) for x in values], [float(x["prediction"]) for x in values]
            )["spearman_rho"]
            if len(values) != measured["n"] or len({x["id"] for x in values}) != len(values):
                raise ValueError("Duplicate or missing prediction IDs")
            if score is None and measured["spearman_rho"] is None:
                pass
            elif (
                score is None
                or measured["spearman_rho"] is None
                or not np.isclose(score, measured["spearman_rho"], atol=1e-12)
            ):
                raise ValueError("Predictions do not reproduce metrics")
            role = next(
                a
                for a in audit["roles"]
                if a["protocol"] == r["protocol"]
                and a["fold"] == r["fold"]
                and a["cohort"] == r["cohort"]
            )
            expected = {
                i
                for i in role["ids"]["test"]
                if i.startswith(
                    next(
                        s["assay_id"]
                        for s in manifest["config"]["collection"]
                        if s["protein_id"] == p
                    )
                    + ":"
                )
            }
            if {x["id"] for x in values} != expected:
                raise ValueError("Models use unmatched test IDs")
            groups[r["protocol"], r["cohort"], r["model"], p].append(measured)
    per = []
    for (protocol, cohort, model, p), values in sorted(groups.items()):
        scores = [x["spearman_rho"] if x["spearman_rho"] is not None else 0 for x in values]
        per.append(
            {
                "protocol": protocol,
                "cohort": cohort,
                "model": model,
                "protein_id": p,
                "n_test": values[0]["n"],
                "n_seeds": len(values),
                "rho_mean": float(np.mean(scores))
                if any(x["spearman_rho"] is not None for x in values)
                else None,
                "rho_seed_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                "undefined_seed_correlations": sum(x["spearman_rho"] is None for x in values),
            }
        )
    aggregate = []
    for protocol, cohort, model in sorted({(r["protocol"], r["cohort"], r["model"]) for r in per}):
        selected = [
            r for r in per if (r["protocol"], r["cohort"], r["model"]) == (protocol, cohort, model)
        ]
        values = [r["rho_mean"] for r in selected if r["rho_mean"] is not None]
        aggregate.append(
            {
                "protocol": protocol,
                "cohort": cohort,
                "model": model,
                "macro_rho": float(
                    np.mean([r["rho_mean"] if r["rho_mean"] is not None else 0 for r in selected])
                ),
                "proteins_with_defined_rho": len(values),
                "proteins_total": len(selected),
            }
        )
    write_csv(output / "per_protein.csv", per)
    write_csv(output / "aggregate.csv", aggregate)
    lines = [
        "# M4 cross-protein pilot",
        "",
        "Three assays, 600 label-blind sampled variants each. G1 retains original variant splits; G2 assigns entire proteins to train, validation and test.",
        "All selections use validation macro Spearman. Targets are midrank percentiles fitted only within each source-training assay; held-out proteins have no label calibration.",
        "Only rank correlations are aggregated. Raw assay values, especially ordinal NucB categories, are not commensurate.",
        "",
        "## Macro Spearman (equal protein weight)",
        "",
        "| Protocol | Cohort | Model | Macro rho | Defined proteins |",
        "|---|---|---|---:|---:|",
    ]
    for r in aggregate:
        value = "undefined" if r["macro_rho"] is None else f"{r['macro_rho']:.5f}"
        lines.append(
            f"| {r['protocol']} | {r['cohort']} | {r['model']} | {value} | {r['proteins_with_defined_rho']}/{r['proteins_total']} |"
        )
    lines += [
        "",
        "## Per-protein results",
        "",
        "| Protocol | Cohort | Model | Protein | Test n | Rho mean | Seed SD |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in per:
        mean = "undefined" if r["rho_mean"] is None else f"{r['rho_mean']:.5f}"
        sd = "—" if r["rho_seed_sd"] is None else f"{r['rho_seed_sd']:.5f}"
        lines.append(
            f"| {r['protocol']} | {r['cohort']} | {r['model']} | {r['protein_id']} | {r['n_test']} | {mean} | {sd} |"
        )
    lines += ["", "## G2 roles and leakage controls", ""]
    for r in audit["roles"]:
        if r["protocol"] == "G2" and r["cohort"] == "sequence_all":
            lines.append(
                f"- Test {r['fold']}: train {r['proteins']['train']}, validation {r['proteins']['validation']}, test {r['proteins']['test']}."
            )
    lines += [
        "",
        "No supervised rows from validation/test proteins enter model fitting or feature scaling. Validation-protein labels only select hyperparameters/epochs. Unit tests poison test labels and require unchanged fits and predictions.",
        "Three proteins permit one source-training protein per G2 fold; results therefore depend on which source and validation protein are paired with each test protein. This small pilot is not a population-level estimate.",
        "Within-protein one-hot ridge is G1-only: unaligned position-specific columns cannot defensibly transfer across unrelated lengths. G2 uses a shared mutation-composition/property sequence ridge and mutation-count ridge, plus the unchanged frozen ESM site representation.",
        "The neural model shares one scalar head and all weights; there are no assay IDs in model inputs or task-specific heads. Source-assay rank normalization is target handling, not a stability/function multitask objective.",
        "",
        "## Reference and structural coverage",
        "",
    ]
    for p, a in audit["data"].items():
        lines.append(
            f"- {p}: {a['graph']['nodes']} compatible nodes; {len(a['excluded'])} sampled structural exclusions; {len(a['data']['outside_crop_exclusions'])} source rows excluded for changes outside the declared domain crop. PDB differences: {a['graph']['mapping']['differences']}."
        )
    lines += [
        "",
        "GB1 is cropped to the variable 56-residue binding domain, removing its fixed assay fusion partner. NucB removes a fixed signal-peptide prefix; 35 source variants that change this prefix are explicitly excluded. This conditions the benchmark on the defined domain and does not model secretion/fusion effects.",
        "Sequence-only cohorts include structural-invalid variants. Structure-matched cohorts refit every compared model on exactly the same eligible train/validation/test rows. Missing graph residues are excluded, not imputed. Descriptor-missing masks remain explicit. Experimental B-factors are not pLDDT.",
        "Geometry describes reference constructs, not mutant rearrangements. The full M2 alignment/confidence audits are nested in audit.json.",
        "",
        "## Sequence similarity (reference domains)",
        "",
        "| Pair | Aligned identity | Coverage left/right | Identity / shorter length |",
        "|---|---:|---|---:|",
    ]
    for r in audit["sequence_similarity"]:
        lines.append(
            f"| {r['left']} / {r['right']} | {r['aligned_identity']:.4f} | {r['left_coverage']:.4f} / {r['right_coverage']:.4f} | {r['identity_over_shorter_length']:.4f} |"
        )
    lines += [
        "",
        "Global alignment has gap/length biases, especially for short domains. Family annotations are descriptive; no profile-based evolutionary family separation has been verified. **No unseen-family generalization is claimed.** Frozen ESM pretraining overlap is unknown; this is supervised protein-held-out evaluation, not pretraining exclusion.",
        "",
        "## Failure analysis",
        "",
        "Compare G1 with G2 on the same cohort rather than selecting a model after seeing held-out test scores. Negative correlations and undefined constant-prediction correlations are retained. Undefined validation correlations count as zero during selection; reports disclose undefined test correlations and their denominators.",
        "NucB is a tied four-level ordinal activity label, GB1 measures binding in a fusion assay, and Alpha Amylase measures activity. Rank alignment removes scale differences but does not make these biological endpoints identical. Source/assay shift, sparse GB1 mutated sites, mutation-burden differences and structure exclusions can dominate transfer.",
        "Two GNN seeds measure initialization/training variation, not uncertainty over future proteins. Three protein folds are too few for a broad generalization claim.",
        "",
        "## Compute and reproducibility",
        "",
        f"Actual local run: {manifest['elapsed_seconds']:.3f} seconds; peak process RSS {manifest['peak_rss_bytes']} bytes; Python {manifest['python']}; deterministic CPU GNN. Frozen ESM preparation uses the existing pinned checkpoint (MPS where available).",
        f"Historical M0-M3 artifacts verified unchanged: {manifest['historical_files_unchanged']}.",
        "Per-fit runtime, selections, training/validation correlations and epoch traces are in runs.json. Model weights and dense embeddings stay in ignored caches. Dataset hashes, source revisions, structure identities, seeds, exact role IDs and exclusions are in config/manifest/audit files.",
        "See docs/m4_protocol.md for exact commands and citations. Cross-attention, fusion, LoRA, generative design and multitask heads are excluded.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, protocol in zip(axes, ["G1", "G2"], strict=True):
        selected = [
            r for r in aggregate if r["protocol"] == protocol and r["cohort"] == "structure_matched"
        ]
        ax.barh([r["model"] for r in selected], [r["macro_rho"] or 0 for r in selected])
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set(
            title=protocol + " matched structural cohort",
            xlabel="Macro per-protein Spearman rho",
            xlim=(-1, 1),
        )
    fig.suptitle("Three-protein pilot: protein-held-out ≠ unseen-family evaluation")
    fig.savefig(output / "generalization.png", dpi=160)
    fig.savefig(output / "generalization.svg")
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/m4"))
    report(p.parse_args().output)
