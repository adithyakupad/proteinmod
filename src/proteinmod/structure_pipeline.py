"""Auditable offline residue-level extraction, without a structural predictor."""

import csv
import importlib.metadata
import json
import platform
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, load_dataset, mutations, verified_reference, write_json
from proteinmod.structure_features import contact_graph, residue_features, variant_context
from proteinmod.structure_mapping import map_reference, parse_structure
from proteinmod.structure_sources import verify_structure


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def structure_plot(features, edges, variants, output, name, semantics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    layout = fig.add_gridspec(2, 2)
    spatial = fig.add_subplot(layout[:, 0], projection="3d")
    for edge in edges:
        if not edge["backbone"]:
            continue
        points = np.array([features[edge["source"] - 1]["ca"], features[edge["target"] - 1]["ca"]])
        spatial.plot(*points.T, color="slategray", linewidth=0.7, alpha=0.65)
    visible = [row for row in features if row["ca"]]
    xyz = np.array([row["ca"] for row in visible])
    colors = [row["reference_position"] for row in visible]
    spatial.scatter(*xyz.T, c=colors, cmap="viridis", s=7)
    sites = sorted(
        {site["reference_position"] for v in variants for site in v["sites"] if site["usable"]}
    )
    if sites:
        points = np.array([features[position - 1]["ca"] for position in sites])
        spatial.scatter(*points.T, c="crimson", s=32, label="demonstrated mutation sites")
        spatial.legend(fontsize=7)
    spatial.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), 1))
    spatial.set(
        xlabel="x (Å)",
        ylabel="y (Å)",
        zlabel="z (Å)",
        title="Reference Cα geometry; color = reference position",
    )
    positions = [row["reference_position"] for row in features]
    upper = fig.add_subplot(layout[0, 1])
    upper.plot(
        positions, [np.nan if row["rsa"] is None else row["rsa"] for row in features], linewidth=0.8
    )
    upper.set(
        xlabel="Reference position (1-based)",
        ylabel="Relative solvent accessibility",
        title="Shrake–Rupley / Tien maxima (unclipped)",
    )
    lower = fig.add_subplot(layout[1, 1])
    field = (
        "plddt"
        if semantics == "alphafold_plddt"
        else "bfactor_a2"
        if semantics == "experimental_bfactor"
        else "contact_degree"
    )
    lower.plot(
        positions, [np.nan if row[field] is None else row[field] for row in features], linewidth=0.8
    )
    lower.set(
        xlabel="Reference position (1-based)",
        ylabel="pLDDT (0–100)"
        if field == "plddt"
        else "Experimental Cα B-factor (Å²)"
        if field == "bfactor_a2"
        else "Cα contact count",
    )
    if field == "plddt":
        lower.axhline(70, color="crimson", linestyle="--", linewidth=0.7)
        lower.set_ylim(0, 100)
    for ax in (upper, lower):
        for row in features:
            if row["status"] != "mapped":
                ax.axvspan(
                    row["reference_position"] - 0.5,
                    row["reference_position"] + 0.5,
                    color="crimson",
                    alpha=0.2,
                )
    fig.suptitle(f"{name}: reference-structure context, not mutant rearrangements")
    fig.savefig(output / "structure_context.png", dpi=160)
    fig.savefig(output / "structure_context.svg")
    plt.close(fig)


def report(output, config, audit, features, edges, variants, sasa_info):
    usable = [row for row in features if row["status"] == "mapped"]
    rsas = [row["rsa"] for row in usable if row["rsa"] is not None]
    lines = [
        f"# Structural mapping: {config['structure']['name']}",
        "",
        "All variant features are reference-structure context. No mutant coordinates were predicted.",
        "",
        f"Selected author chain: {audit['selected_chain']}; coordinate coverage: {audit['coordinate_coverage']:.6f}.",
        f"Aligned identity: {audit['aligned_identity']:.6f}; exact compatible Cα positions: {audit['exact_compatible_positions']}.",
        f"Missing/unobserved reference positions: {audit['missing_positions']}.",
        f"Reference/construct differences (explicitly reviewed): {audit['differences']}.",
        f"Structure insertions/tags (polymer positions): {audit['construct_insertions']}.",
        f"Chain break diagnostics: {audit['chain_breaks']}.",
        "",
        (
            f"Contacts: {len(edges)} undirected edges; {sum(e['backbone'] for e in edges)} peptide backbone links; "
            f"{sum(e['spatial'] and e['nonlocal'] for e in edges)} nonlocal Cα contacts."
        ),
        f"ASA: {sasa_info}; mean RSA over identity-compatible Cα sites: {float(np.mean(rsas)) if rsas else None}.",
        f"Confidence semantics: **{config['structure']['confidence']}**. Experimental B-factors are not pLDDT.",
        "",
        "| Variant | Mutation set | Usable sites | Unmapped/mismatched sites | Mean site RSA | Mean pair distance (Å) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        summary = variant["summary"]
        lines.append(
            f"| {variant['id']} | {';'.join(variant['mutation_set']) or 'reference'} | "
            f"{summary['n_mapped']} | {summary['n_unmapped']} | {summary['mean_rsa']} | {summary['mean_pair_distance_a']} |"
        )
    lines += [
        "",
        "![Reference structure context](structure_context.png)",
        "",
        "## Limits and interpretation",
        "",
        (
            "Sequence neighbors are nearby FASTA indices. Spatial neighbors can be distant in sequence "
            "but within the Cα distance cutoff after folding. A sequence-adjacent pair is marked backbone "
            "only with adjacent polymer indices, no TER break and a C–N distance ≤1.8 Å. "
            "Local contact density is spatial degree divided by a sphere volume of radius cutoff (residues/Å³). Missing loops "
            "are never bridged as peptide bonds. Contacts are single-chain and omit unmapped insertions; "
            "SASA includes occupied heavy atoms of the selected deposited model, excluding water/H/D, "
            "including ligands/other chains and no crystal symmetry expansion. Missing atoms can inflate SASA."
        ),
        "",
        (
            "Author residue numbers need not start at one, can be negative or include insertion codes, "
            "and differ from polymer and assay reference indices. Tags, missing coordinates and engineered "
            "substitutions require explicit alignment; equally optimal alignments are rejected."
        ),
        "",
        (
            "pLDDT describes local prediction confidence. Low values may reflect disorder/flexibility or "
            "insufficient information, not a measured stability change. High pLDDT does not establish "
            "reliable relative domain placement; PAE is not analyzed here. Experimental B-factors depend "
            "on displacement, occupancy and refinement and are retained in Å² without pLDDT thresholds."
        ),
        "",
        (
            "Mutation sites at mismatched or missing residues produce explicit null features/distances, "
            "not guessed mappings. Symmetric aggregation sorts mutation positions, averages available "
            "site features and tracks missing counts. Reference geometry says nothing about mutant "
            "relaxation, epistasis, activity or stability improvements. No predictive baseline is trained."
        ),
        "",
        f"Coordinate source: {config['structure']['source']}",
        f"License: {config['structure']['license']}",
        f"Reference provenance: {config['reference']['source']}",
        "See audit.json, residues.csv/json, contacts.csv, variants.csv/json and manifest.json for full evidence.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _run_structure(config, render=True):
    started = time.perf_counter()
    output = Path(config["structure_run"]["output"])
    for historical in (Path("benchmarks/amylase"), Path("benchmarks/amylase_m1")):
        if (
            output.resolve() == historical.resolve()
            or historical.resolve() in output.resolve().parents
        ):
            raise ValueError("Structural extraction must not overwrite M0/M1 history")
    path = verify_structure(config["structure"])
    reference = verified_reference(config["reference"])
    if reference is None:
        raise ValueError("A provenance-verified reference is required")
    output.mkdir(parents=True, exist_ok=True)
    try:
        parsed = parse_structure(path, config["mapping"].get("model_index", 0))
        rows, audit = map_reference(parsed, reference, config["mapping"])
    except ValueError as error:
        write_json(output / "audit.json", {"accepted": False, "errors": [str(error)]})
        raise
    audit.update(
        parser_warnings=parsed["parser_warnings"],
        altlocs=parsed["altlocs"],
        skipped=parsed["skipped"],
        model_count=parsed["model_count"],
        model_index=parsed["model_index"],
        format=parsed["format"],
        deposition=parsed["deposition"],
    )
    write_json(output / "audit.json", audit)
    if not audit["accepted"]:
        raise ValueError(f"Structure mapping rejected; inspect {output / 'audit.json'}")
    features, sasa_info = residue_features(
        parsed, rows, config["geometry"], path, config["structure"]["confidence"]
    )
    edges, breaks = contact_graph(rows, features, config["geometry"].get("contact_cutoff_a", 8.0))
    audit["chain_breaks"] = breaks
    audit["spatial_contact_cutoff_a"] = config["geometry"].get("contact_cutoff_a", 8.0)
    audit["sasa"] = sasa_info
    field = (
        "plddt"
        if config["structure"]["confidence"] == "alphafold_plddt"
        else "bfactor_a2"
        if config["structure"]["confidence"] == "experimental_bfactor"
        else "raw_ca_b_iso_or_equiv"
    )
    values = [feature[field] for feature in features if feature[field] is not None]
    audit["confidence_summary"] = {
        "field": field,
        "n_available": len(values),
        "mean": float(np.mean(values)) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "n_low_plddt": sum(feature["low_confidence"] is True for feature in features)
        if field == "plddt"
        else None,
    }
    definitions = list(config.get("variants", []))
    dataset_evidence = None
    if config.get("dataset_variants"):
        with open(config["dataset_variants"]["config"], "rb") as handle:
            original = tomllib.load(handle)
        from proteinmod.data import verify_file

        verify_file(original["dataset"]["path"], original["dataset"])
        if verified_reference(original["reference"]) != reference:
            raise ValueError("Variant dataset reference differs from structural reference")
        dataset, _ = load_dataset(original["dataset"]["path"], reference)
        selected = dataset[: config["dataset_variants"].get("first_n", 5)]
        definitions.extend(
            {
                "id": f"FLIP2-row-{row['id']}",
                "mutations": list(mutations(reference, row["sequence"])),
            }
            for row in selected
        )
        dataset_evidence = {
            "dataset_sha256": checksum(original["dataset"]["path"]),
            "selected_source_row_ids": [row["id"] for row in selected],
            "fitness_labels_used": False,
        }
    if len({variant["id"] for variant in definitions}) != len(definitions):
        raise ValueError("Duplicate variant identifiers")
    variants = [
        variant_context(variant["id"], variant["mutations"], reference, features, edges)
        for variant in definitions
    ]
    audit["pipeline_status"] = "features_ready"
    write_json(output / "audit.json", audit)
    write_json(output / "residues.json", features)
    write_csv(output / "residues.csv", features)
    write_csv(output / "contacts.csv", edges)
    write_json(output / "variants.json", variants)
    write_csv(
        output / "variants.csv",
        [
            {"id": v["id"], "mutation_set": ";".join(v["mutation_set"]), **v["summary"]}
            for v in variants
        ],
    )
    if render:
        structure_plot(
            features,
            edges,
            variants,
            output,
            config["structure"]["name"],
            config["structure"]["confidence"],
        )
    report(output, config, audit, features, edges, variants, sasa_info)
    audit["pipeline_status"] = "complete"
    write_json(output / "audit.json", audit)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    write_json(
        output / "manifest.json",
        {
            "config": config,
            "coordinate_sha256": checksum(path),
            "dataset_variants": dataset_evidence,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "plotter_version": importlib.metadata.version("matplotlib") if render else "not used",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("proteinmod", "biopython", "numpy")
            },
            "git_revision": revision.stdout.strip(),
            "git_dirty": bool(dirty.stdout.strip()),
            "source_sha256": {
                p.name: checksum(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "artifact_sha256": {
                p.name: checksum(p)
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
            "confidence_semantics": config["structure"]["confidence"],
            "context": "reference-structure context only",
        },
    )
    return {
        "accepted": True,
        "chain": audit["selected_chain"],
        "reference_length": len(reference),
        "coordinate_coverage": audit["coordinate_coverage"],
        "aligned_identity": audit["aligned_identity"],
        "exact_compatible_positions": audit["exact_compatible_positions"],
        "differences": audit["differences"],
        "missing_positions": audit["missing_positions"],
        "edges": len(edges),
        "variants": len(variants),
        "output": str(output),
    }


def run_structure(config, render=True):
    output = Path(config["structure_run"]["output"])
    for historical in (Path("benchmarks/amylase"), Path("benchmarks/amylase_m1")):
        if (
            output.resolve() == historical.resolve()
            or historical.resolve() in output.resolve().parents
        ):
            raise ValueError("Structural extraction must not overwrite M0/M1 history")
    try:
        return _run_structure(config, render=render)
    except (ValueError, OSError, ImportError) as error:
        output.mkdir(parents=True, exist_ok=True)
        audit_path = output / "audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"accepted": False}
        audit.update(pipeline_status="failed", pipeline_error=str(error))
        write_json(audit_path, audit)
        raise
