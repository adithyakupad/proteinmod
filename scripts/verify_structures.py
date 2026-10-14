"""Opt-in local M2 verification on pinned real structures; never downloads data."""

import argparse
import json
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, write_json
from proteinmod.structure_pipeline import run_structure

CONFIGS = (
    "amylase_m2",
    "amylase_m2_pdb",
    "ubiquitin_m2",
    "ubiquitin_m2_pdb",
    "amylase_m2_alphafold",
)


def historical_hashes():
    return {
        str(path): checksum(path)
        for directory in ("benchmarks/amylase", "benchmarks/amylase_m1")
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }


def compare_formats(first, second):
    a = json.loads((first / "residues.json").read_text())
    b = json.loads((second / "residues.json").read_text())
    assert [
        (r["reference_position"], r["author_number"], r["insertion_code"], r["status"]) for r in a
    ] == [
        (r["reference_position"], r["author_number"], r["insertion_code"], r["status"]) for r in b
    ]
    coordinate_error, rsa_error = 0.0, 0.0
    for left, right in zip(a, b, strict=True):
        if left["ca"] is None:
            assert right["ca"] is None
        else:
            coordinate_error = max(
                coordinate_error, float(np.max(np.abs(np.array(left["ca"]) - right["ca"])))
            )
        if left["rsa"] is not None:
            rsa_error = max(rsa_error, abs(left["rsa"] - right["rsa"]))
    assert coordinate_error <= 1e-6, "PDB/mmCIF C-alpha correspondence differs"
    assert rsa_error <= 1e-6, "PDB/mmCIF solvent accessibility differs"
    import csv

    with (first / "contacts.csv").open() as handle:
        left_edges = list(csv.DictReader(handle))
    with (second / "contacts.csv").open() as handle:
        right_edges = list(csv.DictReader(handle))
    assert left_edges == right_edges, "PDB/mmCIF contact graphs differ"
    return {
        "passed": True,
        "max_ca_coordinate_difference_a": coordinate_error,
        "max_rsa_difference": rsa_error,
        "identical_contact_edges": len(left_edges),
    }


def verify(output):
    started = time.perf_counter()
    before = historical_hashes()
    results = {}
    for name in CONFIGS:
        with open(f"configs/{name}.toml", "rb") as handle:
            config = tomllib.load(handle)
        config["structure_run"]["output"] = str(output / name)
        results[name] = run_structure(config)
        print(json.dumps(results[name]), flush=True)
    comparisons = {
        name: compare_formats(output / name, output / (name + "_pdb"))
        for name in ("amylase_m2", "ubiquitin_m2")
    }
    # Rejection is deliberate evidence: a related structure is not an exact WT.
    with open("configs/amylase_m2.toml", "rb") as handle:
        unreviewed = tomllib.load(handle)
    unreviewed["mapping"]["allowed_mismatches"] = []
    unreviewed["structure_run"]["output"] = str(output / "unreviewed_amylase_rejected")
    try:
        run_structure(unreviewed, render=False)
    except ValueError:
        rejected = json.loads((output / "unreviewed_amylase_rejected/audit.json").read_text())
        assert not rejected["accepted"] and len(rejected["differences"]) == 7
    else:
        raise AssertionError("Unreviewed construct differences were not rejected")
    after = historical_hashes()
    assert before == after, "Historical M0/M1 evidence changed"
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "structures": results,
        "format_comparisons": comparisons,
        "historical_files_unchanged": len(before),
        "historical_sha256": before,
        "unreviewed_construct_rejected": True,
        "script_sha256": checksum(__file__),
        "predictive_experiment": "none; no M0/M1 fitness evaluation or tuning",
    }
    write_json(output / "verification.json", result)
    lines = [
        "# M2 real-structure verification",
        "",
        "All extraction was local and used pinned official coordinates; no predictive model was trained.",
        "",
        "| Structure | Cα coverage | Aligned identity | Exact compatible sites | Edges |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, entry in results.items():
        lines.append(
            f"| {name} | {entry['coordinate_coverage']:.6f} | {entry['aligned_identity']:.6f} | {entry['exact_compatible_positions']} | {entry['edges']} |"
        )
    lines += [
        "",
        f"Format consistency: {comparisons}.",
        f"Historical M0/M1 files unchanged: {len(before)}.",
        "An unreviewed 1UA7-to-assay mapping was rejected for its seven sequence differences.",
        "Reviewed differences are visible in the accepted audit; mutation sites at those residues stay unusable.",
        "The AlphaFold native sequence is a related context, not the assay WT, and has four flagged differences.",
        "DSSP is optional; inspect each audit for actual availability. Null SS annotations are not guessed.",
        "No GNN, cross-attention, fitness score, mutant rearrangement or stability effect is inferred.",
        "",
        "## Reports",
        "",
    ]
    lines.extend(f"- [{name}]({name}/report.md)" for name in CONFIGS)
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/m2_verification"))
    args = parser.parse_args()
    verify(args.output)
