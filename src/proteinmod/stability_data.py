"""Experimental stability contracts: explicit thermodynamics, coordinates and leakage groups."""

import csv
import hashlib
import io
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from proteinmod.data import AMINO_ACIDS, checksum, download, mutations, verified_reference
from proteinmod.embeddings import fingerprint
from proteinmod.multiprotein_data import audit_roles, protocols, similarity_audit
from proteinmod.structure_features import parse_mutation_set


def ddg_target(value, unit, convention, task_type="ddg"):
    """Canonical folding ΔΔG = ΔG_fold(mut) − ΔG_fold(WT), positive destabilization."""
    if task_type != "ddg":
        raise ValueError("ΔTm, absolute ΔG and proxy labels are separate tasks, not ΔΔG")
    if unit not in ("kcal/mol", "kJ/mol") or convention not in (
        "positive_destabilizing",
        "positive_stabilizing",
    ):
        raise ValueError("Explicit supported ΔΔG units and sign convention required")
    if value in (None, "", "-", "NA", "nan"):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None  # censored/interval strings are not point measurements
    if not math.isfinite(result):
        return None
    return (
        result
        * (1 if convention == "positive_destabilizing" else -1)
        / (4.184 if unit == "kJ/mol" else 1)
    )


def mutant_sequence(reference, tokens):
    normalized = parse_mutation_set(tokens, reference)
    mutant_residues = list(reference)
    for token in normalized:
        mutant_residues[int(token[1:-1]) - 1] = token[-1]
    sequence = "".join(mutant_residues)
    if set(mutations(reference, sequence)) != set(normalized):
        raise ValueError("Mutation/reference correspondence failed")
    return sequence


def pair_key(reference, sequence):
    return fingerprint(sorted((reference, sequence)))


def merge_measurements(rows):
    """Aggregate only identical directed sequence pairs under identical assay conditions."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            row["reference"], row["sequence"], row["assay_id"], fingerprint(row["conditions"])
        ].append(row)
    result = []
    for group in grouped.values():
        unique = {}
        for measurement in group:
            previous = unique.get(measurement["id"])
            if previous is not None and previous != measurement:
                raise ValueError("Conflicting records for the same experimental measurement ID")
            unique[measurement["id"]] = measurement
        duplicates = len(group) - len(unique)
        group = list(unique.values())
        row = dict(group[0])
        row["duplicate_records_removed"] = duplicates
        values = [
            measurement_row["target"] for measurement_row in group if measurement_row["target_mask"]
        ]
        row["target"] = float(np.mean(values)) if values else None
        row["target_mask"] = bool(values)
        row["source_ids"] = [measurement_row["id"] for measurement_row in group]
        row["measurement_count"] = len(values)
        row["measurement_sd_kcal_mol"] = float(np.std(values)) if len(values) > 1 else None
        result.append(row)
    return result


def rows_from_table(table, specs, assay, cap):
    references = {s["protein_id"]: verified_reference(s["reference"]) for s in specs}
    rows, rejected, seen = [], [], set()
    for index, record in enumerate(table):
        protein = record["pdb_name"].removesuffix(".pdb")
        if protein not in references:
            continue
        position = int(record["pos"])
        ref = references[protein]
        if not 1 <= position <= len(ref) or record["wt_aa"] != ref[position - 1]:
            raise ValueError(f"Source WT/coordinate mismatch: {protein}:{position}")
        if (protein, position) in seen:
            raise ValueError(
                "Duplicate source position row; resolve repeated measurements explicitly"
            )
        seen.add((protein, position))
        wt = ddg_target(
            record["ddg_" + ref[position - 1]],
            assay["unit"],
            assay["source_convention"],
            assay["task_type"],
        )
        if wt is None or abs(wt) > 1e-8:
            raise ValueError("Source WT ΔΔG anchor must be zero")
        for aa in AMINO_ACIDS:
            if aa == ref[position - 1]:
                continue
            token = f"{ref[position - 1]}{position}{aa}"
            identifier = f"{assay['id']}:{protein}:{token}"
            value = ddg_target(
                record["ddg_" + aa], assay["unit"], assay["source_convention"], assay["task_type"]
            )
            reason = (
                "cysteine_substitution_prespecified"
                if aa == "C"
                else "missing_or_censored_label"
                if value is None
                else None
            )
            if reason:
                rejected.append({"id": identifier, "reason": reason, "source_row": index + 2})
                continue
            sequence = mutant_sequence(ref, (token,))
            rows.append(
                {
                    "id": identifier,
                    "protein_id": protein,
                    "assay_id": assay["id"],
                    "reference": ref,
                    "sequence": sequence,
                    "mutations": (token,),
                    "source_value": float(record["ddg_" + aa]),
                    "source_unit": assay["unit"],
                    "source_convention": assay["source_convention"],
                    "target": value,
                    "target_unit": "kcal/mol",
                    "target_convention": "positive_destabilizing",
                    "task_type": "ddg",
                    "target_mask": True,
                    "conditions": assay["conditions"],
                    "conditions_mask": {k: v is not None for k, v in assay["conditions"].items()},
                    "source_row": index + 2,
                    "source_column": "ddg_" + aa,
                    "replicate_quality_available": False,
                    "label_ci_available": False,
                    "pair_key": pair_key(ref, sequence),
                }
            )
    selected = []
    counts = Counter(r["protein_id"] for r in rows)
    for spec in specs:
        part = sorted(
            (r for r in rows if r["protein_id"] == spec["protein_id"]),
            key=lambda r: fingerprint([r["protein_id"], r["sequence"]]),
        )[:cap]
        if len(part) < 10:
            raise ValueError("Too few stability measurements for a protein")
        for i, r in enumerate(part):
            r["partition"] = (
                "train"
                if i < int(0.7 * len(part))
                else "validation"
                if i < int(0.85 * len(part))
                else "test"
            )
        selected.extend(part)
    return selected, {
        "available_by_protein": dict(counts),
        "excluded": rejected,
        "selected_by_protein": dict(Counter(r["protein_id"] for r in selected)),
        "sampling": "Smallest sequence-identity hashes, label-blind cap; 70/15/15 within-protein diagnostic",
        "source_columns_used": ["pdb_name", "pos", "wt_aa", "ddg_<AA>"],
        "source_analysis_features_never_used": [
            "meanzero_dg_*",
            "pc*",
            "5pc_error",
            "K50*",
            "author burial/contact/SASA",
            "GEMME",
            "WT mean stability",
        ],
    }


def load_stability(config):
    for source in config["sources"].values():
        from proteinmod.data import verify_file

        verify_file(source["path"], source)
    with zipfile.ZipFile(config["sources"]["measurements"]["path"]) as z:
        raw = z.read(config["table_member"])
    if hashlib.sha256(raw).hexdigest() != config["table_sha256"]:
        raise ValueError("Stability table hash mismatch")
    return rows_from_table(
        csv.DictReader(io.StringIO(raw.decode())),
        config["proteins"],
        config["assay"],
        config["cap_per_protein"],
    )


def acquire(config):
    for source in config["sources"].values():
        if Path(source["path"]).exists():
            from proteinmod.data import verify_file

            verify_file(source["path"], source)
        else:
            download(source)


def audit_stability_roles(rows, roles, held_out, specs):
    audit_roles(rows, roles, held_out)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {rows[i]["pair_key"] for i in roles[left]} & {rows[i]["pair_key"] for i in roles[right]}:
            raise ValueError("Forward/reverse mutation pair leakage")
    if held_out:
        assignment = {rows[i]["protein_id"]: role for role, ix in roles.items() for i in ix}
        for a in specs:
            for b in specs:
                if (
                    a["protein_id"] != b["protein_id"]
                    and (
                        a["structure_member"] == b["structure_member"]
                        or (
                            a.get("structure_sha256") is not None
                            and a.get("structure_sha256") == b.get("structure_sha256")
                        )
                    )
                    and assignment[a["protein_id"]] != assignment[b["protein_id"]]
                ):
                    raise ValueError("Shared structure construct across protein roles")
        for pair in similarity_audit(specs):
            if (
                pair["aligned_identity"] >= 0.3
                and min(pair["left_coverage"], pair["right_coverage"]) >= 0.8
                and assignment[pair["left"]] != assignment[pair["right"]]
            ):
                raise ValueError("Homolog/near-duplicate protein leakage")


def stability_folds(rows, config):
    result = []
    for protocol, fold, roles in protocols(rows, [s["protein_id"] for s in config["proteins"]]):
        audit_stability_roles(rows, roles, protocol == "G2", config["proteins"])
        result.append(("S1" if protocol == "G1" else "S2", fold, roles))
    return result


def history():
    return {
        str(p): checksum(p)
        for p in Path("benchmarks").rglob("*")
        if p.is_file() and "m6" not in p.parts
    }


def output_directory(value):
    output = Path(value).resolve()
    # Historical trees include M5; protect whole directories, including new child paths.
    for name in ("amylase", "amylase_m1", "structures", "amylase_m3", "m4", "m5"):
        protected = (Path("benchmarks") / name).resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("M6 output overlaps historical artifacts")
    output.mkdir(parents=True, exist_ok=True)
    return output
