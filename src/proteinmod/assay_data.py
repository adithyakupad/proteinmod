"""M9 crossed endpoint observations; genotype-grouped splits and pinned sources."""

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from proteinmod.assay_metadata import AssayMetadata
from proteinmod.data import download, verify_file
from proteinmod.multiprotein_data import audit_roles, similarity_audit
from proteinmod.range_zip import selected_members
from proteinmod.stability_data import mutant_sequence
from proteinmod.structure_features import parse_mutation_set


@dataclass(frozen=True)
class AssayObservation:
    id: str
    protein_id: str
    assay_id: str
    genotype_id: str
    sequence: str
    mutations: tuple[str, ...]
    target: float
    partition: str
    family: str
    source_row: int
    target_unit: str = "assay_specific_ProteinGym_processed_score"


def acquire(config):
    missing = []
    for assay in config["assays"]:
        spec = assay["dataset"]
        if Path(spec["path"]).exists():
            verify_file(spec["path"], spec)
        else:
            missing.append(spec)
    evidence = {"downloaded_bytes": 0}
    if missing:
        blobs, evidence = selected_members(
            config["archive"]["url"], [s["archive_member"] for s in missing]
        )
        for spec in missing:
            raw = blobs[spec["archive_member"]]
            if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
                raise ValueError("Pinned assay member checksum mismatch")
            path = Path(spec["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    for protein in config["proteins"]:
        spec = protein["structure"]
        if Path(spec["path"]).exists():
            verify_file(spec["path"], spec)
        else:
            download(spec)
    return evidence


def read_assay(spec, reference):
    dataset = spec["dataset"]
    verify_file(dataset["path"], dataset)
    records = {}
    with Path(dataset["path"]).open() as handle:
        for index, raw in enumerate(csv.DictReader(handle), 1):
            tokens = tuple(parse_mutation_set(tuple(raw["mutant"].split(":")), reference))
            sequence = mutant_sequence(reference, tokens)
            if sequence != raw["mutated_sequence"] or not tokens:
                raise ValueError("Source sequence/reference mutation mapping mismatch")
            value = float(raw["DMS_score"])
            if not np.isfinite(value) or sequence in records:
                raise ValueError("Nonfinite or duplicate assay observation")
            records[sequence] = {"mutations": tokens, "target": value, "source_row": index}
    return records


def collection(config):
    if config["schema"] != "m9-crossed-assays-v1":
        raise ValueError("Unknown assay collection schema")
    rows, evidence = [], {}
    for protein in config["proteins"]:
        pid, reference = protein["protein_id"], protein["reference"]["sequence"]
        if hashlib.sha256(reference.encode()).hexdigest() != protein["reference"]["sha256"]:
            raise ValueError("Reference checksum mismatch")
        assays = [a for a in config["assays"] if a["protein_id"] == pid]
        metadata = [AssayMetadata.from_dict(a["metadata"]) for a in assays]
        if len({m.family for m in metadata}) < 2:
            raise ValueError("Crossed endpoint design requires repeated assays per protein")
        source = [read_assay(a, reference) for a in assays]
        common = set.intersection(*(set(a) for a in source))
        ordered = sorted(
            common,
            key=lambda s: hashlib.sha256((config["split_salt"] + pid + s).encode()).hexdigest(),
        )
        cap = config["cap_genotypes_per_protein"]
        if cap < 10 or len(ordered) < cap:
            raise ValueError("Insufficient common genotypes")
        chosen = ordered[:cap]
        fractions = config["split_fractions"]
        if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
            raise ValueError("Three positive partition fractions summing to one required")
        end_train, end_validation = round(cap * fractions[0]), round(cap * sum(fractions[:2]))
        for j, sequence in enumerate(chosen):
            partition = "train" if j < end_train else "validation" if j < end_validation else "test"
            genotype = pid + ":" + ";".join(source[0][sequence]["mutations"])
            for spec, meta, data in zip(assays, metadata, source, strict=True):
                item = data[sequence]
                rows.append(
                    asdict(
                        AssayObservation(
                            meta.assay_id + ":" + genotype,
                            pid,
                            meta.assay_id,
                            genotype,
                            sequence,
                            item["mutations"],
                            item["target"],
                            partition,
                            meta.family,
                            item["source_row"],
                        )
                    )
                )
        evidence[pid] = {
            "assays": [m.assay_id for m in metadata],
            "source_counts": {m.assay_id: len(r) for m, r in zip(metadata, source, strict=True)},
            "common_genotypes": len(common),
            "selected_genotypes": cap,
            "excluded_unpaired_by_assay": {
                m.assay_id: len(r) - len(common) for m, r in zip(metadata, source, strict=True)
            },
            "sampling": "label-blind sequence SHA256; paired endpoints grouped before partition",
        }
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate observation IDs")
    return rows, {
        "proteins": evidence,
        "sequence_similarity": similarity_audit(config["proteins"]),
        "endpoint_families_per_protein": {
            p: sorted({r["family"] for r in rows if r["protein_id"] == p})
            for p in config["protein_order"]
        },
        "assay_id_is_unique_to_protein": True,
        "endpoint_family_is_not_protein_id": True,
        "temperature_ph_effects_identifiable": False,
    }


def folds(rows, order):
    result = [
        (
            "G1",
            "within",
            {
                p: np.array([i for i, r in enumerate(rows) if r["partition"] == p])
                for p in ("train", "validation", "test")
            },
        )
    ]
    for j, test in enumerate(order):
        validation = order[(j + 1) % len(order)]
        roles = {
            p: np.array(
                [
                    i
                    for i, r in enumerate(rows)
                    if (
                        r["protein_id"] == test
                        if p == "test"
                        else r["protein_id"] == validation
                        if p == "validation"
                        else r["protein_id"] not in (test, validation)
                    )
                ]
            )
            for p in ("train", "validation", "test")
        }
        result.append(("G2", test, roles))
    for protocol, _, roles in result:
        audit_roles(rows, roles, held_out=protocol == "G2")
        training = {rows[i]["family"] for i in roles["train"]}
        if any(
            not {rows[i]["family"] for i in roles[p]} <= training for p in ("validation", "test")
        ):
            raise ValueError("Primary G2 must contain known assay families")
    return result
