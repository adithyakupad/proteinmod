"""Small pinned FLIP2 collection with explicit construct crops and protein-held-out roles."""

import csv
import gzip
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from scipy.stats import rankdata

from proteinmod.data import (
    checksum,
    download,
    load_dataset,
    mutations,
    verified_reference,
    verify_file,
    write_json,
)
from proteinmod.embeddings import FrozenESM, cache_location, extract, fingerprint, load_features
from proteinmod.structural_graph import AA, IDENTITY, PROPERTIES, load_graph, variant_inputs
from proteinmod.structure_pipeline import run_structure, write_csv
from proteinmod.structure_sources import acquire_structure


def collection_rows(spec, caps):
    verify_file(spec["dataset"]["path"], spec["dataset"])
    full = verified_reference(spec.get("full_reference", spec["reference"]))
    reference = verified_reference(spec["reference"])
    start, end = spec["crop"]
    if full[start:end] != reference:
        raise ValueError("Crop does not match the verified assay reference")
    source = Path(spec["dataset"]["path"])
    if spec.get("adapter") == "flip2_blank_validation_false":
        with gzip.open(source, "rt") as handle:
            adapted = list(csv.DictReader(handle))
        for row in adapted:
            if row["validation"] == "":
                row["validation"] = "False"
        source = Path("data/cache/m4/adapted") / (checksum(source) + ".csv")
        source.parent.mkdir(parents=True, exist_ok=True)
        write_csv(source, adapted)
    rows, qc = load_dataset(source, full)
    selected, rejected = [], []
    for row in rows:
        seq = row["sequence"]
        if seq[:start] != full[:start] or seq[end:] != full[end:]:
            rejected.append(row["id"])
            continue
        row = dict(
            row,
            source_id=row["id"],
            id=spec["assay_id"] + ":" + row["id"],
            protein_id=spec["protein_id"],
            assay_id=spec["assay_id"],
            sequence=seq[start:end],
        )
        row["mutations"] = mutations(reference, row["sequence"])
        selected.append(row)
    keep = set()
    for part, cap in caps.items():
        candidates = sorted(
            [r for r in selected if r["partition"] == part],
            key=lambda r: fingerprint([r["protein_id"], r["sequence"]]),
        )
        keep.update(r["id"] for r in candidates[:cap])
    selected = [r for r in selected if r["id"] in keep]
    if len({r["sequence"] for r in selected}) != len(selected):
        raise ValueError("Construct cropping created duplicate sequences")
    return selected, {
        "raw_qc": qc,
        "outside_crop_exclusions": rejected,
        "selected_counts": dict(Counter(r["partition"] for r in selected)),
        "sampling": "Smallest SHA256(protein_id, cropped sequence) in each original partition; label blind",
        "dataset_sha256": checksum(spec["dataset"]["path"]),
    }


def sequence_features(row, reference):
    positions = [
        i for i, (a, b) in enumerate(zip(reference, row["sequence"], strict=True)) if a != b
    ]
    old = [AA[reference[i]] for i in positions]
    new = [AA[row["sequence"][i]] for i in positions]
    composition = IDENTITY[[AA[a] for a in row["sequence"]]].mean(axis=0)
    mean = lambda x, width: np.mean(x, axis=0) if len(x) else np.zeros(width)
    relative = [p / max(len(reference) - 1, 1) for p in positions]
    return np.concatenate(
        [
            composition,
            mean(IDENTITY[old], 20),
            mean(IDENTITY[new], 20),
            mean(PROPERTIES[new] - PROPERTIES[old], 4),
            [len(positions), len(positions) / len(reference), np.mean(relative) if relative else 0],
        ]
    )


def train_percentiles(values):
    """Only source-assay training labels define regression targets; no held-out calibration."""
    values = np.asarray(values)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need finite training labels")
    return (rankdata(values, method="average") - 0.5) / len(values)


def protocols(rows, order):
    ids = {r["protein_id"] for r in rows}
    if len(ids) < 3 or set(order) != ids or len(order) != len(ids):
        raise ValueError("G2 requires at least three distinct proteins and an exact fold order")
    folds = [
        (
            "G1",
            "within",
            {
                p: np.array([i for i, r in enumerate(rows) if r["partition"] == p], dtype=int)
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
                ],
                dtype=int,
            )
            for p in ("train", "validation", "test")
        }
        audit_roles(rows, roles, held_out=True)
        folds.append(("G2", test, roles))
    return folds


def audit_roles(rows, roles, held_out=False):
    sets = {p: set(map(int, ix)) for p, ix in roles.items()}
    if any(not x for x in sets.values()) or any(
        sets[a] & sets[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Empty or overlapping variant partitions")
    sequences = {p: {rows[i]["sequence"] for i in ix} for p, ix in sets.items()}
    if any(
        sequences[a] & sequences[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    ):
        raise ValueError("Complete sequence leakage across roles")
    if held_out:
        proteins = {p: {rows[i]["protein_id"] for i in ix} for p, ix in sets.items()}
        if any(
            proteins[a] & proteins[b]
            for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
        ):
            raise ValueError("Supervised protein leakage")


def similarity_audit(specs):
    aligner = PairwiseAligner(
        mode="global", match_score=2, mismatch_score=-1, open_gap_score=-3, extend_gap_score=-0.5
    )
    result = []
    for i, a in enumerate(specs):
        for b in specs[:i]:
            left, right = a["reference"]["sequence"], b["reference"]["sequence"]
            alignment = aligner.align(left, right)[0]
            matches, aligned = 0, 0
            for (start, end), (other_start, other_end) in zip(*alignment.aligned, strict=True):
                aligned += end - start
                matches += sum(
                    x == y
                    for x, y in zip(left[start:end], right[other_start:other_end], strict=True)
                )
            result.append(
                {
                    "left": a["protein_id"],
                    "right": b["protein_id"],
                    "aligned_identity": matches / max(aligned, 1),
                    "left_coverage": aligned / len(left),
                    "right_coverage": aligned / len(right),
                    "identity_over_shorter_length": matches / min(len(left), len(right)),
                    "method": "Global alignment, match 2 mismatch -1 gap -3/-0.5; descriptive, not a family-separation certificate",
                }
            )
    return result


def acquire_collection(config):
    for spec in config["collection"]:
        path = Path(spec["dataset"]["path"])
        if path.exists():
            verify_file(path, spec["dataset"])
        else:
            download(spec["dataset"])
        if "structure_config" in spec:
            structure = spec["structure_config"]["structure"]
            if not Path(structure["path"]).exists():
                acquire_structure(structure)


def collection_output(config):
    output = Path(config["experiment"]["output"])
    for directory in (
        "benchmarks/amylase",
        "benchmarks/amylase_m1",
        "benchmarks/structures",
        "benchmarks/amylase_m3",
    ):
        protected = Path(directory).resolve()
        if (
            output.resolve() == protected
            or protected in output.resolve().parents
            or output.resolve() in protected.parents
        ):
            raise ValueError("M4 output overlaps historical benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_collection(config, local_only=False):
    output = collection_output(config)
    encoder, evidence = None, {}
    for spec in config["collection"]:
        rows, audit = collection_rows(spec, config["experiment"]["cap_per_partition"])
        if "structure_config" in spec:
            run_structure(spec["structure_config"], render=False)
        graph = load_graph(spec["structure_artifacts"], spec["reference"]["sequence"])
        audit["graph"] = graph["audit"]
        settings = {"embedding": config["embedding"]}
        if "existing_embedding_config" in spec:
            with open(spec["existing_embedding_config"], "rb") as f:
                old = tomllib.load(f)
            original, _ = load_dataset(old["dataset"]["path"], verified_reference(old["reference"]))
            directory, identity = cache_location(
                old,
                [r["sequence"] for r in original],
                spec["reference"]["sequence"],
                checksum(old["dataset"]["path"]),
            )
            load_features(directory, identity, len(original), old["embedding"]["dimension"])
        else:
            encoder = encoder or FrozenESM(config["embedding"], local_only=local_only)
            extract(
                settings,
                [r["sequence"] for r in rows],
                spec["reference"]["sequence"],
                checksum(spec["dataset"]["path"]),
                encoder=encoder,
                local_only=local_only,
            )
        audit["selected_ids"] = [r["id"] for r in rows]
        evidence[spec["protein_id"]] = audit
    write_json(output / "preparation.json", evidence)
    return evidence


def model_inputs(config):
    specs = config["collection"]
    if (
        len({s["protein_id"] for s in specs}) != len(specs)
        or len({s["assay_id"] for s in specs}) != len(specs)
        or len({s["reference"]["sequence"] for s in specs}) != len(specs)
    ):
        raise ValueError("Pilot requires distinct protein/assay IDs and reference sequences")
    rows, groups, graph_audits = [], {}, {}
    sequence, structures, embeddings, masks = [], [], [], []
    for spec in config["collection"]:
        part, audit = collection_rows(spec, config["experiment"]["cap_per_partition"])
        ref = spec["reference"]["sequence"]
        graph = load_graph(spec["structure_artifacts"], ref)
        if "existing_embedding_config" in spec:
            with open(spec["existing_embedding_config"], "rb") as f:
                old = tomllib.load(f)
            original, _ = load_dataset(old["dataset"]["path"], verified_reference(old["reference"]))
            directory, identity = cache_location(
                old, [r["sequence"] for r in original], ref, checksum(old["dataset"]["path"])
            )
            em, _ = load_features(directory, identity, len(original), old["embedding"]["dimension"])
            lookup = {r["id"]: i for i, r in enumerate(original)}
            em = em["esm_sites"][[lookup[r["source_id"]] for r in part]]
        else:
            directory, identity = cache_location(
                {"embedding": config["embedding"]},
                [r["sequence"] for r in part],
                ref,
                checksum(spec["dataset"]["path"]),
            )
            em, _ = load_features(directory, identity, len(part), config["embedding"]["dimension"])
            em = em["esm_sites"]
        dynamic, local_indices, excluded = [], [], []
        for j, row in enumerate(part):
            d, s, reasons = variant_inputs(row["mutations"], ref, graph)
            valid = not reasons
            masks.append(valid)
            row["structural_valid"] = valid
            row["structural_exclusions"] = reasons
            if valid:
                dynamic.append(d)
                local_indices.append(len(rows) + j)
            else:
                excluded.append(
                    {"id": row["id"], "partition": row["partition"], "reasons": reasons}
                )
            structures.append(s if valid else np.zeros(270, dtype=np.float32))
            sequence.append(sequence_features(row, ref))
        groups[spec["protein_id"]] = {
            "graph": graph,
            "indices": np.array(local_indices),
            "dynamic": np.stack(dynamic),
        }
        graph_audits[spec["protein_id"]] = {
            "data": audit,
            "graph": graph["audit"],
            "excluded": excluded,
            "embedding_manifest_sha256": checksum(directory / "manifest.json"),
        }
        embeddings.extend(em)
        rows.extend(part)
    features = {
        "mutation_count_ridge": np.array(
            [[len(r["mutations"]), len(r["mutations"]) / len(r["sequence"])] for r in rows]
        ),
        "sequence_ridge": np.array(sequence),
        "esm_sites": np.array(embeddings),
        "structural_ridge": np.array(structures),
    }
    return rows, features, np.array(masks), groups, graph_audits
