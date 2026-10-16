"""M12 externally classified domains, label-blind cohorts and nested family roles."""

import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner

from proteinmod.data import verified_reference, verify_file
from proteinmod.multiprotein_data import audit_roles
from proteinmod.stability_data import mutant_sequence
from proteinmod.structure_features import parse_mutation_set


def collection(config):
    if config["schema"] != "m12-family-held-out-v1":
        raise ValueError("Unknown family schema")
    rows, evidence = [], {}
    counts = config["partition_counts"]
    if len(counts) != 3 or min(counts) < 2 or sum(counts) != config["cap_per_protein"]:
        raise ValueError("Three nonempty partition counts must sum to cap")
    for spec in config["proteins"]:
        pid, reference = spec["protein_id"], verified_reference(spec["reference"])
        for name in ("dataset", "classification"):
            verify_file(spec[name]["path"], spec[name])
        external = json.loads(Path(spec["classification"]["path"]).read_text())
        domains = [a for a in external["rcsb_polymer_entity_feature"] if a["type"] == "Pfam"]
        if spec["family"] not in {a["feature_id"] for a in domains}:
            raise ValueError("Family assignment absent from external domain annotations")
        records, missing = {}, 0
        with Path(spec["dataset"]["path"]).open() as handle:
            for number, raw in enumerate(csv.DictReader(handle), 1):
                tokens = tuple(parse_mutation_set(tuple(raw["mutant"].split(":")), reference))
                sequence = mutant_sequence(reference, tokens)
                if sequence != raw["mutated_sequence"] or not tokens:
                    raise ValueError("Reference/mutant mismatch or WT row")
                if raw["DMS_score"].strip().lower() in ("", "nan", "na"):
                    missing += 1
                    continue
                target = float(raw["DMS_score"])
                if not np.isfinite(target) or sequence in records:
                    raise ValueError("Nonfinite or duplicate mutation observation")
                records[sequence] = (tokens, target, number)
        ordered = sorted(
            records,
            key=lambda s: hashlib.sha256((config["split_salt"] + pid + s).encode()).hexdigest(),
        )
        if len(ordered) < sum(counts):
            raise ValueError("Insufficient measured variants")
        for j, sequence in enumerate(ordered[: sum(counts)]):
            tokens, target, number = records[sequence]
            partition = (
                "train" if j < counts[0] else "validation" if j < sum(counts[:2]) else "test"
            )
            rows.append(
                {
                    "id": spec["assay_id"] + ":" + ":".join(tokens),
                    "protein_id": pid,
                    "assay_id": spec["assay_id"],
                    "family": spec["family"],
                    "sequence": sequence,
                    "mutations": tokens,
                    "target": target,
                    "source_row": number,
                    "partition": partition,
                    "target_unit": "assay-specific ProteinGym DMS_score; higher is better",
                }
            )
        evidence[pid] = {
            "available": len(records),
            "missing_labels": missing,
            "selected": sum(counts),
            "mutation_counts": {
                str(k): sum(len(r["mutations"]) == k for r in rows if r["protein_id"] == pid)
                for k in (1, 2, 3, 4)
            },
        }
    return rows, evidence


def folds(rows, families):
    by_family = {f: sorted({r["protein_id"] for r in rows if r["family"] == f}) for f in families}
    if (
        len(families) < 4
        or any(len(v) < 2 for v in by_family.values())
        or set(by_family) != {r["family"] for r in rows}
    ):
        raise ValueError("At least four classified families with two proteins each required")
    output = [
        (
            "G1",
            "within",
            {
                role: np.array([i for i, r in enumerate(rows) if r["partition"] == role], dtype=int)
                for role in ("train", "validation", "test")
            },
        )
    ]
    for j, family in enumerate(families):
        for k, pid in enumerate(by_family[family]):
            validation = by_family[families[(j + 1) % len(families)]][
                k % len(by_family[families[(j + 1) % len(families)]])
            ]
            role = lambda r, pid=pid, validation=validation: (
                "test"
                if r["protein_id"] == pid
                else "validation"
                if r["protein_id"] == validation
                else "train"
            )
            output.append(
                (
                    "G2",
                    pid,
                    {
                        s: np.array([i for i, r in enumerate(rows) if role(r) == s], dtype=int)
                        for s in ("train", "validation", "test")
                    },
                )
            )
        validation_family = families[(j + 1) % len(families)]
        role = lambda r, family=family, validation_family=validation_family: (
            "test"
            if r["family"] == family
            else "validation"
            if r["family"] == validation_family
            else "train"
        )
        output.append(
            (
                "G3",
                family,
                {
                    s: np.array([i for i, r in enumerate(rows) if role(r) == s], dtype=int)
                    for s in ("train", "validation", "test")
                },
            )
        )
    for protocol, _, roles in output:
        audit(rows, roles, protocol)
    return output


def audit(rows, roles, protocol):
    audit_roles(rows, roles, held_out=protocol != "G1")
    sets = {s: {rows[i]["family"] for i in ix} for s, ix in roles.items()}
    if protocol == "G3" and any(sets[a] & sets[b] for a, b in combinations(sets, 2)):
        raise ValueError("G3 family leakage")
    if protocol == "G2" and not (sets["test"] | sets["validation"]) <= sets["train"]:
        raise ValueError("G2 requires known training families")
    return {
        s: {
            "ids": [rows[i]["id"] for i in ix],
            "proteins": sorted({rows[i]["protein_id"] for i in ix}),
            "families": sorted(sets[s]),
        }
        for s, ix in roles.items()
    }


def alignment(a, b, mode="global"):
    aligner = PairwiseAligner(
        mode=mode, match_score=2, mismatch_score=-1, open_gap_score=-3, extend_gap_score=-0.5
    )
    candidates = iter(aligner.align(a, b))
    match = next(candidates)
    blocks = list(zip(match.aligned[0], match.aligned[1], strict=True))
    pairs = [(i, j) for x, y in blocks for i, j in zip(range(*x), range(*y), strict=True)]
    n = len(pairs)
    return {
        "identity": sum(a[i] == b[j] for i, j in pairs) / n if n else 0,
        "aligned": n,
        "coverage_a": n / len(a),
        "coverage_b": n / len(b),
        "blocks": [[[int(v) for v in x], [int(v) for v in y]] for x, y in blocks],
        "ambiguous_optimum": next(candidates, None) is not None,
    }


def similarity(config):
    specs = config["proteins"]
    settings = config["similarity"]
    pairs = []
    for a, b in combinations(specs, 2):
        x, y = a["reference"]["sequence"], b["reference"]["sequence"]
        global_match, local_match = alignment(x, y), alignment(x, y, "local")
        partial = (
            local_match["aligned"] >= settings["partial_min_aligned"]
            and local_match["identity"] >= settings["partial_min_identity"]
            and max(local_match["coverage_a"], local_match["coverage_b"])
            >= settings["partial_min_shorter_coverage"]
        )
        pairs.append(
            {
                "a": a["protein_id"],
                "b": b["protein_id"],
                "same_family": a["family"] == b["family"],
                "global_alignment": global_match,
                "local_alignment": local_match,
                "partial_homology_flag": partial,
            }
        )
    clusters = {}
    for threshold in settings["cluster_identity_thresholds"]:
        groups = [{s["protein_id"]} for s in specs]
        for p in pairs:
            g = p["global_alignment"]
            if (
                g["identity"] >= threshold
                and min(g["coverage_a"], g["coverage_b"]) >= settings["min_both_coverage"]
            ):
                left = next(v for v in groups if p["a"] in v)
                right = next(v for v in groups if p["b"] in v)
                if left is not right:
                    left.update(right)
                    groups.remove(right)
        clusters[str(threshold)] = sorted([sorted(v) for v in groups])
    cross = [
        p
        for p in pairs
        if not p["same_family"]
        and (
            p["partial_homology_flag"]
            or any(
                p["a"] in group and p["b"] in group
                for groups in clusters.values()
                for group in groups
            )
        )
    ]
    if cross:
        raise ValueError(
            "Unresolved cross-family sequence/domain similarity: "
            + str([(p["a"], p["b"]) for p in cross])
        )
    return {
        "pairs": pairs,
        "clusters": clusters,
        "unresolved_cross_family": [],
        "definition": "Global aligned-column identity with both coverage; local blocks separately screen partial/domain homologs. External Pfam assignments define families, never this threshold alone.",
    }
