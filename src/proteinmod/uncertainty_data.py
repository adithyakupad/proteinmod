"""M13 native-label provenance and disjoint fit/selection/calibration/test roles."""

import csv
import hashlib
import io
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, verify_file
from proteinmod.family_data import folds
from proteinmod.multiprotein_data import audit_roles
from proteinmod.stability_data import rows_from_table


def historical():
    return {
        str(p): checksum(p)
        for root in ("benchmarks", "configs")
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
        and not any(
            part.startswith("m")
            and part[1:].split(".")[0].isdigit()
            and int(part[1:].split(".")[0]) >= 13
            for part in p.parts
        )
    }


def native_labels(rows, m12, m6):
    source = m6["sources"]["measurements"]
    verify_file(source["path"], source)
    with zipfile.ZipFile(source["path"]) as z:
        raw = z.read(m6["table_member"])
    if hashlib.sha256(raw).hexdigest() != m6["table_sha256"]:
        raise ValueError("Primary native label checksum mismatch")
    specs = [
        {"protein_id": p["structure"]["id"], "reference": p["reference"]} for p in m12["proteins"]
    ]
    native, _ = rows_from_table(
        list(csv.DictReader(io.StringIO(raw.decode()))), specs, m6["assay"], 10000
    )
    lookup = {(r["protein_id"], r["sequence"]): r for r in native}
    pdb = {p["protein_id"]: p["structure"]["id"] for p in m12["proteins"]}
    values = np.full(len(rows), np.nan)
    evidence = []
    for i, r in enumerate(rows):
        match = lookup.get((pdb[r["protein_id"]], r["sequence"]))
        reason = (
            "multiple_mutation_not_in_primary_single_table"
            if len(r["mutations"]) != 1
            else "M6_prespecified_Cys_exclusion"
            if r["mutations"][0][-1] == "C"
            else "missing_censored_primary_single_label"
        )
        if match is not None:
            if tuple(r["mutations"]) != tuple(match["mutations"]):
                raise ValueError("Native-label mutation mapping mismatch")
            values[i] = match["target"]
        evidence.append(
            {
                "id": r["id"],
                "eligible": match is not None,
                "reason": None if match else reason,
                "native_target_kcal_mol": float(values[i]) if match else None,
                "source_value": match["source_value"] if match else None,
                "source_row": match["source_row"] if match else None,
                "source_column": match["source_column"] if match else None,
                "source_convention": m6["assay"]["source_convention"],
                "target_convention": "positive_destabilizing",
            }
        )
    return values, evidence


def roles_audit(rows, roles, protocol):
    audit_roles(rows, roles, held_out=protocol in ("G2", "G3"))
    if any(
        {rows[i]["sequence"] for i in roles[a]} & {rows[i]["sequence"] for i in roles[b]}
        for a, b in combinations(roles, 2)
    ):
        raise ValueError("Complete sequence calibration leakage")
    if protocol in ("G2", "G3") and any(
        {rows[i]["protein_id"] for i in roles[a]} & {rows[i]["protein_id"] for i in roles[b]}
        for a, b in combinations(roles, 2)
    ):
        raise ValueError("Calibration protein leakage")
    if any(set(roles[a]) & set(roles[b]) for a, b in combinations(roles, 2)):
        raise ValueError("Calibration role leakage")
    families = (
        {s: {rows[i]["family"] for i in ix} for s, ix in roles.items()}
        if "family" in rows[0]
        else {}
    )
    if protocol == "G3" and any(families[a] & families[b] for a, b in combinations(roles, 2)):
        raise ValueError("G3 calibration family leakage")
    if protocol == "G2" and not (families["test"] | families["validation"]) <= families["train"]:
        raise ValueError("G2 known-family condition violated")
    return {
        s: {
            "ids": [rows[i]["id"] for i in ix],
            "proteins": sorted({rows[i]["protein_id"] for i in ix}),
            "families": sorted(families[s]) if families else None,
        }
        for s, ix in roles.items()
    }


def uq_folds(rows, native, m12, config):
    output = []
    families = m12["family_order"]
    order = lambda i: hashlib.sha256(
        (config["calibration_salt"] + rows[i]["id"]).encode()
    ).hexdigest()
    for protocol, fold, original in folds(rows, families):
        if protocol == "G1":
            calibration = []
            for pid in sorted({r["protein_id"] for r in rows}):
                eligible = sorted(
                    (
                        i
                        for i in original["train"]
                        if rows[i]["protein_id"] == pid and np.isfinite(native[i])
                    ),
                    key=order,
                )
                n = config["g1_calibration_native_per_protein"]
                if len(eligible) < n:
                    raise ValueError("Insufficient independent native calibration genotypes")
                calibration.extend(eligible[:n])
        elif protocol == "G2":
            family = next(r["family"] for r in rows if r["protein_id"] == fold)
            j = families.index(family)
            peers = sorted({r["protein_id"] for r in rows if r["family"] == family})
            k = peers.index(fold)
            pids = {
                sorted(
                    {
                        r["protein_id"]
                        for r in rows
                        if r["family"] == families[(j + d) % len(families)]
                    }
                )[k]
                for d in (2, 3)
            }
            calibration = [i for i in original["train"] if rows[i]["protein_id"] in pids]
        else:
            calibration_family = families[(families.index(fold) + 2) % len(families)]
            calibration = [i for i in original["train"] if rows[i]["family"] == calibration_family]
        roles = {
            **original,
            "train": np.array([i for i in original["train"] if i not in set(calibration)], int),
            "calibration": np.array(sorted(calibration), int),
        }
        roles_audit(rows, roles, protocol)
        output.append((protocol, fold, roles))
    return output


def native_control_features(config):
    """Only the three used M6 features; no unrelated supervised transfer checkpoint."""
    from proteinmod.embeddings import site_features
    from proteinmod.multiprotein_data import sequence_features
    from proteinmod.residue_cache import residue_cache
    from proteinmod.stability_data import load_stability
    from proteinmod.structural_graph import load_graph, variant_inputs

    rows, data = load_stability(config)
    values = {"physicochemical_ridge": [], "esm_sites_ridge": [], "structural_ridge": []}
    evidence = {
        "data": data,
        "graphs": {},
        "residue_manifests": {},
        "missingness": [],
        "label_free": True,
        "supervised_transfer_used": False,
    }
    for spec in config["proteins"]:
        pid, ref = spec["protein_id"], spec["reference"]["sequence"]
        part = [r for r in rows if r["protein_id"] == pid]
        graph = load_graph(spec["structure_artifacts"], ref)
        cache = {"reference": spec["reference"], "dataset": config["sources"]["measurements"]}
        tokens, manifest, directory = residue_cache(
            cache, part, config["embedding"], config["residue_cache"]
        )
        wt, _, _ = residue_cache(
            cache,
            [{"id": pid + ":WT", "sequence": ref}],
            config["embedding"],
            config["residue_cache"],
        )
        for j, row in enumerate(part):
            mapped = tuple(t for t in row["mutations"] if int(t[1:-1]) in graph["lookup"])
            _, features, reasons = variant_inputs(mapped, ref, graph)
            if reasons:
                raise ValueError("Native control graph mapping failed")
            fraction = len(mapped) / len(row["mutations"])
            values["physicochemical_ridge"].append(sequence_features(row, ref))
            values["structural_ridge"].append(np.r_[features, fraction, 1.0])
            values["esm_sites_ridge"].append(
                site_features(tokens[j], wt[0], [int(t[1:-1]) - 1 for t in row["mutations"]])
            )
            evidence["missingness"].append({"id": row["id"], "mutation_mapping_fraction": fraction})
        evidence["graphs"][pid] = graph["audit"]
        evidence["residue_manifests"][pid] = {
            "manifest_sha256": checksum(directory / "manifest.json"),
            "manifest": manifest,
        }
    values = {k: np.asarray(v) for k, v in values.items()}
    if any(len(v) != len(rows) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Invalid native control features")
    return rows, values, evidence
