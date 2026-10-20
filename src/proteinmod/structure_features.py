"""Reference-geometry features, with confidence semantics kept separate."""

import itertools
import re
import shutil

import numpy as np
from Bio.PDB import DSSP
from Bio.PDB.SASA import ShrakeRupley

from proteinmod.data import AMINO_ACIDS

# Tien et al. 2013 theoretical Gly-X-Gly maxima, Angstrom^2 (Wilke table).
# https://doi.org/10.1371/journal.pone.0080635
MAX_ASA = dict(
    zip(
        AMINO_ACIDS,
        [
            129,
            167,
            193,
            223,
            240,
            104,
            224,
            197,
            236,
            201,
            224,
            195,
            159,
            225,
            274,
            155,
            172,
            174,
            285,
            263,
        ],
        strict=True,
    )
)


def confidence_features(value, semantics):
    if value is None or not np.isfinite(value):
        return {"bfactor_a2": None, "plddt": None, "low_confidence": None}
    if semantics == "alphafold_plddt":
        if not 0 <= value <= 100:
            raise ValueError("AlphaFold pLDDT outside [0,100]; check confidence semantics")
        return {"bfactor_a2": None, "plddt": float(value), "low_confidence": bool(value < 70)}
    if semantics == "experimental_bfactor":
        if value < 0:
            raise ValueError("Negative experimental B-factor")
        return {"bfactor_a2": float(value), "plddt": None, "low_confidence": None}
    if semantics != "unknown":
        raise ValueError("Unknown confidence semantics")
    return {"bfactor_a2": None, "plddt": None, "low_confidence": None}


def secondary_structure(parsed, path, mode):
    if mode not in ("off", "auto", "required"):
        raise ValueError("DSSP mode must be off, auto or required")
    if mode == "off":
        return {}, {"status": "disabled"}
    executable = shutil.which("mkdssp") or shutil.which("dssp")
    if executable is None:
        if mode == "required":
            raise ValueError("DSSP requested but mkdssp/dssp is not installed")
        return {}, {"status": "unavailable", "reason": "mkdssp/dssp not installed; no SS invented"}
    if parsed["model_index"] != 0:
        if mode == "required":
            raise ValueError("DSSP only supports the first deposited model")
        return {}, {"status": "unavailable", "reason": "DSSP first-model restriction"}
    try:
        result = DSSP(parsed["model"], str(path), dssp=executable, acc_array="Wilke")
        values = {
            (chain, residue[1], residue[2].strip()): result[(chain, residue)][2]
            for chain, residue in result.keys()  # noqa: SIM118 -- Bio.PDB iterates values, keys required
        }
        return values, {
            "status": "computed",
            "executable": executable,
            "coordinates": "original first-model file (DSSP conformer policy)",
        }
    except Exception as error:
        if mode == "required":
            raise ValueError(f"DSSP failed: {error}") from error
        return {}, {"status": "failed", "reason": str(error)}


def residue_features(parsed, rows, settings, path, semantics):
    points = settings.get("sasa_points", 960)
    if not isinstance(points, int) or points < 20:
        raise ValueError("SASA sphere sampling must use at least 20 points")
    ShrakeRupley(probe_radius=1.4, n_points=points).compute(parsed["model"], level="R")
    ss, ss_info = secondary_structure(parsed, path, settings.get("dssp", "auto"))
    features = []
    for row in rows:
        record = row["record"]
        feature = {key: value for key, value in row.items() if key not in ("record", "has_ca")}
        feature.update(
            ca=None,
            asa_a2=None,
            rsa=None,
            rsa_above_one=False,
            secondary_structure=None,
            bfactor_a2=None,
            plddt=None,
            low_confidence=None,
            raw_ca_b_iso_or_equiv=None,
        )
        if record is not None:
            residue = record["residue"]
            feature["asa_a2"] = float(residue.sasa)
            maximum = MAX_ASA.get(record["letter"])
            if maximum is not None:
                feature["rsa"] = float(residue.sasa / maximum)
                feature["rsa_above_one"] = feature["rsa"] > 1
            if "CA" in residue:
                feature["ca"] = residue["CA"].coord.astype(float).tolist()
                raw_b = residue["CA"].xtra.get("raw_bfactor", residue["CA"].bfactor)
                raw_b = float(raw_b) if raw_b is not None and np.isfinite(raw_b) else None
                feature["raw_ca_b_iso_or_equiv"] = raw_b
                feature.update(confidence_features(raw_b, semantics))
            feature["secondary_structure"] = ss.get(
                (row["chain"], row["author_number"], row["insertion_code"])
            )
        features.append(feature)
    return features, {
        "algorithm": "Biopython Shrake-Rupley",
        "probe_radius_a": 1.4,
        "sphere_points": points,
        "environment": "entire selected deposited model, occupied heavy atoms including ligands/other chains; no waters/H/D; no symmetry mates",
        "rsa_table": "Tien et al. 2013 theoretical maxima (Wilke), doi:10.1371/journal.pone.0080635",
        "max_asa_a2": MAX_ASA,
        "rsa_clipped": False,
        "secondary_structure": ss_info,
    }


def contact_graph(rows, features, cutoff=8.0, backbone_cutoff=1.8):
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Contact cutoff must be finite and positive")
    edges = []
    observed = [(left_index, row) for left_index, row in enumerate(rows) if row["has_ca"]]
    for offset, (left_index, left) in enumerate(observed):
        left_record = left["record"]
        for right_index, right in observed[offset + 1 :]:
            right_record = right["record"]
            distance = float(
                np.linalg.norm(
                    left_record["residue"]["CA"].coord.astype(float)
                    - right_record["residue"]["CA"].coord.astype(float)
                )
            )
            sequential = right["reference_position"] - left["reference_position"] == 1
            adjacent_polymer = right_record["polymer_index"] - left_record["polymer_index"] == 1
            backbone = (
                adjacent_polymer
                and left_record["segment"] == right_record["segment"]
                and "C" in left_record["residue"]
                and "N" in right_record["residue"]
                and float(
                    np.linalg.norm(
                        left_record["residue"]["C"].coord - right_record["residue"]["N"].coord
                    )
                )
                <= backbone_cutoff
            )
            spatial = distance <= cutoff
            if spatial or backbone:
                edges.append(
                    {
                        "source": left_index + 1,
                        "target": right_index + 1,
                        "distance_a": distance,
                        "spatial": spatial,
                        "backbone": bool(backbone),
                        "sequence_neighbor": sequential,
                        "nonlocal": abs(right_index - left_index) > 2,
                    }
                )
    for feature in features:
        position = feature["reference_position"]
        touching = [
            edge
            for edge in edges
            if position in (edge["source"], edge["target"]) and edge["spatial"]
        ]
        feature["contact_degree"] = len(touching) if feature["ca"] else None
        feature["contact_density_per_a3"] = (
            len(touching) / (4 * np.pi * cutoff**3 / 3) if feature["ca"] else None
        )
        feature["nonlocal_contact_degree"] = (
            sum(edge["nonlocal"] for edge in touching) if feature["ca"] else None
        )
    breaks = []
    for (_, left), (_, right) in itertools.pairwise(observed):
        left_record, right_record = left["record"], right["record"]
        if (
            right_record["polymer_index"] - left_record["polymer_index"] != 1
            or left_record["segment"] != right_record["segment"]
        ):
            reason = "missing polymer residues or TER boundary"
        elif "C" not in left_record["residue"] or "N" not in right_record["residue"]:
            reason = "missing peptide backbone atoms"
        elif (
            float(
                np.linalg.norm(
                    left_record["residue"]["C"].coord - right_record["residue"]["N"].coord
                )
            )
            > backbone_cutoff
        ):
            reason = "C-N distance exceeds peptide-bond cutoff"
        else:
            continue
        breaks.append(
            {
                "left": left["reference_position"],
                "right": right["reference_position"],
                "reason": reason,
            }
        )
    return edges, breaks


def parse_mutation_set(tokens, reference):
    if isinstance(tokens, str):
        tokens = tokens.split(";") if tokens else []
    sites = {}
    for token in tokens:
        match = re.fullmatch(
            r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])", token
        )
        if not match:
            raise ValueError(f"Invalid substitution: {token}")
        old, position, new = match.groups()
        position = int(position)
        if position > len(reference) or reference[position - 1] != old:
            raise ValueError(f"Mutation WT identity/position mismatch: {token}")
        if old == new or position in sites:
            raise ValueError("Mutation set contains no-op, duplicate or conflicting site")
        sites[position] = token
    return [sites[position] for position in sorted(sites)]


def variant_context(identifier, tokens, reference, features, edges):
    normalized = parse_mutation_set(tokens, reference)
    positions = [int(token[1:-1]) for token in normalized]
    sites, coordinates = [], []
    for token, position in zip(normalized, positions, strict=True):
        feature = features[position - 1]
        usable = feature["status"] == "mapped"
        neighbors = (
            sorted(
                {
                    edge["target"] if edge["source"] == position else edge["source"]
                    for edge in edges
                    if position in (edge["source"], edge["target"]) and edge["spatial"]
                }
            )
            if usable
            else []
        )
        site = {
            "mutation": token,
            "reference_position": position,
            "chain": feature["chain"],
            "author_number": feature["author_number"],
            "insertion_code": feature["insertion_code"],
            "mapping_status": feature["status"],
            "usable": usable,
            "spatial_neighbors": neighbors,
            "sequence_neighbors": list(
                range(max(1, position - 2), min(len(reference), position + 2) + 1)
            ),
        }
        site["sequence_neighbors"].remove(position)
        for name in (
            "rsa",
            "contact_degree",
            "nonlocal_contact_degree",
            "plddt",
            "bfactor_a2",
            "low_confidence",
        ):
            site[name] = feature[name] if usable else None
        sites.append(site)
        coordinates.append(feature["ca"] if usable else None)
    distances = [
        [
            float(np.linalg.norm(np.array(a) - b)) if a is not None and b is not None else None
            for b in coordinates
        ]
        for a in coordinates
    ]
    summary = {
        "n_sites": len(sites),
        "n_mapped": sum(site["usable"] for site in sites),
        "n_unmapped": sum(not site["usable"] for site in sites),
        "n_low_plddt": sum(site["low_confidence"] is True for site in sites),
        "spatial_neighborhood_union": sorted(
            {n for site in sites for n in site["spatial_neighbors"]}
        ),
    }
    for name in ("rsa", "contact_degree", "nonlocal_contact_degree", "plddt", "bfactor_a2"):
        values = [site[name] for site in sites if site[name] is not None]
        summary["mean_" + name] = float(np.mean(values)) if values else None
    pairs = [
        distances[i][j]
        for i in range(len(sites))
        for j in range(i + 1, len(sites))
        if distances[i][j] is not None
    ]
    summary["mean_pair_distance_a"] = float(np.mean(pairs)) if pairs else None
    summary["min_pair_distance_a"] = min(pairs) if pairs else None
    return {
        "id": identifier,
        "context": "reference-structure context; no mutant coordinate relaxation",
        "mutation_set": normalized,
        "sites": sites,
        "mutation_distance_matrix_a": distances,
        "summary": summary,
    }
