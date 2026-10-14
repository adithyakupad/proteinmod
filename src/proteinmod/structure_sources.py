"""Opt-in official coordinate acquisition and checksum-bound local structures."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from proteinmod.data import checksum, write_json


def acquire_structure(spec):
    kind = spec["kind"]
    file_format = spec.get("format", "cif")
    if file_format not in ("cif", "pdb"):
        raise ValueError("Structure format must be cif or pdb")
    path = Path(spec["path"])
    if kind == "local":
        verify_structure(spec)
        return path
    metadata = None
    if kind == "pdb":
        identifier = spec["id"].upper()
        if not re.fullmatch(r"[0-9][A-Z0-9]{3}", identifier):
            raise ValueError("Use an explicit four-character PDB ID")
        url = f"https://files.rcsb.org/download/{identifier}.{file_format}"
        license_name = "wwPDB CC0; cite deposition and primary publication"
    elif kind == "alphafold":
        accession = spec["accession"]
        if not re.fullmatch(r"[A-Z0-9]+(?:-\d+)?", accession):
            raise ValueError("Invalid explicit AlphaFold accession")
        api = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
        with urlopen(api, timeout=60) as response:
            metadata = json.load(response)
        candidates = [item for item in metadata if item["entryId"] == spec["entry_id"]]
        if len(candidates) != 1:
            raise ValueError("AlphaFold entry ambiguous/absent; specify exact entry_id")
        record = candidates[0]
        url = record["cifUrl" if file_format == "cif" else "pdbUrl"]
        if not url.startswith("https://alphafold.ebi.ac.uk/files/"):
            raise ValueError("AlphaFold API returned an unexpected coordinate source")
        license_name = "AlphaFold DB CC-BY-4.0; cite database and AlphaFold publications"
    else:
        raise ValueError("Structure source kind must be pdb, alphafold or local")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        content = response.read()
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        if spec.get("sha256") and checksum(temporary) != spec["sha256"]:
            raise ValueError("Structure checksum mismatch; original cached file preserved")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(
        path.with_suffix(path.suffix + ".source.json"),
        {
            "specification": spec,
            "url": url,
            "sha256": checksum(path),
            "license": license_name,
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "provider_metadata": metadata,
        },
    )
    return path


def verify_structure(spec):
    if spec.get("kind") not in ("local", "pdb", "alphafold"):
        raise ValueError("Structure source kind must be pdb, alphafold or local")
    path = Path(spec["path"])
    if not spec.get("sha256"):
        raise ValueError("A pinned structure SHA256 is required for offline extraction")
    if checksum(path) != spec["sha256"]:
        raise ValueError("Structure checksum mismatch")
    if not spec.get("source") or not spec.get("license"):
        raise ValueError("Structure provenance requires source attribution and license")
    if spec.get("confidence") not in ("experimental_bfactor", "alphafold_plddt", "unknown"):
        raise ValueError("Specify confidence semantics explicitly; never infer pLDDT from B-factor")
    if spec["kind"] == "pdb" and spec["confidence"] != "experimental_bfactor":
        raise ValueError("Experimental PDB acquisition cannot be labeled AlphaFold pLDDT")
    if spec["kind"] == "alphafold" and spec["confidence"] != "alphafold_plddt":
        raise ValueError("AlphaFold acquisition must explicitly label pLDDT")
    return path
