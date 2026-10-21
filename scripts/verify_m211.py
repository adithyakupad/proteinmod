"""Offline document/metadata integrity checks; no research imports or network."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "benchmarks/m211"
IDS = {"GCK", "ASPA", "PRKN", "SOD1"}
FORBIDDEN_KEYS = {
    "scoredistribution",
    "measurements",
    "predictions",
    "clinicalclassifications",
    "scores",
    "score",
    "llr",
    "evidence_weight",
}


def load(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def metadata_boundary(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"Excluded outcome field: {key}")
            metadata_boundary(child)
    elif isinstance(value, list):
        for child in value:
            metadata_boundary(child)


def qualification_boundary(record):
    rows = record["candidates"]
    if len(rows) != 4 or {row["id"] for row in rows} != IDS:
        raise ValueError("Exactly the four frozen candidate identities are required")
    if record["decision"] != "NO_GO" or record["four_family_denominator"] != 4:
        raise ValueError("M21.1 preserves the four-family no-go decision")
    for key in (
        "cohort_admitted",
        "outcome_access_authorized",
        "outcome_manifest_issued",
        "new_variants_generated",
        "models_executed",
    ):
        if record[key] is not False:
            raise ValueError(f"M21.1 does not authorize {key}")
    if record["candidate_search_closed"] is not True:
        raise ValueError("No open-ended candidate search")
    for row in rows:
        if row["overall"] != "PARTIAL" or not row["blockers"]:
            raise ValueError("Unresolved blockers must remain explicit")
        if any(gate not in {"PASS", "PARTIAL", "FAIL"} for gate in row["gates"].values()):
            raise ValueError("Invalid gate status")
        if all(gate == "PASS" for gate in row["gates"].values()):
            raise ValueError("This stopped packet cannot claim complete admission")


def safe_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path escapes repository")
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    for path in PACKET.rglob("*.json"):
        record = load(path)
        if path.parent.name == "evidence":
            metadata_boundary(record)
    qualification = load(PACKET / "qualification.json")
    qualification_boundary(qualification)
    for row in qualification["candidates"]:
        for reference in row["evidence"]:
            if not (PACKET / reference).is_file():
                raise ValueError(f"Missing candidate evidence: {reference}")
    for path in PACKET.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not (path.parent / target.split("#")[0]).is_file():
                raise ValueError(f"Broken document link: {path.name}: {target}")
    historical = load(PACKET / "historical_hashes.json")
    if len(historical) != 953:
        raise ValueError("Main baseline inventory changed")
    for name, expected in historical.items():
        if digest(safe_path(name).read_bytes()) != expected:
            raise ValueError(f"Historical working file changed: {name}")
    baseline = load(PACKET / "evidence/preservation-baseline.json")
    for ref, expected in baseline["preserved_local_branch_refs"].items():
        if git("rev-parse", ref).decode().strip() != expected:
            raise ValueError(f"Historical branch moved: {ref}")
    archive = baseline["starting_archive"]
    for name, expected in baseline["archive_files_sha256"].items():
        # Opaque hashing only; never parse a historical phenotype/prediction file.
        if digest(git("cat-file", "blob", f"{archive}:{name}")) != expected:
            raise ValueError(f"Archive byte mismatch: {name}")
    identities = load(PACKET / "artifact_hashes.json")
    for name, expected in identities.items():
        if digest(safe_path(name).read_bytes()) != expected:
            raise ValueError(f"M21.1 artifact changed: {name}")
    print(
        f"PASS: four-candidate stopped metadata packet; {len(historical)} main files; "
        f"{len(baseline['archive_files_sha256'])} archive files; "
        "historical branch refs, evidence hashes, JSON and document links"
    )


if __name__ == "__main__":
    main()
