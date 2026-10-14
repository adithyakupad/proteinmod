"""Explicit downloads, biological validation and official split interpretation."""

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PARTITIONS = ("train", "validation", "test")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_file(path, specification):
    content = Path(path).read_bytes()
    for algorithm in ("md5", "sha256"):
        expected = specification.get(algorithm)
        if expected and hashlib.new(algorithm, content).hexdigest() != expected:
            raise ValueError(f"{algorithm} checksum mismatch: {path}")


def download(specification):
    path = Path(specification["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # Verify before replacing the cached file, including on failed downloads.
    with urlopen(specification["url"], timeout=60) as response:
        content = response.read()
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(content)
        verify_file(temporary, specification)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(
        path.with_suffix(path.suffix + ".manifest.json"),
        {
            "source": specification,
            "sha256": checksum(path),
            "bytes": path.stat().st_size,
        },
    )
    return path


def valid_sequence(sequence):
    if not isinstance(sequence, str) or not sequence or set(sequence) - set(AMINO_ACIDS):
        raise ValueError("Sequence must contain only the 20 canonical uppercase amino acids")


def mutations(reference, sequence):
    valid_sequence(reference)
    valid_sequence(sequence)
    if len(reference) != len(sequence):
        raise ValueError("Reference/variant length mismatch; substitutions only")
    return tuple(
        f"{old}{position}{new}"
        for position, (old, new) in enumerate(zip(reference, sequence), 1)
        if old != new
    )


def verified_reference(specification):
    if not specification:
        return None
    sequence = specification.get("sequence", "")
    if not specification.get("source"):
        raise ValueError("WT reference requires provenance; never infer WT from variants")
    valid_sequence(sequence)
    if not specification.get("git_blob") and not specification.get("sha256"):
        raise ValueError("WT reference requires a provenance checksum")
    if (
        specification.get("sha256")
        and hashlib.sha256(sequence.encode()).hexdigest() != specification["sha256"]
    ):
        raise ValueError("WT reference checksum mismatch")
    # A Git blob digest binds the exact original FASTA bytes, including its header.
    if specification.get("git_blob"):
        fasta = f"> 0\n{sequence}\n".encode()
        digest = hashlib.sha1(f"blob {len(fasta)}\0".encode() + fasta).hexdigest()
        if digest != specification["git_blob"]:
            raise ValueError("WT reference does not match its provenance Git blob")
    return sequence


def load_dataset(path, reference=None):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sequence", "target", "set", "validation"} <= set(reader.fieldnames or []):
            raise ValueError("Missing required FLIP2 columns")
        raw = list(reader)
    if not raw:
        raise ValueError("Empty dataset")
    rows, ids, sequences = [], set(), set()
    lengths, overlap = Counter(), Counter()
    for index, row in enumerate(raw, 1):
        sequence = row["sequence"]
        valid_sequence(sequence)
        lengths[len(sequence)] += 1
        identifier = row.get("id", str(index))
        if not identifier or identifier in ids:
            raise ValueError("Missing or duplicate variant ID")
        if sequence in sequences:
            raise ValueError("Duplicate sequence: possible partition leakage")
        ids.add(identifier)
        sequences.add(sequence)
        try:
            target = float(row["target"])
        except (ValueError, TypeError) as error:
            raise ValueError("Target must be numeric") from error
        if not math.isfinite(target):
            raise ValueError("Target must be finite")
        assignment, flag = row["set"], str(row["validation"]).lower()
        if assignment not in PARTITIONS or flag not in ("true", "false"):
            raise ValueError("Invalid split assignment or validation boolean")
        is_validation = flag == "true"
        if assignment == "test" and is_validation:
            raise ValueError("Test/validation overlap would leak test labels")
        if assignment == "validation" and not is_validation:
            raise ValueError("Conflicting explicit validation assignment")
        if is_validation:
            overlap[assignment] += 1
        partition = "validation" if is_validation else assignment
        mutation_set = mutations(reference, sequence) if reference else None
        rows.append(
            {
                "id": identifier,
                "sequence": sequence,
                "target": target,
                "partition": partition,
                "mutations": mutation_set,
            }
        )
    if len(lengths) != 1:
        raise ValueError("Inconsistent variant lengths")
    counts = Counter(row["partition"] for row in rows)
    if any(counts[part] == 0 for part in PARTITIONS):
        raise ValueError("Train, validation and test must all be nonempty")
    burden = (
        {
            part: dict(
                sorted(
                    Counter(
                        len(row["mutations"]) for row in rows if row["partition"] == part
                    ).items()
                )
            )
            for part in PARTITIONS
        }
        if reference
        else None
    )
    substitution_overlap = None
    if reference:
        substitutions = {
            part: {
                mutation
                for row in rows
                if row["partition"] == part
                for mutation in row["mutations"]
            }
            for part in PARTITIONS
        }
        substitution_overlap = {
            f"{left}/{right}": len(substitutions[left] & substitutions[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
    qc = {
        "rows": len(rows),
        "lengths": dict(lengths),
        "counts": dict(counts),
        "ids": "source IDs" if "id" in raw[0] else "generated 1-based source row IDs",
        "missing_ids": 0,
        "duplicate_ids": 0,
        "duplicate_sequences": 0,
        "cross_partition_sequence_overlap": False,
        "validation_flag_set_overlap_counts": dict(overlap),
        "split_semantics": "validation=True overrides set=train; test overlap rejected",
        "reference_verified": reference is not None,
        "mutation_burden": burden,
        "unique_substitution_overlap_counts": substitution_overlap,
    }
    return rows, qc
