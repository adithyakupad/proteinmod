"""Fail-closed physical export of historical M8 predictions.

Legacy experiment files retain raw diagnostic values for exact scientific replay.
No M8 function protein has qualified native thermodynamic prediction support.
"""

import argparse
import csv
import json
from pathlib import Path

from proteinmod.data import checksum

M8_SHA256 = "f2d96ade6b4fd6e5b5471ebacf58315d209f20d5816ad230bb2327529408d5dd"


def physical_stability(row):
    """Never promote an archived functional-domain diagnostic to physical ΔΔG."""
    required = {"id", "protein_id", "source", "domain", "source_weights_sha256"}
    if not required <= row.keys() or not all(row[k] for k in required):
        raise ValueError("Missing prediction provenance")
    # shortcut: no deployed model/scope is qualified; add a validated native contract first.
    return {
        "id": row["id"],
        "protein_id": row["protein_id"],
        "model": row["source"],
        "source_weights_sha256": row["source_weights_sha256"],
        "prediction": None,
        "unit": None,
        "thermodynamic_eligible": False,
        "reason": "No qualified native stability inference contract for this prediction",
    }


def export(source, destination):
    source, destination = Path(source), Path(destination)
    if checksum(source) != M8_SHA256:
        raise ValueError("Expected checksum-pinned historical M8 evidence")
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    safe = [physical_stability(row) for row in rows]
    # Exclusive creation prevents replacing historical evidence or a previous export.
    with destination.open("x") as handle:
        json.dump({"source_sha256": M8_SHA256, "predictions": safe}, handle, allow_nan=False)
        handle.write("\n")
    return len(safe)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", default="benchmarks/m8/predicted_stability.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Invalidated {export(args.source, args.output)} unsupported physical predictions")


if __name__ == "__main__":
    main()
