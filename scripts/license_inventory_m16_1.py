"""Inventory installed distribution license metadata, preserving unknowns for review."""

import importlib.metadata
from pathlib import Path

from proteinmod.data import checksum, write_json


def main():
    records = []
    for distribution in sorted(
        importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
    ):
        metadata = distribution.metadata
        files = [
            p
            for p in distribution.files or []
            if any(word in str(p).lower() for word in ("license", "copying", "notice"))
        ]
        records.append(
            {
                "name": metadata["Name"],
                "version": distribution.version,
                "expression": metadata.get("License-Expression"),
                "legacy_license": metadata.get("License"),
                "classifiers": [
                    c for c in metadata.get_all("Classifier", []) if c.startswith("License")
                ],
                "license_files": {
                    str(p): checksum(distribution.locate_file(p))
                    for p in files
                    if distribution.locate_file(p).is_file()
                },
                "status": "METADATA_ONLY_REQUIRES_REVIEW",
            }
        )
    output = Path("runs/m16.1/license-inventory.json")
    assert not output.exists(), "Preserve prior inventory"
    write_json(output, records)
    print(f"Inventoried {len(records)} distributions; no legal clearance inferred")


if __name__ == "__main__":
    main()
