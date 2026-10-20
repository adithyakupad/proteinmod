"""Outcome-blind primary sequence inventory; search scores never establish WT identity."""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from proteinmod.data import checksum, write_json


def fetch_batch(identifiers):
    query = (
        "{polymer_entities(entity_ids:"
        + json.dumps(identifiers)
        + "){rcsb_id entity_poly{pdbx_seq_one_letter_code_can} "
        + "rcsb_polymer_entity_container_identifiers{auth_asym_ids}}}"
    )
    request = urllib.request.Request(
        "https://data.rcsb.org/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("errors"):
        raise ValueError(result["errors"])
    rows = result["data"]["polymer_entities"]
    if any(row is None for row in rows) or {row["rcsb_id"] for row in rows} != set(identifiers):
        raise ValueError("Incomplete primary entity response")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/m184/evidence"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    query = json.loads(Path("benchmarks/m183/evidence/GFP-search-request.json").read_text())
    query["request_options"] = {"return_all_hits": True}
    request = urllib.request.Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    identifiers = [row["identifier"] for row in result["result_set"]]
    assert len(identifiers) == result["total_count"] == len(set(identifiers))
    batches = [identifiers[start : start + 40] for start in range(0, len(identifiers), 40)]
    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in executor.map(fetch_batch, batches):
            rows.extend(batch)
    reference = query["query"]["parameters"]["value"]
    inventory = []
    for row in sorted(rows, key=lambda record: record["rcsb_id"]):
        sequence = row["entity_poly"]["pdbx_seq_one_letter_code_can"].replace("\n", "")
        inventory.append(
            {
                "entity": row["rcsb_id"],
                "polymer_sequence": sequence,
                "chains": row["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"],
                "length": len(sequence),
                "exact_reference_start_zero_based": sequence.find(reference),
            }
        )
    exact = [row for row in inventory if row["exact_reference_start_zero_based"] >= 0]
    write_json(args.output / "GFP-search-query.json", query)
    write_json(args.output / "GFP-search-response.json", result)
    write_json(args.output / "GFP-polymer-inventory.json", inventory)
    write_json(
        args.output / "GFP-exact-sequence-audit.json",
        {
            "search_endpoint": "https://search.rcsb.org/rcsbsearch/v2/query",
            "metadata_endpoint": "https://data.rcsb.org/graphql",
            "reference": reference,
            "entities": len(inventory),
            "exact_segment_entities": exact,
            "inventory_sha256": checksum(args.output / "GFP-polymer-inventory.json"),
            "candidate_target_cells_accessed": False,
            "scope": "All returned95%-identity sequence hits, not an exhaustive proof over every structure database. Exact polymer candidates still require coordinate/chemical mapping.",
        },
    )
    print("Primary entities", len(inventory), "exact complete assay segments", len(exact))
    print([row["entity"] for row in exact])
