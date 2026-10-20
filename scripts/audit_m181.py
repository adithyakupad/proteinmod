"""Recompute four blinded sequence diagnostics from exact historical references."""

import json
from pathlib import Path

from proteinmod.blinded_homology import alignment_diagnostics
from proteinmod.data import write_json
from proteinmod.external_gate import similarity_review

root = Path("benchmarks/m181")
refs = json.loads(Path("benchmarks/m18/evidence/author-references.json").read_text())
refs["BBC1"] = json.loads(Path("configs/m12.json").read_text())["proteins"][0]["reference"][
    "sequence"
]
for spec in json.loads(Path("configs/m6.json").read_text())["proteins"][:2]:
    refs[spec["protein_id"]] = spec["reference"]["sequence"]
settings = json.loads(Path("configs/m12.json").read_text())["similarity"]
rows = []
for a, b in [("P0A9X9", "BBC1"), ("P01053", "BBC1"), ("P01053", "1LP1"), ("P01053", "1A32")]:
    result = alignment_diagnostics(refs[a], refs[b])
    rows.append(
        {
            "a": a,
            "b": b,
            **result,
            "original_flag": similarity_review(result["global"], result["local"], settings),
        }
    )
write_json(root / "alignment-diagnostics.json", {"settings": settings, "pairs": rows})
assert len(rows) == 4 and all(r["original_flag"] for r in rows)
print("All four original flags retained; full-sequence identity diagnostics written")
