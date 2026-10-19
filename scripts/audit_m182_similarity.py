"""Unchanged sequence review flags for all four blinded external references."""

import json
from itertools import combinations
from pathlib import Path

from proteinmod.blinded_homology import alignment_diagnostics
from proteinmod.data import write_json
from proteinmod.external_gate import similarity_review

root = Path("benchmarks/m182")
panel = json.loads((root / "reference-panel.json").read_text())
settings = json.loads(Path("configs/m12.json").read_text())["similarity"]
comparisons = [
    (external, old) for external in panel["external_candidates"] for old in panel["historical"]
] + list(combinations(panel["external_candidates"], 2))
rows = []
for a, b in comparisons:
    d = alignment_diagnostics(a["sequence"], b["sequence"])
    rows.append(
        {
            "a": a["id"],
            "b": b["id"],
            **d,
            "original_flag": similarity_review(d["global"], d["local"], settings),
        }
    )
write_json(root / "similarity-audit.json", {"settings": settings, "pairs": rows})
assert len(rows) == 78
print("78 comparisons; flags", [(r["a"], r["b"]) for r in rows if r["original_flag"]])
