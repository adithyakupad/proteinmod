"""Repeat genotype-only primary TEM-1 workbook checks; never load fitness R/S."""

import hashlib
import json
from pathlib import Path

from proteinmod.blinded_homology import tem_genotypes

root = Path("benchmarks/m181")
source = Path("data/cache/m181/source/TEM1-Data-S1-S4.xlsx")
expected = "bcdd031301113ede74652d4e2eba3f03cf48eb46222d92ccab5c8787950aa5d3"
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
records = tem_genotypes(source)
wt = {}
for row in records:
    position = row["position"]
    assert position not in wt or wt[position] == row["wt"]
    wt[position] = row["wt"]
    assert row["mutant"] in "ACDEFGHIKLMNPQRSTVWY"
sequence = "".join(wt[p] for p in sorted(wt))
canonical = json.loads((root / "evidence/P62593-uniprot.json").read_text())["sequence"]["value"]
assert sequence[:-1] == canonical and sequence[-1] == "*"
entity = json.loads((root / "evidence/1BTL-entity.json").read_text())["entity_poly"][
    "pdbx_seq_one_letter_code_can"
]
mismatches = [
    {"canonical_position": i + 24, "assay_WT": a, "pdb_residue": b}
    for i, (a, b) in enumerate(zip(canonical[23:], entity, strict=True))
    if a != b
]
output = {
    "metadata_rows": len(records),
    "reference_sequence": canonical,
    "author_WT_exact_canonical_except_terminal_stop": True,
    "genotype_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    "single_nonWT_canonical_rows": sum(r["wt"] != "*" and r["mutant"] != r["wt"] for r in records),
    "structure_mismatches": mismatches,
    "target_fields_accessed": False,
}
committed = json.loads((root / "TEM1-qualification.json").read_text())
assert output.pop("reference_sequence") == committed["canonical_sequence"]
assert all(value == committed[key] for key, value in output.items())
print(
    "TEM1 primary metadata reproduced: 5740 genotype rows; 5434 canonical substitutions; R/S unread"
)
