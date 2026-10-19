import collections
import hashlib
import json
from pathlib import Path

from Bio.Seq import Seq

from proteinmod.structure_mapping import map_reference, parse_structure

root = Path("data/cache/m182/source")
tem = json.loads(Path("benchmarks/m181/TEM1-qualification.json").read_text())["canonical_sequence"]
gfp = str(Seq("".join((root / "GFP-reference.fa").read_text().splitlines()[1:])).translate())[:-1]
expected = {
    r["pdb"]: r["coordinate_sha256"]
    for r in json.loads(Path("benchmarks/m182/evidence/structure-mask-audits.json").read_text())
}
assert (
    hashlib.sha256((root / "GFP-reference.fa").read_bytes()).hexdigest()
    == "2559ee1afe9d0f41e03ea75db96448a34178c2dd597c78e08f01b1ee4ab5e248"
)
out = []
for pid, ref, pdbs in [
    ("TEM1", tem, ["1XPB", "1ZG4"]),
    ("GFP", gfp, ["1GFL", "2WUR", "2B3P", "2G2S", "2G3D"]),
]:
    for pdb in pdbs:
        p = root / f"{pdb}.cif"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected[pdb]
        try:
            parsed = parse_structure(p, 0)
            rows, a = map_reference(
                parsed,
                ref,
                {"chain": "A", "min_coverage": 1.0, "min_identity": 1.0, "allowed_mismatches": []},
            )
            slim = [
                {
                    k: r[k]
                    for k in [
                        "reference_position",
                        "reference_aa",
                        "structure_aa",
                        "chain",
                        "status",
                        "author_number",
                        "insertion_code",
                        "noncanonical",
                        "polymer_position",
                        "has_ca",
                    ]
                }
                for r in rows
            ]
            record = {
                "protein_id": pid,
                "pdb": pdb,
                "coordinate_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "reference": ref,
                "audit": a,
                "residues": slim,
                "status_counts": dict(collections.Counter(r["status"] for r in rows)),
            }
            out.append(record)
            print(
                pid,
                pdb,
                "accepted",
                a["accepted"],
                "identity",
                a.get("aligned_identity"),
                "CAcoverage",
                a.get("coordinate_coverage"),
                "differences",
                a.get("differences"),
                "counts",
                record["status_counts"],
            )
        except (OSError, ValueError) as e:
            out.append({"protein_id": pid, "pdb": pdb, "error": type(e).__name__ + ": " + str(e)})
            print(pid, pdb, type(e).__name__, str(e))
(Path("benchmarks/m182/evidence/structure-mask-audits.json")).write_text(
    json.dumps(out, indent=2) + "\n"
)
