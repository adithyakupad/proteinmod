import hashlib
import json
from collections import Counter
from pathlib import Path

from Bio.Seq import Seq

from proteinmod.admission_masks import compatible_positions, gfp_mutations
from proteinmod.blinded_homology import tem_genotypes
from proteinmod.data import checksum, write_json

root = Path("data/cache/m182/source")
out = Path("benchmarks/m182")
assert (
    checksum(root / "GFP-reference.fa")
    == "2559ee1afe9d0f41e03ea75db96448a34178c2dd597c78e08f01b1ee4ab5e248"
)
assert (
    checksum(root / "GFP-primary.tsv")
    == "5581230b9888f9960708a4769ce1c9b4453561f0b5f0374525445b4a984d54d5"
)
dna = "".join((root / "GFP-reference.fa").read_text().splitlines()[1:])
translation = str(Seq(dna).translate())
assert translation.endswith("*") and "*" not in translation[:-1]
ref = translation[:-1]
audits = json.loads((out / "evidence/structure-mask-audits.json").read_text())
masks = {
    pid: compatible_positions(
        next(a for a in audits if a["protein_id"] == pid and a["pdb"] == pdb)["residues"]
    )
    for pid, pdb in [("TEM1", "1XPB"), ("GFP", "1GFL")]
}
rows = []
excluded = Counter()
hist = Counter()
seen = set()
raw_n = 0
with (root / "GFP-primary.tsv").open() as f:
    header = f.readline().rstrip("\r\n").split("\t")
    assert header == ["aaMutations", "uniqueBarcodes", "medianBrightness", "std"]
    for number, line in enumerate(f, 1):
        raw_n += 1
        tokens, reason = gfp_mutations(line, ref)
        if reason:
            excluded[reason] += 1
            continue
        if not tokens:
            excluded["WT"] += 1
            continue
        pos = [int(t[1:-1]) for t in tokens]
        hist[len(pos)] += 1
        s = list(ref)
        for t in tokens:
            s[int(t[1:-1]) - 1] = t[-1]
        seq = "".join(s)
        assert seq not in seen
        seen.add(seq)
        if set(pos) - set(masks["GFP"]):
            excluded["outside_common_structure_mask"] += 1
            continue
        rows.append(
            {
                "id": "Sarkisyan2016:" + hashlib.sha256(seq.encode()).hexdigest()[:16],
                "protein_id": "GFP",
                "source_row": number,
                "sequence": seq,
                "mutations": tokens,
                "candidate_role": "test_only_if_admitted",
            }
        )
rows = sorted(
    rows, key=lambda r: hashlib.sha256(("m18:external:GFP" + r["sequence"]).encode()).hexdigest()
)
selected = rows[:96]
assert len(selected) == 96
write_json(
    out / "GFP-genotype-audit.json",
    {
        "reference": ref,
        "reference_sha256": hashlib.sha256(ref.encode()).hexdigest(),
        "reference_DNA_sha256": checksum(root / "GFP-reference.fa"),
        "dataset_sha256": checksum(root / "GFP-primary.tsv"),
        "raw_rows": raw_n,
        "unique_canonical_nonWT_sequences": len(seen),
        "common_mask_candidate_count": len(rows),
        "exclusions": dict(excluded),
        "canonical_mutation_burden": dict(hist),
        "selected_candidates": selected,
        "target_cells_accessed": False,
        "reference_semantics": "Author DNA translation minus termination; lacks initiator M. All source positions zero-based in this237aa segment. No native WT or extra initiator is inferred.",
    },
)
assert (
    checksum("data/cache/m181/source/TEM1-Data-S1-S4.xlsx")
    == "bcdd031301113ede74652d4e2eba3f03cf48eb46222d92ccab5c8787950aa5d3"
)
tem = tem_genotypes("data/cache/m181/source/TEM1-Data-S1-S4.xlsx")
canonical = json.loads(Path("benchmarks/m181/TEM1-qualification.json").read_text())[
    "canonical_sequence"
]
positions = sorted({r["position"] for r in tem})
index = {p: i + 1 for i, p in enumerate(positions)}
records = []
te = Counter()
for r in tem:
    p = index[r["position"]]
    if r["wt"] == "*":
        te["termination_site"] += 1
        continue
    assert canonical[p - 1] == r["wt"]
    if r["mutant"] == r["wt"]:
        te["WT_symbol"] += 1
        continue
    if p not in masks["TEM1"]:
        te["outside_common_structure_mask"] += 1
        continue
    seq = canonical[: p - 1] + r["mutant"] + canonical[p:]
    records.append(
        {
            "id": f"Firnberg2014:{r['wt']}{p}{r['mutant']}",
            "protein_id": "TEM1",
            "source_row": r["source_row"],
            "sequence": seq,
            "mutations": [f"{r['wt']}{p}{r['mutant']}"],
            "candidate_role": "test_only_if_admitted",
        }
    )
records = sorted(
    records,
    key=lambda r: hashlib.sha256(("m18:external:TEM1" + r["sequence"]).encode()).hexdigest(),
)
write_json(
    out / "TEM1-common-mask-audit.json",
    {
        "source_rows": len(tem),
        "common_mask_candidate_count": len(records),
        "exclusions": dict(te),
        "selected_candidates": records[:96],
        "common_valid_positions": masks["TEM1"],
        "target_cells_accessed": False,
        "accepted_graph": False,
        "reason": "Strict full-reference M2 coverage remains below1.0; mask is prospective metadata only, not accepted graph.",
    },
)
print("GFP", raw_n, "unique canonical", len(seen), "common", len(rows), "excluded", dict(excluded))
print("TEM1 common", len(records), "excluded", dict(te))
