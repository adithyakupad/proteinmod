import csv
import gzip
import hashlib
import json
from pathlib import Path

from proteinmod.data import checksum, write_json
from proteinmod.external_gate import project_row
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import run_structure
from proteinmod.structure_sources import acquire_structure

SOURCE_SHA256 = "4cd224d56a0004be70f1b33f055cb2a0d2846065497c94e4fcb8bb289a167b84"
STRUCTURE_SHA256 = {
    "1MJC": "2ce3a3b1694eb4a07930bcd4350892066ce6b064dffb495435427e645013d9f4",
    "3CI2": "bf5e06fc219ebce54da5ab1d7b252694e1fd41e6655f78c0bd72e251521f3100",
}
assert checksum("data/cache/m18/source/primary-scores.csv.gz") == SOURCE_SHA256

refs = json.loads(Path("data/cache/m18/source/author-references.json").read_text())
family = {"P0A9X9": "PF00313", "P01053": "PF00280"}
cores = {"P0A9X9": [4, 16, 25, 27, 46, 48, 62], "P01053": [11, 18, 27, 45, 47, 49, 55]}
projected = {p: {} for p in family}
counts = {}
excluded = []
with gzip.open("data/cache/m18/source/primary-scores.csv.gz", "rt") as f:
    for number, row in enumerate(csv.DictReader(f), 1):
        row = project_row(row)
        pid = row["UniprotCode"]
        counts[pid] = counts.get(pid, 0) + 1
        if pid not in family:
            continue
        seq = row["aa_seq"]
        ref = refs[pid]
        if len(seq) != len(ref) or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            excluded.append({"row": number, "reason": "noncanonical/length"})
            continue
        positions = [i for i, (a, b) in enumerate(zip(ref, seq)) if a != b]
        if not positions:
            continue
        if set(positions) - set(cores[pid]) or any(seq[i] not in "FLIMV" for i in positions):
            excluded.append({"row": number, "reason": "outside randomized core/alphabet"})
            continue
        assert int(float(row["Nham_aa"])) == len(positions)
        assert row["core"] == "".join(seq[i] for i in cores[pid])
        assert seq not in projected[pid], ("duplicate amino-acid genotype", pid)
        projected[pid][seq] = {
            "id": f"GSE266299:{pid}:" + hashlib.sha256(seq.encode()).hexdigest()[:16],
            "protein_id": pid,
            "family": family[pid],
            "sequence": seq,
            "mutations": [f"{ref[i]}{i + 1}{seq[i]}" for i in positions],
            "source_row": number,
            "role": "test",
        }
rows = []
specs = []
graphs = {}
for pid, pdb in [("P0A9X9", "1MJC"), ("P01053", "3CI2")]:
    u = json.loads(Path(f"data/cache/m18/source/{pid}-uniprot.json").read_text())
    assert family[pid] in {
        r["id"] for r in u["uniProtKBCrossReferences"] if r["database"] == "Pfam"
    }
    ref = refs[pid]
    spec = {
        "protein_id": pid,
        "family": family[pid],
        "assay_id": "GSE266299_sparse_DTS_abundancePCA",
        "reference": {
            "sequence": ref,
            "sha256": hashlib.sha256(ref.encode()).hexdigest(),
            "source": "https://github.com/lehner-lab/combinatorialcores/blob/"
            + "f978897ef3a278796acc0e87e5cd82d4d0d10082"
            + "/MainFigures/Fig2.ipynb (translated author WT DNA)",
        },
        "structure": {
            "name": pid + " reference " + pdb,
            "sha256": STRUCTURE_SHA256[pdb],
            "id": pdb,
            "kind": "pdb",
            "format": "cif",
            "path": f"data/cache/m18/source/{pdb}.cif",
            "confidence": "experimental_bfactor",
            "source": f"https://www.rcsb.org/structure/{pdb}",
            "license": "wwPDB CC0",
        },
        "mapping": {
            "chain": "A",
            "model_index": 0,
            "min_coverage": 1.0,
            "min_identity": 1.0,
            "allowed_mismatches": [],
        },
        "structure_artifacts": f"data/cache/m18/graphs/{pid}",
        "dataset": {
            "path": "data/cache/m18/source/primary-scores.csv.gz",
            "sha256": checksum("data/cache/m18/source/primary-scores.csv.gz"),
        },
    }
    acquire_structure(spec["structure"])
    spec["structure"]["sha256"] = checksum(spec["structure"]["path"])
    settings = {
        "structure": spec["structure"],
        "reference": spec["reference"],
        "mapping": spec["mapping"],
        "geometry": {"contact_cutoff_a": 8.0, "sasa_points": 960, "dssp": "off"},
        "structure_run": {"output": spec["structure_artifacts"]},
    }
    run_structure(settings, render=False)
    graph = load_graph(spec["structure_artifacts"], ref)
    candidates = sorted(
        projected[pid].values(),
        key=lambda r: hashlib.sha256(("m18:external:" + pid + r["sequence"]).encode()).hexdigest(),
    )
    selected = candidates[:96]
    assert len(selected) == 96
    for r in selected:
        _, _, reasons = variant_inputs(r["mutations"], ref, graph)
        assert not reasons, (r["id"], reasons)
    rows.extend(selected)
    specs.append(spec)
    graphs[pid] = graph["audit"]
write_json(
    "data/cache/m18/source/blind-index.json",
    {
        "source_sha256": checksum("data/cache/m18/source/primary-scores.csv.gz"),
        "counts": counts,
        "selected": rows,
        "specs": specs,
        "graphs": graphs,
        "excluded": excluded,
        "target_fields_accessed": False,
    },
)
print(
    "BLIND CENSUS",
    counts,
    "eligible genotypes",
    {p: len(v) for p, v in projected.items()},
    "selected",
    len(rows),
    "exclusions",
    len(excluded),
)
