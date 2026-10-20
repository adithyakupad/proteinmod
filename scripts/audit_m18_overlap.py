import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

from proteinmod.data import write_json
from proteinmod.external_gate import similarity_review
from proteinmod.family_data import alignment

refs = {}


def walk(v, where):
    if isinstance(v, dict):
        if isinstance(v.get("reference"), dict) and "sequence" in v["reference"]:
            refs.setdefault(v["reference"]["sequence"], []).append(where)
        for k, x in v.items():
            walk(x, where + "/" + k)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            walk(x, where + "/" + str(i))


for name in subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", "73f0666a87560dadffa76f34f5255856d2ce7459", "configs"],
    text=True,
).splitlines():
    if name.endswith((".json", ".toml")):
        p = Path(name)
        raw = subprocess.check_output(
            ["git", "show", "73f0666a87560dadffa76f34f5255856d2ce7459:" + name], text=True
        )
        walk(json.loads(raw) if p.suffix == ".json" else tomllib.loads(raw), name)
settings = json.loads(Path("configs/m12.json").read_text())["similarity"]
pairs = []
for spec in json.loads(Path("data/cache/m18/source/blind-index.json").read_text())["specs"]:
    for sequence, where in refs.items():
        g = alignment(spec["reference"]["sequence"], sequence)
        l = alignment(spec["reference"]["sequence"], sequence, "local")
        flag = similarity_review(g, l, settings)
        pairs.append(
            {
                "external": spec["protein_id"],
                "legacy_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "locations": where,
                "global_alignment": g,
                "local_alignment": l,
                "review_flag": flag,
            }
        )
write_json(
    "data/cache/m18/source/reference-overlap.json",
    {"unique_historical_references": len(refs), "settings": settings, "pairs": pairs},
)
print(
    "historical refs",
    len(refs),
    "pairs",
    len(pairs),
    "flags",
    [(p["external"], p["locations"][0]) for p in pairs if p["review_flag"]],
)
