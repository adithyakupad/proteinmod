"""Commit exact outcome-free roles before fitting any M17 model."""

import json
from pathlib import Path

from proteinmod.data import checksum, write_json
from proteinmod.mechanism import balanced, study_roles

rows = json.loads(Path("benchmarks/m12/rows.json").read_text())
config = json.loads(Path("configs/m17.json").read_text())
families = json.loads(Path("configs/m12.json").read_text())["family_order"]
roles = []
for study, fold, protocol, rr in study_roles(rows, families):
    for seed in config["seeds"] if study == "H1" else [0]:
        for fraction in config["fractions"] if study == "H1" else [1.0]:
            selected = {**rr, "train": balanced(rows, rr["train"], fraction, seed)}
            roles.append(
                dict(
                    study=study,
                    fold=fold,
                    protocol=protocol,
                    seed=seed,
                    fraction=fraction,
                    roles={k: [rows[i]["id"] for i in ix] for k, ix in selected.items()},
                )
            )
path = Path("benchmarks/m17/split-manifest.json")
assert not path.exists()
write_json(
    path,
    {
        "config_sha256": checksum("configs/m17.json"),
        "row_source_sha256": checksum("benchmarks/m12/rows.json"),
        "folds": roles,
    },
)
print(len(roles), "exact controlled roles, outcomes omitted")
