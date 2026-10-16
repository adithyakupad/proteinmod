"""Repeat the first G3 fold's three neural fits and require exact saved predictions."""

import csv
import json
from pathlib import Path

import numpy as np

from proteinmod.data import write_json
from proteinmod.family_data import folds
from proteinmod.family_experiments import inputs
from proteinmod.hybrid_experiments import fit

config = json.loads(Path("configs/m12.json").read_text())
root = Path("benchmarks/m12")
rows, _, mask, groups, _ = inputs(config)
protocol, fold, original = next(r for r in folds(rows, config["family_order"]) if r[0] == "G3")
roles = {role: ix[mask[ix]] for role, ix in original.items()}
with (root / "predictions.csv").open() as f:
    predictions = list(csv.DictReader(f))
records = json.loads((root / "runs.json").read_text())
results = []
seed = config["seeds"][0]
for mode in config["modes"]:
    saved = [
        p
        for p in predictions
        if (p["protocol"], p["fold"], p["cohort"], p["model"], int(p["seed"]))
        == (protocol, fold, "structure_matched", mode, seed)
    ]
    pred, selection, _ = fit(groups, rows, roles, config["training"], seed, mode)
    np.testing.assert_array_equal(pred, [float(p["prediction"]) for p in saved])
    old = next(
        r["selection"]
        for r in records
        if (r["protocol"], r["fold"], r["cohort"], r["model"], r["seed"])
        == (protocol, fold, "structure_matched", mode, seed)
    )
    assert all(selection[k] == old[k] for k in selection)
    results.append(
        {
            "protocol": protocol,
            "fold": fold,
            "model": mode,
            "seed": seed,
            "exact_predictions": len(pred),
            "selected_epoch": selection["epoch"],
            "trace_identical": True,
        }
    )
write_json(root / "replay.json", results)
print(results)
