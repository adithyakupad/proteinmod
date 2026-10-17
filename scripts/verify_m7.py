"""Repeat the prespecified first held-out equal-joint fit and verify cached tensors."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import write_json
from proteinmod.embeddings import fingerprint
from proteinmod.multitask_experiments import fit, inputs

c = json.loads(Path("configs/m7.json").read_text())
rows, groups, folds, _ = inputs(c)
f, s, roles = next(x for x in folds if x[0] == "amylase")
pred, selection, state = fit(rows, groups, roles, c["training"], 0, "joint", 1.0)
with open("runs/m7/predictions.csv") as handle:
    old = list(csv.DictReader(handle))
result = {}
for task in pred:
    by_id = {
        r["id"]: float(r["prediction"])
        for r in old
        if r["task"] == task
        and r["function_fold"] == f
        and r["model"] == "joint"
        and float(r["stability_weight"]) == 1
        and r["seed"] == "0"
    }
    expected = np.array([by_id[rows[task][i]["id"]] for i in roles[task]["test"]])
    result[task] = {
        "predictions": len(expected),
        "max_absolute_difference": float(np.max(np.abs(pred[task] - expected))),
    }
    np.testing.assert_array_equal(pred[task], expected)
runs = json.loads(Path("runs/m7/runs.json").read_text())
prior = next(
    r
    for r in runs
    if r["function_fold"] == f
    and r["seed"] == 0
    and r["mode"] == "joint"
    and r["stability_weight"] == 1
)
for key in ["epoch", "score", "trace", "calibration", "examples_processed", "gradient"]:
    assert selection[key] == prior["selection"][key], key
# Check serialized tensors and both calibrations independently of Python object identity.
manifest = json.loads(Path("runs/m7/manifest.json").read_text())
audit = json.loads(Path("runs/m7/audit.json").read_text())
identity = {"config": c, "historical_data": audit["history"], "sources": manifest["source_hashes"]}
checkpoint = Path("data/cache/m7/checkpoints") / fingerprint(identity) / f"{f}-{s}-joint-1.0-0.pt"
loaded = torch.load(checkpoint, weights_only=True)
assert all(torch.equal(loaded["state"][k], v) for k, v in state.items())
assert loaded["calibration"] == selection["calibration"]
write_json(
    "runs/m7/reproducibility.json",
    {
        "fold_pair": [f, s],
        "seed": 0,
        "mode": "equal_joint",
        "exact_prediction_replay": result,
        "selection_and_training_trace_exact": True,
        "serialized_state_exact": True,
    },
)
print(result)
