"""Reload every saved M8 neural head and replay all its held-out predictions."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from proteinmod.data import write_json
from proteinmod.embeddings import fingerprint
from proteinmod.stability_conditioning import ConditionedHead, controls, scalar_inputs

root = Path("runs/m8")
manifest = json.loads((root / "manifest.json").read_text())
audit = json.loads((root / "audit.json").read_text())
config = manifest["config"]
checkpoint_root = (
    Path(config["cache"])
    / "heads"
    / fingerprint(
        {
            "config": config,
            "historical": audit["historical_sha256"],
            "source": manifest["source_sha256"],
        }
    )
)
rows = json.loads(Path("benchmarks/m7/rows.json").read_text())["function"]
lookup = {r["id"]: i for i, r in enumerate(rows)}
with (root / "predicted_stability.csv").open() as handle:
    raw = list(csv.DictReader(handle))
stability = {}
for r in raw:
    key = r["fold"], r["source"]
    stability.setdefault(key, np.full(len(rows), np.nan))[lookup[r["id"]]] = float(
        r["predicted_ddg_kcal_mol"]
    )
with (root / "predictions.csv").open() as handle:
    expected = [r for r in csv.DictReader(handle) if r["model"] == "neural_head"]
baselines = json.loads((root / "baseline_replay.json").read_text())
torch.set_num_threads(config["head"]["cpu_threads"])
torch.use_deterministic_algorithms(True)
count, maximum, heads = 0, 0.0, 0
for baseline in baselines:
    fold, seed = baseline["fold"], baseline["seed"]
    role = next(r for r in audit["roles"] if r["function_fold"] == fold)["ids"]["function"]
    roles = {p: np.array([lookup[i] for i in ids]) for p, ids in role.items()}
    eligible = np.unique(np.concatenate(list(roles.values())))
    with np.load(baseline["provenance"]["path"], allow_pickle=False) as data:
        h = data["representation"]
    variants = [("none", "none")] + [
        (s, c) for s in config["stability_sources"] for c in config["controls"]
    ]
    for source, control in variants:
        filename = (
            f"{fold}-{seed}-independent.pt"
            if source == "none"
            else f"{fold}-{seed}-{source}-{control}.pt"
        )
        state = torch.load(checkpoint_root / filename, weights_only=True)
        assert (
            state["stability_input_unit"] == "kcal/mol"
            and state["stability_positive_means"] == "destabilization"
        )
        if source == "none":
            scalar = np.column_stack((np.zeros(len(rows)), np.ones(len(rows))))
        else:
            values = controls(stability[fold, source], rows, roles, seed)[control]
            scalar, calibration = scalar_inputs(values, roles["train"])
            assert calibration == state["scalar_calibration"]
            assert (
                state["function_encoder_sha256"]
                == baseline["provenance"]["identity"]["checkpoint_sha256"]
            )
        scaler = StandardScaler()
        scaler.mean_ = np.array(state["representation_mean"])
        scaler.scale_ = np.array(state["representation_scale"])
        scaler.n_features_in_ = h.shape[1]
        x = torch.tensor(scaler.transform(h), dtype=torch.float32)
        model = ConditionedHead(h.shape[1], config["head"]["hidden"])
        model.load_state_dict(state["head"])
        model.eval()
        with torch.no_grad():
            prediction = (
                model(x[eligible], torch.tensor(scalar[eligible], dtype=torch.float32))
                .double()
                .numpy()
            )
        prediction = prediction * state["target_scale"] + state["target_mean"]
        found = dict(zip(eligible, prediction, strict=True))
        targets = [
            r
            for r in expected
            if r["fold"] == fold
            and int(r["seed"]) == seed
            and r["source"] == source
            and r["control"] == control
        ]
        delta = max(abs(found[lookup[r["id"]]] - float(r["prediction"])) for r in targets)
        assert delta <= 1e-6, (fold, seed, source, control, delta)
        maximum = max(maximum, delta)
        count += len(targets)
        heads += 1
write_json(
    root / "checkpoint_replay.json",
    {
        "heads": heads,
        "predictions": count,
        "max_absolute_difference": maximum,
        "tolerance": 1e-6,
        "units_and_calibration_verified": True,
    },
)
print(heads, "heads;", count, "predictions; maximum difference", maximum)
