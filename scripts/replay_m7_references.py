"""Replay fixed M5 bidirectional and M6 stability heads without changing their artifacts."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import write_json
from proteinmod.embeddings import fingerprint
from proteinmod.hybrid import HybridRegressor
from proteinmod.hybrid_experiments import inputs
from proteinmod.multiprotein_data import protocols
from proteinmod.multitask_experiments import history
from proteinmod.stability_data import stability_folds
from proteinmod.stability_features import model_inputs


def replay():
    before = history()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    m5 = json.loads(Path("configs/m5.json").read_text())
    m4 = json.loads(Path(m5["m4_config"]).read_text())
    rows, mask, groups, _, _ = inputs(m5, m4)
    with Path("benchmarks/m5/predictions.csv").open() as handle:
        archived = list(csv.DictReader(handle))
    maximum, count = 0.0, 0
    for protocol, fold, roles in protocols(rows, m4["experiment"]["fold_order"]):
        indices = roles["test"][mask[roles["test"]]]
        for seed in m5["seeds"]:
            first = next(iter(groups.values()))["graph"]
            model = HybridRegressor(first, 480, 16, 2, 2, "bidirectional")
            path = (
                Path("data/cache/m5/checkpoints")
                / fingerprint(m5)
                / f"{protocol}-{fold}-bidirectional-{seed}.pt"
            )
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            model.eval()
            predictions = {}
            with torch.no_grad():
                for group in groups.values():
                    g = group["graph"]
                    edge = g["edge_index"]
                    topology = (
                        torch.as_tensor(g["static"]),
                        torch.as_tensor(edge[0]),
                        torch.as_tensor(edge[1]),
                        torch.as_tensor(g["edge_features"]),
                        torch.tensor(
                            np.maximum(np.bincount(edge[1], minlength=len(g["static"])), 1),
                            dtype=torch.float32,
                        )[None, :, None],
                    )
                    positions = torch.tensor([n["reference_position"] - 1 for n in g["nodes"]])
                    local = np.flatnonzero(np.isin(group["indices"], indices))
                    for start in range(0, len(local), 32):
                        ix = local[start : start + 32]
                        values = model(
                            torch.from_numpy(
                                np.array(group["residues"][group["token_indices"][ix]], copy=True)
                            ),
                            torch.as_tensor(group["dynamic"][ix]),
                            topology,
                            positions,
                        )
                        predictions.update(zip(group["indices"][ix], values.numpy(), strict=True))
            expected = {
                r["id"]: float(r["prediction"])
                for r in archived
                if r["protocol"] == protocol
                and r["fold"] == fold
                and r["model"] == "bidirectional"
                and int(r["seed"]) == seed
            }
            assert set(expected) == {rows[i]["id"] for i in indices}
            maximum = max(
                maximum, max(abs(predictions[i] - expected[rows[i]["id"]]) for i in indices)
            )
            count += len(indices)
    m5_result = {
        "predictions": count,
        "max_absolute_difference": float(maximum),
        "tolerance": 1e-6,
        "passed": maximum <= 1e-6,
    }
    m6 = json.loads(Path("configs/m6.json").read_text())
    rows, features, _ = model_inputs(m6)
    with Path("benchmarks/m6/predictions.csv").open() as handle:
        archived = list(csv.DictReader(handle))
    maximum, count = 0.0, 0
    for protocol, fold, roles in stability_folds(rows, m6):
        for seed in m6["seeds"]:
            path = Path("data/cache/m6/heads") / fingerprint(m6) / f"{protocol}-{fold}-{seed}.pt"
            state = torch.load(path, weights_only=True, map_location="cpu")
            head = torch.nn.Sequential(
                torch.nn.Linear(100, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
            )
            head.load_state_dict(state["stability_head"])
            with torch.no_grad():
                prediction = (
                    head(
                        torch.tensor(
                            features["frozen_hybrid_ridge"][roles["test"]], dtype=torch.float32
                        )
                    )
                    .squeeze(-1)
                    .double()
                    .numpy()
                )
            prediction = prediction * state["target_scale"] + state["target_mean"]
            expected = {
                r["id"]: float(r["prediction_ddg_kcal_mol"])
                for r in archived
                if r["protocol"] == protocol
                and r["fold"] == fold
                and r["model"] == "stability_head"
                and int(r["seed"]) == seed
            }
            assert set(expected) == {rows[i]["id"] for i in roles["test"]}
            maximum = max(
                maximum,
                max(
                    abs(p - expected[rows[i]["id"]])
                    for i, p in zip(roles["test"], prediction, strict=True)
                ),
            )
            count += len(prediction)
    assert history() == before
    result = {
        "m5": m5_result,
        "m6": {
            "predictions": count,
            "max_absolute_difference_kcal_mol": float(maximum),
            "tolerance": 1e-5,
            "passed": maximum <= 1e-5,
        },
        "historical_files_unchanged": len(before),
    }
    assert result["m5"]["passed"] and result["m6"]["passed"]
    Path("runs/m7").mkdir(parents=True, exist_ok=True)
    write_json("runs/m7/historical_replay.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    replay()
