"""Infer a measured M12 benchmark variant with its held-out-family M15 checkpoint.

This bounded example requires the pinned caches/weights; it is not a universal
unknown-protein model or a thermodynamic stability predictor.
"""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch

from proteinmod.data import checksum
from proteinmod.lora_experiments import predict, prepare
from proteinmod.lora_model import AdaptedESM


def variant_index(rows, protein, mutations):
    requested = mutations.split(";")
    if len(requested) != len(set(requested)) or not all(requested):
        raise ValueError("Supply unique substitutions separated by semicolons")
    found = [
        i
        for i, row in enumerate(rows)
        if row["protein_id"] == protein and set(row["mutations"]) == set(requested)
    ]
    if len(found) != 1:
        raise ValueError("Variant is not in the validated benchmark; use a source-matched M12 row")
    return found[0]


def infer(protein, mutations):
    config = json.loads(Path("configs/m15.json").read_text())
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    with redirect_stdout(sys.stderr):
        encoder, rows, _, groups, _, partitions = prepare(config)
    index = variant_index(rows, protein, mutations)
    fold = next(
        f for protocol, f, roles in partitions if protocol == "G3" and index in roles["test"]
    )
    record = next(
        r
        for r in json.loads(Path("benchmarks/m15/metrics.json").read_text())
        if r["protocol"] == "G3"
        and r["fold"] == fold
        and r["fraction"] == 1
        and r["seed"] == 0
        and r["model"] == "lora"
    )
    if checksum(record["state_path"]) != record["state_sha256"]:
        raise ValueError("Supervised checkpoint checksum mismatch")
    model = AdaptedESM(encoder.model, config["lora"], config["training"]["hidden"], "lora", 0)
    model.load_selected(torch.load(record["state_path"], weights_only=True, map_location="cpu"))
    value = float(predict(model, groups, [index], config["training"]["batch_size"])[0])
    row = rows[index]
    return {
        "protein_id": protein,
        "assay_id": row["assay_id"],
        "variant_id": row["id"],
        "mutations": row["mutations"],
        "protocol": "G3",
        "fold": fold,
        "model": "M15 last-block LoRA, seed 0",
        "prediction": value,
        "unit": "training-assay rank prediction (unbounded regression), not kcal/mol",
        "task": "stability mutation-effect ranking",
        "checkpoint_sha256": record["state_sha256"],
        "test_family_supervised_training": False,
        "confidence_type": "not a model uncertainty",
        "interval": None,
        "interval_eligibility": False,
        "interval_reason": "M13 calibration is not valid for this separately fitted M15 model",
        "geometry": "reference structure only; this sequence model does not infer mutant rearrangement",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--protein", required=True)
    parser.add_argument("--mutations", required=True)
    args = parser.parse_args()
    print(json.dumps(infer(args.protein, args.mutations), indent=2, allow_nan=False))
