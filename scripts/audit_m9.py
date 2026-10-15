"""Describe M9 feature shift, bounded-score saturation and generalization gaps."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from proteinmod.assay_data import collection, folds
from proteinmod.assay_features import load
from proteinmod.assay_models import scale_fit
from proteinmod.data import write_json
from proteinmod.structure_pipeline import write_csv


def audit(root):
    config = json.loads((root / "manifest.json").read_text())["config"]
    rows, _ = collection(config)
    features, _ = load(config)
    shifts = []
    for protocol, fold, roles in folds(rows, config["protein_order"]):
        for name, x in features.items():
            train, test = x[roles["train"]], x[roles["test"]]
            scaler = scale_fit(x, roles["train"], config["training"]["scale_floor"])
            z = scaler.transform(test)
            shifts.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "features": name,
                    "dimensions": x.shape[1],
                    "guarded_dimensions": int(
                        (np.sqrt(scaler.var_) < config["training"]["scale_floor"]).sum()
                    ),
                    "test_fraction_outside_training_range": float(
                        ((test < train.min(0)) | (test > train.max(0))).mean()
                    ),
                    "test_fraction_abs_standardized_above_10": float((np.abs(z) > 10).mean()),
                    "max_abs_standardized_test": float(np.abs(z).max()),
                }
            )
    with (root / "predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    runs = json.loads((root / "runs.json").read_text())
    saturation, gaps = [], []
    for run in runs:
        local = [
            r
            for r in predictions
            if r["protocol"] == run["protocol"]
            and r["fold"] == run["fold"]
            and r["model"] == run["model"]
            and int(r["seed"]) == run["seed"]
        ]
        p = np.array([float(r["predicted_train_percentile"]) for r in local])
        saturation.append(
            {
                "protocol": run["protocol"],
                "fold": run["fold"],
                "model": run["model"],
                "seed": run["seed"],
                "minimum": float(p.min()),
                "maximum": float(p.max()),
                "unique_predictions": len(np.unique(p)),
                "boundary_fraction": float(((p == 0) | (p == 1)).mean()),
                "near_boundary_fraction": float(((p <= 1e-6) | (p >= 1 - 1e-6)).mean()),
            }
        )
        if "train_macro" in run["selection"]:
            gaps.append(
                {
                    "protocol": run["protocol"],
                    "fold": run["fold"],
                    "model": run["model"],
                    "seed": run["seed"],
                    "train_macro": run["selection"]["train_macro"],
                    "validation_macro": run["selection"]["validation_macro"],
                    "test_macro": run["test"]["macro"],
                    "selected_epoch": run["selection"]["epoch"],
                }
            )
    write_json(root / "feature_shift.json", shifts)
    write_csv(root / "saturation.csv", saturation)
    write_csv(root / "generalization_gaps.csv", gaps)
    return shifts, saturation, gaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m9"))
    audit(parser.parse_args().input)
