"""Replay M10 saved predictions, reliability gates and absent-structure fallbacks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from proteinmod.assay_models import evaluate
from proteinmod.confidence_fusion import ReliabilityFusion
from proteinmod.data import checksum, write_json
from proteinmod.reliability_experiments import history, inputs, load_features, mode_inputs


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    audit = json.loads((root / "audit.json").read_text())
    assert audit["history"] == history()
    for name, sha in manifest["files"].items():
        assert checksum(root / name) == sha, name
    _, rows, _, roles = inputs(config)
    features, feature_manifest = load_features(config, manifest["feature_root"])
    with (root / "predictions.csv").open() as handle:
        expected = list(csv.DictReader(handle))
    runs = json.loads((root / "runs.json").read_text())
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    count = 0
    for run in runs:
        protocol, fold, seed, mode = run["protocol"], run["fold"], run["seed"], run["mode"]
        role = next(r for p, f, r in roles if p == protocol and f == fold)
        path = Path(manifest["checkpoint_root"]) / f"{protocol}-{fold}-{mode}-{seed}.pt"
        assert checksum(path) == run["checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["feature_sha256"] == feature_manifest["feature_sha256"]
        model = ReliabilityFusion(hidden=state["hidden"])
        model.load_state_dict(state["state"])
        model.eval()
        h, q = mode_inputs(features, mode, seed)
        with torch.no_grad():
            pred = model(torch.tensor(h), torch.tensor(q), mode, True)
            missing = model(
                torch.tensor(features["h_missing"]), torch.zeros_like(torch.tensor(q)), mode, True
            )
        target = [
            r
            for r in expected
            if (r["protocol"], r["fold"], int(r["seed"]), r["mode"]) == (protocol, fold, seed, mode)
        ]
        assert [r["id"] for r in target] == [rows[i]["id"] for i in role["test"]]
        for column, values in [
            ("prediction", pred[0]),
            ("structural_gate", pred[1]),
            ("sequence_query_expert", pred[2]),
            ("structure_query_expert", pred[3]),
            ("missing_structure_prediction", missing[0]),
            ("missing_structure_gate", missing[1]),
        ]:
            np.testing.assert_array_equal(
                values[role["test"]].double().numpy(), np.array([float(r[column]) for r in target])
            )
        assert (missing[1] == 0).all()
        assert evaluate(rows, role["test"], pred[0][role["test"]].double().numpy()) == run["test"]
        assert (
            evaluate(rows, role["test"], missing[0][role["test"]].double().numpy())
            == run["missing_structure_test"]
        )
        count += len(target)
    result = {
        "saved_heads": len(runs),
        "test_rows": count,
        "replayed_numeric_columns": 6,
        "maximum_absolute_error": 0,
        "missing_structure_gate_exactly_zero": True,
        "all_metrics_recomputed": True,
        "matched_m9_role_ids": True,
        "historical_files_and_configs_preserved": len(history()),
        "m9_real_encoder_replay_exact": True,
    }
    write_json(root / "replay.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m10"))
    verify(parser.parse_args().input)
