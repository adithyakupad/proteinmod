"""Replay every M11 model and every model-implied measured factorial contrast."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import checksum, write_json
from proteinmod.epistasis_data import protocols
from proteinmod.epistasis_experiments import (
    PairResidual,
    contrast_predictions,
    epsilon_metrics,
    historical,
    load,
    neural_features,
)
from proteinmod.models import metrics
from proteinmod.mutation_interactions import PAIR_WIDTH, SEQUENCE_WIDTH


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    config = manifest["config"]
    audit = json.loads((root / "audit.json").read_text())
    assert audit["history"] == historical()
    for name, sha in manifest["files"].items():
        assert checksum(root / name) == sha, name
    rows = json.loads((root / "rows.json").read_text())
    contrasts = json.loads((root / "contrasts.json").read_text())
    x, _ = load(config, rows)
    with (root / "predictions.csv").open() as handle:
        expected = list(csv.DictReader(handle))
    with (root / "epistasis_predictions.csv").open() as handle:
        eps_expected = list(csv.DictReader(handle))
    runs = json.loads((root / "runs.json").read_text())
    roles = dict(protocols(rows, config))
    heads = count = eps_count = 0
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    for run in runs:
        protocol, name, seed = run["protocol"], run["model"], run["seed"]
        role = roles[protocol]
        path = Path(manifest["checkpoint_root"]) / f"{protocol}-{name}-{seed}.pt"
        assert checksum(path) == run["checkpoint_sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["kind"] == "fixed":
            pred = np.array(saved["prediction"])
        elif saved["kind"] == "ridge":
            if name == "mutation_count":
                matrix = np.array([[len(r["mutations"])] for r in rows])
            elif name in ("sequence_pair_ridge", "structural_pair_ridge"):
                matrix = x["pairs_real"][
                    ..., : SEQUENCE_WIDTH if name == "sequence_pair_ridge" else PAIR_WIDTH
                ].sum(1)
            else:
                matrix = x[name.removesuffix("_ridge")]
            state = saved["state"]
            z = (matrix.astype(np.float64) - state["mean"]) / state["scale"]
            pred = np.array(state["residual"]) + (
                z @ np.array(state["coefficient"]) + state["intercept"]
            )
        else:
            state = saved["state"]
            pairs = neural_features(x, name, seed)
            z = torch.tensor((pairs - state["mean"]) / state["scale"], dtype=torch.float32)
            model = PairResidual(state["width"], state["hidden"])
            model.load_state_dict(state["state"])
            model.eval()
            with torch.no_grad():
                pred = (
                    np.array(state["base"])
                    + model(z, torch.tensor(x["pair_mask"])).double().numpy()
                    * state["target_scale"]
                )
            heads += 1
        found = [
            r
            for r in expected
            if (r["protocol"], r["model"], int(r["seed"])) == (protocol, name, seed)
        ]
        assert [r["id"] for r in found] == [rows[i]["id"] for i in role["test"]]
        np.testing.assert_array_equal(
            pred[role["test"]], np.array([float(r["predicted_log_fitness"]) for r in found])
        )
        assert metrics([rows[i]["target"] for i in role["test"]], pred[role["test"]]) == run["test"]
        contrasts_found = contrast_predictions(rows, contrasts, pred, role["test"])
        target = [
            r
            for r in eps_expected
            if (r["protocol"], r["model"], int(r["seed"])) == (protocol, name, seed)
        ]
        assert len(target) == len(contrasts_found)
        for a, b in zip(contrasts_found, target, strict=True):
            for key in ("target", "a", "b", "background", "position_pair"):
                assert a[key] == b[key]
            for key in (
                "measured_log_epistasis",
                "predicted_log_epistasis",
                "measured_linear_epistasis",
            ):
                assert a[key] == float(b[key])
        assert (
            epsilon_metrics([r for r in contrasts_found if not r["background_mutations"]])
            == run["epistasis_WT"]
        )
        assert (
            epsilon_metrics([r for r in contrasts_found if r["background_mutations"]])
            == run["epistasis_nonWT"]
        )
        count += len(found)
        eps_count += len(target)
    result = {
        "models": len(runs),
        "neural_heads": heads,
        "held_out_predictions": count,
        "factorial_predictions": eps_count,
        "maximum_absolute_replay_error": 0,
        "model_implied_contrasts_recomputed": True,
        "matched_test_ids": True,
        "historical_files_and_configs": len(historical()),
        "probe_only_labels_fitted": False,
    }
    write_json(root / "replay.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m11"))
    verify(parser.parse_args().input)
