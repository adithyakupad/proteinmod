"""Replay all M9 saved heads and verify matched observations/historical hashes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from proteinmod.assay_experiments import history
from proteinmod.assay_features import load
from proteinmod.assay_metadata import AssayHead, AssayMetadata, MetadataVocabulary
from proteinmod.data import checksum, write_json


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    config = manifest["config"]
    assert history() == audit["historical_sha256"]
    for name, expected in manifest["files"].items():
        assert checksum(root / name) == expected, name
    features, feature_manifest = load(config)
    rows = json.loads((root / "rows.json").read_text())
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    metadata = {
        a["metadata"]["assay_id"]: AssayMetadata.from_dict(a["metadata"]) for a in config["assays"]
    }
    with (root / "predictions.csv").open() as handle:
        expected = list(csv.DictReader(handle))
    runs = json.loads((root / "runs.json").read_text())
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    heads, count, maximum = set(), 0, 0.0
    for run in runs:
        mode = run["model"]
        if mode.endswith(("ridge", "heads")):
            continue
        saved_mode = "structured" if mode == "structured_missing_at_test" else mode
        path = (
            Path(manifest["checkpoint_root"])
            / f"{run['protocol']}-{run['fold']}-{saved_mode}-{run['seed']}.pt"
        )
        assert checksum(path) == run["selection"]["checkpoint_sha256"]
        state = torch.load(path, weights_only=True, map_location="cpu")
        assert state["feature_sha256"] == feature_manifest["feature_sha256"]
        vocab = MetadataVocabulary(state["vocabulary"])
        c, n = vocab.transform(
            [metadata[r["assay_id"]] for r in rows], [r["protein_id"] for r in rows]
        )
        if mode == "structured_missing_at_test":
            c, n = np.zeros_like(c), np.zeros_like(n)
        scaler = StandardScaler()
        scaler.mean_ = np.array(state["representation_mean"])
        scaler.scale_ = np.array(state["representation_scale"])
        scaler.n_features_in_ = features["multimodal"].shape[1]
        x = torch.tensor(scaler.transform(features["multimodal"]), dtype=torch.float32)
        head = AssayHead(x.shape[1], vocab, state["hidden"])
        head.load_state_dict(state["head"])
        head.eval()
        with torch.no_grad():
            p = head(x, torch.tensor(c), torch.tensor(n), saved_mode).double().numpy()
        targets = [
            r
            for r in expected
            if r["protocol"] == run["protocol"]
            and r["fold"] == run["fold"]
            and r["model"] == mode
            and int(r["seed"]) == run["seed"]
        ]
        delta = max(
            abs(p[lookup[r["id"]]] - float(r["predicted_train_percentile"])) for r in targets
        )
        assert delta == 0, (path, delta)
        maximum = max(maximum, delta)
        heads.add(str(path))
        count += len(targets)
    for role in audit["roles"]:
        sets = []
        for run in runs:
            if run["protocol"] == role["protocol"] and run["fold"] == role["fold"]:
                ids = {
                    r["id"]
                    for r in expected
                    if r["protocol"] == run["protocol"]
                    and r["fold"] == run["fold"]
                    and r["model"] == run["model"]
                    and int(r["seed"]) == run["seed"]
                }
                assert ids == set(role["ids"]["test"])
                sets.append(ids)
        assert len(sets) == 23
    result = {
        "saved_heads": len(heads),
        "neural_test_predictions": count,
        "max_absolute_replay_error": maximum,
        "matched_test_ids_all_92_cases": True,
        "historical_files_unchanged": len(history()),
        "stability_scalar_inputs": False,
    }
    write_json(root / "replay.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/m9"))
    verify(parser.parse_args().input)
