"""Replay every M15 adapter/head/graph; validate provenance, roles and complete refits."""

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import checksum, write_json
from proteinmod.edge_experiments import make_model, ridge_predict, topology
from proteinmod.family_data import audit
from proteinmod.lora_data import history, subset
from proteinmod.lora_experiments import call, predict, prepare
from proteinmod.lora_model import AdaptedESM, site_representation, tensor_digest
from proteinmod.multiprotein_experiments import per_protein


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(str(row[k]) for k in ("fraction", "protocol", "fold", "model", "seed"))


def verify(root, repeat=None):
    config = json.loads(Path("configs/m15.json").read_text())
    registered = json.loads(Path("benchmarks/m15/preregistration.json").read_text())
    assert checksum("configs/m15.json") == registered["config_sha256"]
    original = subprocess.check_output(["git", "show", registered["commit"] + ":configs/m15.json"])
    assert hashlib.sha256(original).hexdigest() == registered["config_sha256"]
    assert checksum("benchmarks/m15/PREREGISTRATION.md") == registered["plan_sha256"]
    before = history()
    assert before == json.loads(Path("benchmarks/m15/historical_hashes.json").read_text())
    manifest = root / "artifact_hashes.json"
    if manifest.exists():
        for name, digest in json.loads(manifest.read_text()).items():
            assert checksum(root / name) == digest, name
    audit_record = json.loads((root / "audit.json").read_text())
    assert config == audit_record["config"] | {"output": config["output"]}
    assert before == audit_record["historical_sha256"]
    for path, digest in audit_record["sources"].items():
        assert checksum(path) == digest, path
    encoder, rows, matrices, groups, provenance, partitions = prepare(config)
    assert json.loads(json.dumps(rows)) == audit_record["rows"]
    assert provenance == {
        k: v for k, v in audit_record["provenance"].items() if k != "full_backbone_after"
    }
    assert provenance["full_backbone_before"] == audit_record["provenance"]["full_backbone_after"]
    roles_lookup, roles_evidence = {}, []
    for fraction in config["fractions"]:
        for protocol, fold, original in partitions:
            roles = subset(rows, original, fraction, config["subset_salt"])
            audit(rows, roles, protocol)
            assert set(roles["train"]) <= set(original["train"])
            assert all(np.array_equal(roles[k], original[k]) for k in ("validation", "test"))
            roles_lookup[(fraction, protocol, fold)] = roles
            roles_evidence.append(
                {
                    "fraction": fraction,
                    "protocol": protocol,
                    "fold": fold,
                    "original_roles": {
                        k: [rows[i]["id"] for i in ix] for k, ix in original.items()
                    },
                    "roles": {k: [rows[i]["id"] for i in ix] for k, ix in roles.items()},
                }
            )
    assert roles_evidence == audit_record["roles"]
    predictions = read_csv(root / "predictions.csv")
    lookup = defaultdict(list)
    for row in predictions:
        lookup[key(row)].append(row)
    records = json.loads((root / "metrics.json").read_text())
    expected = {
        (str(fraction), protocol, fold, model, str(seed))
        for fraction in config["fractions"]
        for protocol, fold, _ in partitions
        for model in config["models"]
        for seed in ([0] if model.endswith("_ridge") else config["seeds"])
    }
    assert len(records) == len(expected) and {key(r) for r in records} == expected
    max_error = 0.0
    for r in records:
        roles = roles_lookup[(r["fraction"], r["protocol"], r["fold"])]
        saved = lookup.pop(key(r))
        assert [p["id"] for p in saved] == [rows[i]["id"] for i in roles["test"]]
        assert [float(p["target"]) for p in saved] == [rows[i]["target"] for i in roles["test"]]
        assert checksum(r["state_path"]) == r["state_sha256"]
        if r["model"].endswith("_ridge"):
            state = dict(np.load(r["state_path"]))
            matrix = matrices["esm_sites" if r["model"] == "frozen_ridge" else "structural"]
            pred = ridge_predict(matrix, state)[0, roles["test"]]
        elif r["model"] in ("original_gnn", "edge_full"):
            model = make_model(
                next(iter(groups.values())), r["model"], config["structural_training"]
            ).eval()
            model.load_state_dict(torch.load(r["state_path"], weights_only=True))
            found = {}
            with torch.no_grad():
                for g in groups.values():
                    local = np.flatnonzero(np.isin(g["indices"], roles["test"]))
                    for start in range(0, len(local), config["structural_training"]["batch_size"]):
                        ix = local[start : start + config["structural_training"]["batch_size"]]
                        v = (
                            model(torch.tensor(g["dynamic"][ix]), topology(g, r["model"], 314159))
                            .numpy()
                            .astype(float)
                        )
                        v = v * r["selection"]["target_scale"] + r["selection"]["target_mean"]
                        found.update(zip(g["indices"][ix], v, strict=True))
            pred = np.array([found[i] for i in roles["test"]])
        else:
            model = AdaptedESM(
                encoder.model, config["lora"], config["training"]["hidden"], r["model"], r["seed"]
            )
            state = torch.load(r["state_path"], weights_only=True)
            model.load_selected(state)
            assert tensor_digest(state) == r["selection"]["state_tensor_sha256"]
            for candidate in r["selection"]["candidates"]:
                assert candidate["frozen_before"] == candidate["frozen_after"]
                assert candidate["finite_gradients"]
                if r["model"] != "frozen_head":
                    assert candidate["nonzero_adapter_gradients"]
            expected_count = 15409 if r["model"] == "frozen_head" else 19249
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_count
            pred = predict(model, groups, roles["test"], config["training"]["batch_size"])
        error = float(abs(pred - np.array([float(p["prediction"]) for p in saved])).max())
        max_error = max(max_error, error)
        assert error == 0, (key(r), error)
        assert per_protein(rows, roles["test"], pred) == r["per_protein"]
    assert not lookup
    # Full-label classical/structural baselines must remain identical to historical fits.
    historical_matches = {}
    for path, models in [
        (
            "benchmarks/m12/predictions.csv",
            {"esm_sites_ridge": "frozen_ridge", "structural_ridge": "structural_ridge"},
        ),
        (
            "benchmarks/m14/predictions.csv",
            {"original_gnn": "original_gnn", "edge_full": "edge_full"},
        ),
    ]:
        old = {}
        for p in read_csv(path):
            if p["model"] not in models or (
                "cohort" in p and p["cohort"] not in ("sequence_all", "structure_matched", "family")
            ):
                continue
            model = models[p["model"]]
            k = (p["protocol"], p["fold"], model, p["seed"], p["id"])
            old[k] = float(p["prediction"])
        new = {
            (p["protocol"], p["fold"], p["model"], p["seed"], p["id"]): float(p["prediction"])
            for p in predictions
            if float(p["fraction"]) == 1 and p["model"] in set(models.values())
        }
        assert old == new, path
        historical_matches[path] = len(old)
    # Exhaustive mutation metadata order check on actual cached inputs.
    original_config = json.loads(Path(config["m12_config"]["path"]).read_text())
    from proteinmod.lora_data import prefix_cache

    for spec in original_config["proteins"]:
        g = groups[spec["protein_id"]]
        part = [rows[i] for i in g["indices"]]
        reversed_rows = [r | {"mutations": list(reversed(r["mutations"]))} for r in part]
        _, masks, _, _ = prefix_cache(encoder, reversed_rows, spec["reference"]["sequence"], config)
        np.testing.assert_array_equal(masks, g["mutation_mask"])
    # Independently bypass the activation cache with a selected real G3 adapter on each protein.
    raw_model = copy.deepcopy(encoder.model).eval()
    raw_error = 0.0
    for pid, group in groups.items():
        record = next(
            r
            for r in records
            if r["model"] == "lora"
            and r["seed"] == 0
            and r["fraction"] == 1
            and r["protocol"] == "G3"
            and pid in r["per_protein"]
        )
        model = AdaptedESM(encoder.model, config["lora"], config["training"]["hidden"], "lora", 0)
        model.load_selected(torch.load(record["state_path"], weights_only=True))
        raw_model.encoder.layer[-1] = model.tail
        raw_model.encoder.emb_layer_norm_after = model.norm
        spec = next(s for s in original_config["proteins"] if s["protein_id"] == pid)
        tokens = encoder.tokenizer(
            [rows[group["indices"][0]]["sequence"], spec["reference"]["sequence"]],
            return_tensors="pt",
            truncation=False,
            padding=True,
        )
        with torch.no_grad():
            h = raw_model(**tokens).last_hidden_state[:, 1:-1]
            features = site_representation(h[:1], h[-1:], torch.tensor(group["mutation_mask"][:1]))
            difference = float(abs(model.head(features) - call(model, group, np.array([0]))).max())
        raw_error = max(raw_error, difference)
    assert raw_error < 1e-5, raw_error
    assert (
        tensor_digest(dict(encoder.model.named_parameters())) == provenance["full_backbone_before"]
    )
    repeat_evidence = None
    if repeat:
        repeated = read_csv(repeat / "predictions.csv")
        assert predictions == repeated
        again = json.loads((repeat / "metrics.json").read_text())
        assert len(again) == len(records)
        audit2 = json.loads((repeat / "audit.json").read_text())
        assert audit2["roles"] == audit_record["roles"]
        assert audit2["sources"] == audit_record["sources"]
        for a, b in zip(records, again, strict=True):
            for name in (
                "fraction",
                "protocol",
                "fold",
                "model",
                "seed",
                "selection",
                "per_protein",
            ):
                assert a[name] == b[name], name
            if a["model"].endswith("_ridge"):
                x, y = dict(np.load(a["state_path"])), dict(np.load(b["state_path"]))
                for name, value in x.items():
                    np.testing.assert_array_equal(value, y[name])
            else:
                x = torch.load(a["state_path"], weights_only=True)
                y = torch.load(b["state_path"], weights_only=True)
                assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
        repeat_evidence = {
            "fits": len(records),
            "predictions": len(predictions),
            "all_state_values_identical": True,
            "all_selection_traces_identical": True,
            "max_prediction_error": 0.0,
            "seconds": audit2["seconds"],
            "peak_rss_bytes": audit2["peak_rss_bytes"],
        }
    result = {
        "fits_replayed": len(records),
        "predictions_replayed": len(predictions),
        "max_error": max_error,
        "historical_files_unchanged": len(before),
        "historical_baseline_predictions_exact": historical_matches,
        "full_original_backbone_unchanged": True,
        "raw_full_adapted_forward_max_error": raw_error,
        "original_roles_preserved": True,
        "mutation_order_variants": len(rows),
        "full_prefix_parity_max_error": max(
            v["manifest"]["full_forward_max_error"] for v in provenance["prefixes"].values()
        ),
        "repeat": repeat_evidence,
    }
    assert history() == before
    if not manifest.exists() or repeat:
        write_json(root / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmarks/m15"))
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    verify(args.directory, args.repeat)
