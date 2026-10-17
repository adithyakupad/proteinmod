"""Replay M13 saved models, native heads, calibration and every interval/metric offline."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from proteinmod.data import checksum, verify_file, write_json
from proteinmod.family_experiments import inputs
from proteinmod.structure_pipeline import write_csv
from proteinmod.uncertainty import shuffled, training_distance
from proteinmod.uncertainty_calibration import calibration_state, prediction_intervals
from proteinmod.uncertainty_data import historical, native_labels, roles_audit, uq_folds
from proteinmod.uncertainty_experiments import m6_inputs
from proteinmod.uncertainty_models import native_affine, predict_hybrid, ridge_predict
from proteinmod.uncertainty_report import analyze


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            close(x, y)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    else:
        assert a == b, (a, b)


def verify(root):
    import torch

    config = json.loads(Path("configs/m13.json").read_text())
    prereg = json.loads((root / "preregistration.json").read_text())
    assert (
        checksum("configs/m13.json") == prereg["config_sha256"]
        and checksum(root / "PREREGISTRATION.md") == prereg["plan_sha256"]
    )
    if (root / "artifact_hashes.json").exists():
        for name, digest in json.loads((root / "artifact_hashes.json").read_text()).items():
            assert checksum(root / name) == digest, name
    audit = json.loads((root / "audit.json").read_text())
    assert historical() == audit["historical_sha256"]
    for name in ("m12_config", "m6_config"):
        verify_file(config[name]["path"], config[name])
    m12 = json.loads(Path(config["m12_config"]["path"]).read_text())
    m6 = json.loads(Path(config["m6_config"]["path"]).read_text())
    torch.set_num_threads(m12["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    rows, features, mask, groups, _ = inputs(m12)
    assert mask.all()
    native, provenance = native_labels(rows, m12, m6)
    assert provenance == json.loads((root / "native_labels.json").read_text())
    rows6, features6, plans6, _ = m6_inputs(m6, config)
    y6 = np.array([r["target"] for r in rows6])
    datasets = {
        "M12": (rows, features, native, uq_folds(rows, native, m12, config)),
        "M6": (rows6, features6, y6, plans6),
    }
    plans = {}
    distances = {}
    for dataset, (part, matrices, y, folds) in datasets.items():
        assert json.loads(json.dumps(part)) == json.loads(
            (root / (dataset.lower() + "_rows.json")).read_text()
        )
        old = json.loads(Path("benchmarks/" + dataset.lower() + "/audit.json").read_text())["roles"]
        for protocol, fold, roles in folds:
            plans[(dataset, protocol, fold)] = roles
            saved = next(
                r
                for r in audit["roles"]
                if (r["dataset"], r["protocol"], r["fold"]) == (dataset, protocol, fold)
            )
            assert roles_audit(part, roles, protocol) == saved["roles"]
            historical_role = next(
                r
                for r in old
                if r["protocol"] == protocol
                and r["fold"] == fold
                and (dataset == "M6" or r["cohort"] == "structure_matched")
            )
            old_ids = (
                historical_role["ids"]["test"]
                if dataset == "M6"
                else historical_role["roles"]["test"]["ids"]
            )
            assert [part[i]["id"] for i in roles["test"]] == old_ids
            distances[(dataset, protocol, fold)] = training_distance(
                matrices["esm_sites" if dataset == "M12" else "esm_sites_ridge"], roles["train"]
            )
    with (root / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    with (root / "calibration_predictions.csv").open() as f:
        calibration = list(csv.DictReader(f))

    def grouped(records):
        result = defaultdict(list)
        for r in records:
            result[
                tuple(str(r[k]) for k in ("dataset", "protocol", "fold", "model", "repeat"))
            ].append(r)
        return result

    grouped_test, grouped_cal = grouped(predictions), grouped(calibration)
    runs = json.loads((root / "runs.json").read_text())
    assert len(runs) == len(grouped_test) == 180
    max_error = 0.0
    interval_count = 0
    calibration_count = 0
    models = 0
    families = []
    for run in runs:
        dataset, protocol, fold = run["dataset"], run["protocol"], run["fold"]
        part, matrices, y, _ = datasets[dataset]
        roles = plans[(dataset, protocol, fold)]
        key = tuple(str(run[k]) for k in ("dataset", "protocol", "fold", "model", "repeat"))
        test = grouped_test[key]
        assert [r["id"] for r in test] == [part[i]["id"] for i in roles["test"]]
        for name, digest in run["files"].items():
            assert checksum(name) == digest, name
        if isinstance(run["selection"], dict):
            path = next(iter(run["files"]))
            state = dict(np.load(path, allow_pickle=False))
            name = run["model"].removesuffix("_ridge") if dataset == "M12" else run["model"]
            rank = ridge_predict(matrices[name], state)
            models += len(rank)
        else:
            members = [
                predict_hybrid(
                    groups,
                    torch.load(path, weights_only=True, map_location="cpu"),
                    m12["training"],
                    run["model"],
                    len(part),
                    roles,
                )
                for path in run["files"]
            ]
            rank = np.stack([members[0], *members])
            models += len(members)
        if dataset == "M12":
            native_predictions, head = native_affine(
                rank, y, roles["train"], config["native_affine_alpha"]
            )
            close(head, run["native_head"])
            np.testing.assert_allclose(
                np.array([json.loads(r["rank_members"]) for r in test]).T,
                rank[1:, roles["test"]],
                atol=0,
                rtol=0,
            )
        else:
            native_predictions = rank
            np.testing.assert_allclose(
                np.array([json.loads(r["native_members"]) for r in test]).T,
                rank[1:, roles["test"]],
                atol=0,
                rtol=0,
            )
        point = native_predictions[1:].mean(0)
        sd = native_predictions[1:].std(0, ddof=1)
        distance = distances[(dataset, protocol, fold)]
        shuffle = sd.copy()
        for ix in roles.values():
            group = [
                part[i]["protein_id"] + (":native" if np.isfinite(y[i]) else ":missing") for i in ix
            ]
            shuffle[ix] = shuffled(sd[ix], group, config["bootstrap_seed"] + run["repeat"])
            for pid in {part[i]["protein_id"] for i in ix}:
                eligible = [i for i in ix if part[i]["protein_id"] == pid and np.isfinite(y[i])]
                np.testing.assert_array_equal(np.sort(sd[eligible]), np.sort(shuffle[eligible]))
        state = calibration_state(y, point, sd, distance, shuffle, part, roles, config, protocol)
        close(state, run["calibration"])
        bounds = prediction_intervals(point, sd, distance, shuffle, part, state, config)
        if protocol == "G3":
            assert state["family_block_audit"]["independent_calibration_families"] == 1
            assert all(v is None for v in state["family_block_audit"]["quantiles"].values())
            families.append(fold)
        ids = {r["id"]: i for i, r in enumerate(part)}
        for saved in grouped_cal[key] + test:
            i = ids[saved["id"]]
            for name, actual in [
                ("native_point", point[i]),
                ("native_sd", sd[i]),
                ("native_sd_shuffled", shuffle[i]),
                ("distance_to_training", distance[i]),
            ]:
                err = abs(float(saved[name]) - actual)
                max_error = max(max_error, err)
                np.testing.assert_allclose(float(saved[name]), actual, atol=1e-12, rtol=1e-12)
            if saved["measured_native"] != "":
                assert float(saved["measured_native"]) == y[i]
            if saved["role"] != "test":
                calibration_count += 1
                continue
            assert float(saved["measured_original"]) == part[i]["target"]
            assert saved["native_eligible"] == str(bool(np.isfinite(y[i])))
            for name, interval in bounds.items():
                for side in ("lower", "upper"):
                    actual = interval[side][i] if np.isfinite(y[i]) else None
                    value = saved[name + "_" + side]
                    if actual is None:
                        assert value == ""
                    else:
                        np.testing.assert_allclose(float(value), actual, atol=1e-12, rtol=1e-12)
                        interval_count += 1
        if dataset == "M12":
            np.testing.assert_array_equal(
                [float(r["rank_point"]) for r in test], rank[1:].mean(0)[roles["test"]]
            )
    recomputed = analyze(predictions, config)
    with TemporaryDirectory() as temporary:
        for name, values in zip(
            ("per_protein.csv", "interval_metrics.csv", "selective_risk.csv"),
            recomputed,
            strict=True,
        ):
            p = Path(temporary) / name
            write_csv(p, values)
            assert checksum(p) == checksum(root / name), name
    return {
        "verified": True,
        "uncertainty_evaluations": len(runs),
        "serialized_point_models_replayed": models,
        "held_out_predictions": len(predictions),
        "calibration_validation_predictions_verified": calibration_count,
        "finite_interval_endpoints_verified": interval_count,
        "max_absolute_native_prediction_error": float(max_error),
        "g3_test_families": sorted(set(families)),
        "independent_g3_calibration_families_per_fold": 1,
        "finite_family_block_80_90_intervals_possible": False,
        "historical_files_preserved": len(historical()),
        "metrics_recomputed_byte_identically": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", default="benchmarks/m13")
    args = parser.parse_args()
    root = Path(args.directory)
    result = verify(root)
    write_json(root / "validation.json", result)
    print(json.dumps(result, indent=2))
