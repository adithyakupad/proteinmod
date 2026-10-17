"""M13 uncertainty experiments on unchanged M12/M6 test genotypes and held-out calibration."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, verify_file, write_json
from proteinmod.embeddings import fingerprint
from proteinmod.family_experiments import inputs
from proteinmod.structure_pipeline import write_csv
from proteinmod.uncertainty import ensemble_summary, shuffled, training_distance
from proteinmod.uncertainty_calibration import calibration_state, prediction_intervals
from proteinmod.uncertainty_data import historical, native_labels, roles_audit, uq_folds
from proteinmod.uncertainty_models import native_affine, predict_hybrid, ridge_ensemble


def output_path(value):
    output, protected = Path(value).resolve(), Path("benchmarks").resolve()
    if output == protected or protected in output.parents or output in protected.parents:
        raise ValueError("M13 output overlaps historical evidence")
    output.mkdir(parents=True, exist_ok=True)
    return output


def m6_inputs(m6, config):
    from proteinmod.stability_data import stability_folds
    from proteinmod.uncertainty_data import native_control_features

    rows, features, evidence = native_control_features(m6)
    roles = []
    for protocol, fold, original in stability_folds(rows, m6):
        calibration = []
        for pid in sorted({rows[i]["protein_id"] for i in original["train"]}):
            ix = sorted(
                (i for i in original["train"] if rows[i]["protein_id"] == pid),
                key=lambda i: hashlib.sha256(
                    (config["calibration_salt"] + rows[i]["id"]).encode()
                ).hexdigest(),
            )
            count = config["m6_native_control"][
                "s1_calibration_per_protein" if protocol == "S1" else "s2_calibration_from_training"
            ]
            calibration.extend(ix[:count])
        role = {
            **original,
            "train": np.array([i for i in original["train"] if i not in set(calibration)], int),
            "calibration": np.array(sorted(calibration), int),
        }
        roles_audit(rows, role, protocol)
        roles.append((protocol, fold, role))
    return rows, features, roles, evidence


def nearest_identity(rows, roles, similarity):
    train = {rows[i]["protein_id"] for i in roles["train"]}
    result = {}
    for pid in {r["protein_id"] for r in rows}:
        if pid in train:
            result[pid] = 1.0
            continue
        pairs = [
            p["global_alignment"]["identity"]
            for p in similarity["pairs"]
            if (p["a"] == pid and p["b"] in train) or (p["b"] == pid and p["a"] in train)
        ]
        result[pid] = max(pairs) if pairs else None
    return result


def run(config):
    import torch

    from proteinmod.hybrid_experiments import fit

    output = output_path(config["output"])
    before = historical()
    started = time.perf_counter()
    prereg = json.loads(Path("benchmarks/m13/preregistration.json").read_text())
    if (
        config != json.loads(Path("configs/m13.json").read_text())
        or checksum("configs/m13.json") != prereg["config_sha256"]
    ):
        raise ValueError("M13 configuration differs from preregistration")
    for name in ("m12_config", "m6_config"):
        verify_file(config[name]["path"], config[name])
    m12 = json.loads(Path(config["m12_config"]["path"]).read_text())
    m6 = json.loads(Path(config["m6_config"]["path"]).read_text())
    rows, features, mask, groups, evidence = inputs(m12)
    if not mask.all():
        raise ValueError("Registered M12 matched cohort changed")
    native, native_provenance = native_labels(rows, m12, m6)
    plans = uq_folds(rows, native, m12, config)
    similarity = json.loads(Path("benchmarks/m12/audit.json").read_text())["similarity"]
    root = Path(config["cache"]) / fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    predictions = []
    calibration_predictions = []
    role_records = []
    write_json(output / "m12_rows.json", rows)
    write_json(output / "native_labels.json", native_provenance)

    def emit(
        dataset,
        protocol,
        fold,
        roles,
        model,
        repeat,
        rank,
        native_predictions,
        selected,
        native_head,
        distance,
        identity,
        local_rows,
        local_y,
        elapsed,
        files,
    ):
        rank_members = rank[1:]
        native_members = native_predictions[1:]
        rank_summary = ensemble_summary(rank_members)
        native_summary = ensemble_summary(native_members)
        rp = np.array(rank_summary["point_prediction"])
        ru = np.array(rank_summary["uncertainty"])
        point = np.array(native_summary["point_prediction"])
        sd = np.array(native_summary["uncertainty"])
        shuffled_sd = sd.copy()
        shuffled_rank = ru.copy()
        for ix in roles.values():
            pids = [local_rows[i]["protein_id"] for i in ix]
            native_groups = [
                str(p) + (":native" if np.isfinite(local_y[i]) else ":missing")
                for p, i in zip(pids, ix, strict=True)
            ]
            shuffled_sd[ix] = shuffled(sd[ix], native_groups, config["bootstrap_seed"] + repeat)
            shuffled_rank[ix] = shuffled(ru[ix], pids, config["bootstrap_seed"] + repeat)
        state = calibration_state(
            local_y, point, sd, distance, shuffled_sd, local_rows, roles, config, protocol
        )
        intervals = prediction_intervals(
            point, sd, distance, shuffled_sd, local_rows, state, config
        )
        record = {
            "dataset": dataset,
            "protocol": protocol,
            "fold": fold,
            "model": model,
            "repeat": repeat,
            "selection": selected,
            "native_head": native_head,
            "calibration": state,
            "elapsed_seconds": elapsed,
            "files": files,
            "point_and_uncertainty_unit": "native kcal/mol; M12 rank outputs retained separately",
            "uncertainty_type": "epistemic_training_sensitivity_not_posterior",
            "aleatoric_variance_identified": False,
        }
        records.append(record)
        for role in ("calibration", "validation", "test"):
            for i in roles[role]:
                r = local_rows[i]
                eligible = bool(np.isfinite(local_y[i]))
                base = {
                    "dataset": dataset,
                    "protocol": protocol,
                    "fold": fold,
                    "model": model,
                    "repeat": repeat,
                    "role": role,
                    "id": r["id"],
                    "protein_id": r["protein_id"],
                    "family": r.get("family"),
                    "measured_native": float(local_y[i]) if eligible else None,
                    "native_point": float(point[i]),
                    "native_sd": float(sd[i]),
                    "native_sd_shuffled": float(shuffled_sd[i]),
                    "distance_to_training": float(distance[i]),
                }
                if role != "test":
                    if eligible:
                        calibration_predictions.append(base)
                    continue
                base.update(
                    {
                        "assay_id": r["assay_id"],
                        "measured_original": r["target"],
                        "original_unit": r["target_unit"],
                        "native_eligible": eligible,
                        "native_convention": "positive_destabilizing",
                        "native_unit": "kcal/mol",
                        "rank_point": float(rp[i]) if dataset == "M12" else None,
                        "rank_sd": float(ru[i]) if dataset == "M12" else None,
                        "rank_sd_shuffled": float(shuffled_rank[i]) if dataset == "M12" else None,
                        "deterministic_point": float(native_predictions[0, i]),
                        "deterministic_rank": float(rank[0, i]) if dataset == "M12" else None,
                        "rank_members": rank_members[:, i].tolist() if dataset == "M12" else None,
                        "native_members": native_members[:, i].tolist()
                        if dataset == "M6"
                        else None,
                        "mutation_count": len(r["mutations"]),
                        "structure_coverage": 1.0
                        if dataset == "M12"
                        else evidence6["graphs"][r["protein_id"]]["mapping"]["coordinate_coverage"],
                        "nearest_training_reference_identity": identity.get(r["protein_id"]),
                        "structural_eligible": True,
                    }
                )
                for key, bounds in intervals.items():
                    base[key + "_lower"] = bounds["lower"][i] if eligible else None
                    base[key + "_upper"] = bounds["upper"][i] if eligible else None
                predictions.append(base)
        print(
            dataset,
            protocol,
            fold,
            model,
            repeat,
            "cal",
            sum(np.isfinite(local_y[roles["calibration"]])),
            "native-test",
            sum(np.isfinite(local_y[roles["test"]])),
            "seconds",
            round(elapsed, 2),
            flush=True,
        )

    def persist():
        write_json(output / "runs.json", records)
        write_csv(output / "predictions.csv", predictions)
        write_csv(output / "calibration_predictions.csv", calibration_predictions)

    for protocol, fold, roles in plans:
        role_records.append(
            {
                "dataset": "M12",
                "protocol": protocol,
                "fold": fold,
                "roles": roles_audit(rows, roles, protocol),
            }
        )
        distance = training_distance(features["esm_sites"], roles["train"])
        identity = nearest_identity(rows, roles, similarity)
        fit_roles = {k: v for k, v in roles.items() if k != "calibration"}
        for repeat in config["replicates"]:
            for name in config["ridge_models"]:
                start = time.perf_counter()
                rank, selected, state = ridge_ensemble(
                    features[name],
                    rows,
                    roles,
                    m12["alphas"],
                    config["bootstrap_members"],
                    config["bootstrap_seed"] + 1000 * repeat,
                )
                path = root / f"M12-{protocol}-{fold}-{name}-{repeat}.npz"
                np.savez(path, **state)
                native_predictions, head = native_affine(
                    rank, native, roles["train"], config["native_affine_alpha"]
                )
                emit(
                    "M12",
                    protocol,
                    fold,
                    roles,
                    name + "_ridge",
                    repeat,
                    rank,
                    native_predictions,
                    selected,
                    head,
                    distance,
                    identity,
                    rows,
                    native,
                    time.perf_counter() - start,
                    {str(path): checksum(path)},
                )
            for mode in config["neural_modes"]:
                start = time.perf_counter()
                members = []
                selected = []
                files = {}
                for seed in config["ensemble_seeds"][repeat]:
                    pred, selection, state = fit(
                        groups, rows, fit_roles, m12["training"], seed, mode
                    )
                    full = predict_hybrid(groups, state, m12["training"], mode, len(rows), roles)
                    np.testing.assert_array_equal(full[roles["test"]], pred)
                    members.append(full)
                    selected.append({"seed": seed, **selection})
                    path = root / f"M12-{protocol}-{fold}-{mode}-{seed}.pt"
                    torch.save(state, path)
                    files[str(path)] = checksum(path)
                rank = np.stack([members[0], *members])
                native_predictions, head = native_affine(
                    rank, native, roles["train"], config["native_affine_alpha"]
                )
                emit(
                    "M12",
                    protocol,
                    fold,
                    roles,
                    mode,
                    repeat,
                    rank,
                    native_predictions,
                    selected,
                    head,
                    distance,
                    identity,
                    rows,
                    native,
                    time.perf_counter() - start,
                    files,
                )
        persist()
    evidence6 = None
    if config["m6_native_control"]["enabled"]:
        rows6, features6, plans6, evidence6 = m6_inputs(m6, config)
        y6 = np.array([r["target"] for r in rows6])
        write_json(output / "m6_rows.json", rows6)
        for protocol, fold, roles in plans6:
            role_records.append(
                {
                    "dataset": "M6",
                    "protocol": protocol,
                    "fold": fold,
                    "roles": roles_audit(rows6, roles, protocol),
                }
            )
            distance = training_distance(features6["esm_sites_ridge"], roles["train"])
            for repeat in config["replicates"]:
                for name in config["m6_native_control"]["models"]:
                    start = time.perf_counter()
                    pred, selected, state = ridge_ensemble(
                        features6[name],
                        rows6,
                        roles,
                        m6["alphas"],
                        config["bootstrap_members"],
                        config["bootstrap_seed"] + 1000 * repeat,
                        native=True,
                    )
                    path = root / f"M6-{protocol}-{fold}-{name}-{repeat}.npz"
                    np.savez(path, **state)
                    emit(
                        "M6",
                        protocol,
                        fold,
                        roles,
                        name,
                        repeat,
                        pred,
                        pred,
                        selected,
                        None,
                        distance,
                        {},
                        rows6,
                        y6,
                        time.perf_counter() - start,
                        {str(path): checksum(path)},
                    )
            persist()
    if historical() != before:
        raise ValueError("M0-M12 history changed")
    write_json(
        output / "audit.json",
        {
            "roles": role_records,
            "m12_features": evidence,
            "m6_features": evidence6,
            "historical_sha256": before,
            "native_label_source": m6["sources"]["measurements"],
            "native_assay": m6["assay"],
            "native_eligible": int(np.isfinite(native).sum()),
            "native_excluded": int((~np.isfinite(native)).sum()),
            "exchangeability": "G3 family shift and G2 protein shift: coverage empirical only; one G3 calibration family is not many independent mutations.",
        },
    )
    write_json(
        output / "environment.json",
        {
            "config_sha256": checksum("configs/m13.json"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "biopython")
            },
            "device": "deterministic CPU fits; previously cached frozen MPS ESM tokens",
            "seeds": config["ensemble_seeds"],
            "bootstrap_seed": config["bootstrap_seed"],
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    from proteinmod.uncertainty_report import report

    report(output, config)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/m13.json")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.report:
        from proteinmod.uncertainty_report import report

        report(Path(config["output"]), config)
    else:
        run(config)


if __name__ == "__main__":
    main()
