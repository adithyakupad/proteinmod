"""Execute the preregistered M19 development study, without external data access."""

import copy
import csv
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from proteinmod.data import checksum, write_json
from proteinmod.family_experiments import inputs
from proteinmod.multiprotein_experiments import per_protein
from proteinmod.release_integrity import verify_digest
from proteinmod.ridge_geometry import fit, predict


def main():
    root = Path("benchmarks/m19")
    config = json.loads((root / "protocol.json").read_text())
    for path, digest in config["inputs"].items():
        assert checksum(path) == digest
    output = Path("runs/m19")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows, matrices, mask, _, evidence = inputs(json.loads(Path("configs/m12.json").read_text()))
    assert mask.all()
    assert json.loads(json.dumps(rows)) == json.loads(Path("benchmarks/m12/rows.json").read_text())
    matrix = matrices["esm_sites"]
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    cases = json.loads(Path("benchmarks/m17/split-manifest.json").read_text())["folds"]
    cases = [case for case in cases if case["study"] == "H1"]
    assert len(cases) == 80
    historical = {}
    with Path("benchmarks/m17/predictions.csv").open() as handle:
        for r in csv.DictReader(handle):
            if r["study"] == "H1" and r["model"] in ("esm_local_ridge", "esm_unscaled_ridge"):
                historical[
                    (r["fold"], int(r["seed"]), float(r["fraction"]), r["model"], r["id"])
                ] = float(r["prediction"])
    records, predictions = [], []
    max_historical_error = 0.0
    max_scalar_error = 0.0
    scalar_failures = []
    for case_index, case in enumerate(cases):
        roles = {k: np.array([lookup[x] for x in ids]) for k, ids in case["roles"].items()}
        families = [{rows[i]["family"] for i in indices} for indices in roles.values()]
        assert all(not families[a] & families[b] for a, b in [(0, 1), (0, 2), (1, 2)])
        arms = {}
        for normalized in [False, True]:
            for scaled in [False, True]:
                name = ("expanded" if normalized else "historical") + (
                    "_scaled" if scaled else "_unscaled"
                )
                penalties = config["penalties"] if normalized else config["historical_alphas"]
                state, selection = fit(matrix, rows, roles, penalties, scaled, normalized)
                repeated, repeat_selection = fit(matrix, rows, roles, penalties, scaled, normalized)
                assert selection == repeat_selection
                for key in state:
                    np.testing.assert_array_equal(state[key], repeated[key])
                blinded = copy.deepcopy(rows)
                for i in roles["test"]:
                    blinded[i]["target"] = -999999.0
                other, other_selection = fit(matrix, blinded, roles, penalties, scaled, normalized)
                assert other_selection == selection
                for key in state:
                    np.testing.assert_array_equal(state[key], other[key])
                if normalized:
                    exact_matrix = matrix.astype(float)
                    reference, _ = fit(exact_matrix, rows, roles, penalties, scaled, normalized)
                    scalar, _ = fit(exact_matrix * 7, rows, roles, penalties, scaled, normalized)
                    error = float(
                        np.max(
                            np.abs(
                                predict(exact_matrix, reference) - predict(exact_matrix * 7, scalar)
                            )
                        )
                    )
                    max_scalar_error = max(max_scalar_error, error)
                    if error >= 1e-7:
                        scalar_failures.append({"case": case_index, "model": name, "error": error})
                path = output / f"{case_index:03d}_{name}.npz"
                np.savez_compressed(path, **state)
                with np.load(path) as restored:
                    pred = predict(matrix, restored)[roles["test"]]
                np.testing.assert_array_equal(pred, predict(matrix, state)[roles["test"]])
                arms[name] = pred, selection, path
                if not normalized:
                    old_name = "esm_local_ridge" if scaled else "esm_unscaled_ridge"
                    expected = [
                        historical[
                            (case["fold"], case["seed"], case["fraction"], old_name, rows[i]["id"])
                        ]
                        for i in roles["test"]
                    ]
                    error = float(np.max(np.abs(pred - expected)))
                    max_historical_error = max(max_historical_error, error)
                    assert error < 1e-12
        winner = "expanded_unscaled"
        if arms["expanded_scaled"][1]["validation_rho"] > arms[winner][1]["validation_rho"]:
            winner = "expanded_scaled"
        arms["expanded_selected"] = arms[winner]
        control, selection = fit(
            matrix, rows, roles, config["penalties"], False, True, config["controls_seed"]
        )
        path = output / f"{case_index:03d}_shuffled.npz"
        np.savez_compressed(path, **control)
        with np.load(path) as restored:
            control_predictions = predict(matrix, restored)[roles["test"]]
        np.testing.assert_array_equal(control_predictions, predict(matrix, control)[roles["test"]])
        arms["shuffled_train"] = control_predictions, selection, path
        for name, (pred, selection, path) in arms.items():
            protein_scores = per_protein(rows, roles["test"], pred)
            rho = float(
                np.mean(
                    [
                        s["spearman_rho"] if s["spearman_rho"] is not None else 0
                        for s in protein_scores.values()
                    ]
                )
            )
            errors = []
            for pid in protein_scores:
                indices = [j for j, i in enumerate(roles["test"]) if rows[i]["protein_id"] == pid]
                truth = [rows[roles["test"][j]]["target"] for j in indices]
                errors.append(
                    float(np.mean(np.abs(rankdata(pred[indices]) - rankdata(truth))) / len(indices))
                )
            records.append(
                {
                    "fold": case["fold"],
                    "fraction": case["fraction"],
                    "seed": case["seed"],
                    "model": name,
                    "rho": rho,
                    "rank_mae": float(np.mean(errors)),
                    "selection": selection,
                    "selected_geometry": winner if name == "expanded_selected" else name,
                    "per_protein": protein_scores,
                    "state": str(path),
                    "state_sha256": checksum(path),
                }
            )
            for i, value in zip(roles["test"], pred, strict=True):
                predictions.append(
                    {
                        "fold": case["fold"],
                        "fraction": case["fraction"],
                        "seed": case["seed"],
                        "model": name,
                        "id": rows[i]["id"],
                        "protein_id": rows[i]["protein_id"],
                        "target": rows[i]["target"],
                        "prediction": float(value),
                    }
                )
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["fraction"], r["fold"], r["model"])].append(r["rho"])
    effects = []
    for fraction in config["fractions"]:
        for family in sorted({r["fold"] for r in records}):
            baseline = np.mean(grouped[(fraction, family, "historical_unscaled")])
            for name in sorted(arms):
                rho = float(np.mean(grouped[(fraction, family, name)]))
                effects.append(
                    {
                        "fraction": fraction,
                        "family": family,
                        "model": name,
                        "rho": rho,
                        "minus_historical_unscaled": rho - baseline,
                    }
                )
    primary = [
        r["minus_historical_unscaled"]
        for r in effects
        if r["fraction"] == 1 and r["model"] == "expanded_selected"
    ]
    gain = float(np.mean(primary))
    summary = {
        "primary_gain": gain,
        "positive_families": int(sum(x > 0 for x in primary)),
        "family_effects": primary,
        "family_sd": float(np.std(primary, ddof=1)),
        "leave_one_family_out": [float(np.mean(primary[:i] + primary[i + 1 :])) for i in range(4)],
        "success": bool(gain >= 0.02 and sum(x > 0 for x in primary) >= 3 and not scalar_failures),
        "scalar_failures": scalar_failures,
        "historical_max_error": max_historical_error,
        "scalar_max_error": max_scalar_error,
        "exact_refits": 320,
        "exact_replayed_states": 400,
        "seconds": time.perf_counter() - started,
        "device": "cpu",
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "torch"]
        },
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol_commit": "7664fd2",
        "evidence": evidence,
    }
    for filename, value in [
        ("runs.json", records),
        ("family-effects.json", effects),
        ("summary.json", summary),
    ]:
        write_json(output / filename, value)
    with (output / "predictions.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    for path, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        verify_digest(path, digest)
    print(json.dumps({k: v for k, v in summary.items() if k != "evidence"}, indent=2))


if __name__ == "__main__":
    main()
