"""Bounded registered M23 numerical study; no biological imports or inputs."""

import os

for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from proteinmod import geometry_study as g
from proteinmod.synthetic_checkpoint import canonical, environment

ROOT = Path(__file__).resolve().parents[1]


def candidate_table(data, states, scenario, seed, cfg):
    rows = []
    test_predictions = {}
    validation_losses = {}
    for method, models in states.items():
        test_predictions[method] = []
        validation_losses[method] = {n: [] for n in cfg["n_validation"]}
        for index, state in enumerate(models):
            test_x, test_y = data["test"]
            prediction = g.predict(state, test_x)
            test_predictions[method].append(prediction)
            test = g.metrics(test_y, prediction)
            row = {
                "seed": seed,
                "scenario": scenario,
                "method": method,
                "index": index,
                "lambda": state["penalty"],
                "state_sha256": g.state_digest(state),
                "test_mse": test["mse"],
                "test_rho": test["spearman"],
                "population_test_mse": g.population(state, scenario, "test", cfg),
                "population_validation_mse": g.population(state, scenario, "validation", cfg),
                "coefficient_error": g.recovery(state, scenario, cfg),
            }
            x, y = data["validation"]
            val_prediction = g.predict(state, x)
            for n in cfg["n_validation"]:
                observed = g.metrics(y[:n], val_prediction[:n])
                row[f"validation_mse_{n}"] = observed["mse"]
                row[f"validation_rho_{n}"] = observed["spearman"]
                validation_losses[method][n].append(observed["mse"])
            rows.append(row)
    return rows, test_predictions, validation_losses


def selections(rows, losses, seed, scenario, cfg):
    result = []
    for method in g.METHODS:
        candidates = [r for r in rows if r["method"] == method]
        fixed = candidates[cfg["fixed_index"]]
        for n in cfg["n_validation"]:
            for k, indices in ((4, cfg["small_indices"]), (16, list(range(16)))):
                index = g.select(losses[method][n], indices)
                chosen = candidates[index]
                redundant = g.select(losses[method][n], cfg["small_indices"] * 4)
                if redundant != g.select(losses[method][n], cfg["small_indices"]):
                    raise ValueError("Redundant multiplicity control failed")
                population = [candidates[i]["population_test_mse"] for i in indices]
                result.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "method": method,
                        "n_validation": n,
                        "k": k,
                        "index": index,
                        "lambda": chosen["lambda"],
                        "test_mse": chosen["test_mse"],
                        "test_rho": chosen["test_rho"],
                        "validation_mse": chosen[f"validation_mse_{n}"],
                        "validation_test_gap": chosen["test_mse"] - chosen[f"validation_mse_{n}"],
                        "fixed_test_mse": fixed["test_mse"],
                        "selected_minus_fixed": chosen["test_mse"] - fixed["test_mse"],
                        "finite_test_oracle_regret": chosen["test_mse"]
                        - min(candidates[i]["test_mse"] for i in indices),
                        "population_oracle_regret": chosen["population_test_mse"] - min(population)
                        if method != "mlp"
                        else None,
                        "validation_optimism": chosen["population_validation_mse"]
                        - chosen[f"validation_mse_{n}"]
                        if method != "mlp"
                        else None,
                        "coefficient_error": chosen["coefficient_error"],
                    }
                )
    return result


def execute_unit(seed, cfg):
    data = g.generate(seed, cfg)
    for split in ("train", "validation", "test"):
        if not np.array_equal(data["A"][split][1], data["B"][split][1]):
            raise ValueError("A/B target pairing failed")
    if any(
        not np.array_equal(a, b)
        for a, b in zip(data["B"]["train"], data["C"]["train"], strict=True)
    ):
        raise ValueError("B/C training identity failed")
    penalties = np.power(10.0, cfg["lambda_exponents"])
    fitted = {}
    maximum = 0.0
    negative = {}
    input_hashes = {}
    geometries = {}
    for scenario in ("A", "B"):
        x, y = data[scenario]["train"]
        states = {method: [] for method in g.METHODS}
        input_hashes[scenario] = g.m22.array_digest(x, y)
        geometries[scenario] = g.geometry(x)
        for method in g.METHODS:
            for penalty in penalties:
                first = g.fit(x, y, method, float(penalty), seed, cfg)
                repeated = g.fit(x, y, method, float(penalty), seed, cfg)
                g.equal(first, repeated)
                states[method].append(first)
        maximum = max(maximum, g.numerical_controls(states, x, y, penalties, cfg))
        fitted[scenario] = states
        permuted = y[g.rng(seed, 400).permutation(len(y))]
        negative[scenario] = {}
        for method in g.METHODS:
            control = g.fit(x, permuted, method, 0.1, seed, cfg)
            g.equal(control, g.fit(x, permuted, method, 0.1, seed, cfg))
            negative[scenario][method] = control
    fitted["C"] = fitted["B"]
    negative["C"] = negative["B"]
    geometries["C"] = geometries["B"]
    input_hashes["C"] = input_hashes["B"]
    all_rows = []
    all_selected = []
    predictions = {}
    losses = {}
    negative_metrics = {}
    for scenario in ("A", "B", "C"):
        rows, prediction, validation = candidate_table(
            data[scenario], fitted[scenario], scenario, seed, cfg
        )
        all_rows.extend(rows)
        all_selected.extend(selections(rows, validation, seed, scenario, cfg))
        predictions[scenario] = prediction
        losses[scenario] = validation
        negative_metrics[scenario] = {
            method: g.metrics(
                data[scenario]["test"][1], g.predict(state, data[scenario]["test"][0])
            )
            for method, state in negative[scenario].items()
        }
        for method, models in fitted[scenario].items():
            before = [g.state_digest(state) for state in models]
            # Perturb only evaluation labels. Neither training state nor test-independent choices can change.
            for state in models:
                g.metrics(
                    -1000 * data[scenario]["test"][1] + 1,
                    g.predict(state, data[scenario]["test"][0]),
                )
                g.metrics(
                    -7 * data[scenario]["validation"][1],
                    g.predict(state, data[scenario]["validation"][0]),
                )
            if before != [g.state_digest(state) for state in models]:
                raise ValueError("Held-out-label state noninterference failed")
            for n in cfg["n_validation"]:
                chosen = g.select(validation[method][n], list(range(16)))
                again = [
                    g.metrics(
                        data[scenario]["validation"][1][:n],
                        g.predict(state, data[scenario]["validation"][0])[:n],
                    )["mse"]
                    for state in models
                ]
                if chosen != g.select(again, list(range(16))):
                    raise ValueError("Test label affected selection")
    by = {(r["scenario"], r["method"], r["n_validation"], r["k"]): r for r in all_selected}
    decomposition = []
    for method in g.METHODS:
        for n in cfg["n_validation"]:
            for k in (4, 16):
                b = by[("B", method, n, k)]
                c = by[("C", method, n, k)]
                frozen_on_c = g.metrics(data["C"]["test"][1], predictions["C"][method][b["index"]])[
                    "mse"
                ]
                shift = frozen_on_c - b["test_mse"]
                selection = c["test_mse"] - frozen_on_c
                total = c["test_mse"] - b["test_mse"]
                if not np.isclose(total, shift + selection, rtol=1e-12, atol=1e-12):
                    raise ValueError("Shift decomposition sum failed")
                decomposition.append(
                    {
                        "seed": seed,
                        "method": method,
                        "n_validation": n,
                        "k": k,
                        "frozen_model_shift": shift,
                        "selection_change": selection,
                        "total": total,
                    }
                )
    return (
        {
            "seed": seed,
            "input_sha256": input_hashes,
            "geometry": geometries,
            "numerical_max_error": maximum,
            "exact_replay": True,
            "heldout_noninterference": True,
            "negative_controls": negative_metrics,
        },
        all_rows,
        all_selected,
        decomposition,
    )


def summarize(units, rows, selected, cfg):
    candidates = {(r["seed"], r["scenario"], r["method"], r["index"]): r for r in rows}
    choices = {
        (r["seed"], r["scenario"], r["method"], r["n_validation"], r["k"]): r for r in selected
    }
    fixed = cfg["fixed_index"]
    primary = {}
    alpha = 0.05 / 6
    vector = lambda function: [function(seed) for seed in cfg["seeds"]]
    gain = lambda seed, scenario: (
        candidates[(seed, scenario, "unscaled", fixed)]["test_mse"]
        - candidates[(seed, scenario, "standardized", fixed)]["test_mse"]
    )
    primary["H_A"] = g.paired(vector(lambda s: gain(s, "A")), alpha)
    primary["H_A"]["decision"] = (
        "supported equivalence"
        if primary["H_A"]["ci"][0] > -0.05 and primary["H_A"]["ci"][1] < 0.05
        else "inconclusive equivalence"
    )
    primary["H_B"] = g.paired(vector(lambda s: gain(s, "B")), alpha)
    primary["H_B"]["decision"] = (
        "supported geometry benefit"
        if primary["H_B"]["ci"][0] > 0.1 and primary["H_B"]["positive"] >= 30
        else "not established"
    )
    primary["H_V"] = g.paired(
        vector(
            lambda s: (
                choices[(s, "B", "standardized", 16, 16)]["test_mse"]
                - choices[(s, "B", "standardized", 256, 16)]["test_mse"]
            )
        ),
        alpha,
    )
    primary["H_K"] = g.paired(
        vector(
            lambda s: (
                choices[(s, "B", "standardized", 16, 16)]["test_mse"]
                - choices[(s, "B", "standardized", 16, 4)]["test_mse"]
            )
        ),
        alpha,
    )
    for key, positive, negative in (
        ("H_V", "small-validation penalty", "opposite validation-size effect"),
        ("H_K", "multiplicity harm", "multiplicity benefit"),
    ):
        primary[key]["decision"] = (
            positive
            if primary[key]["ci"][0] > 0.05
            else negative
            if primary[key]["ci"][1] < -0.05
            else "inconclusive meaningful direction"
        )
    primary["H_S"] = g.paired(
        vector(
            lambda s: (
                candidates[(s, "C", "unscaled", fixed)]["population_test_mse"]
                - candidates[(s, "B", "unscaled", fixed)]["population_test_mse"]
            )
        ),
        alpha,
    )
    primary["H_S"]["decision"] = (
        "identified covariance-risk effect"
        if primary["H_S"]["ci"][0] > 0.1 or primary["H_S"]["ci"][1] < -0.1
        else "not established"
    )
    primary["H_G"] = {
        method: g.paired(
            vector(
                lambda s, method=method: (
                    candidates[(s, "B", method, fixed)]["test_mse"]
                    - choices[(s, "B", "standardized", 256, 16)]["test_mse"]
                )
            ),
            0.05 / (6 * 4),
        )
        for method in g.METHODS
    }
    primary["H_G"]["decision"] = (
        "meaningful gain over every fixed comparator"
        if all(primary["H_G"][m]["ci"][0] > 0.1 for m in g.METHODS)
        else "meaningful gain over strongest fixed comparator not established"
    )
    reliability = []
    for scenario in ("A", "B", "C"):
        for method in g.METHODS:
            for n in cfg["n_validation"]:
                for k in (4, 16):
                    subset = [choices[(seed, scenario, method, n, k)] for seed in cfg["seeds"]]
                    effects = [r["selected_minus_fixed"] for r in subset]
                    wins = sum(v < -1e-12 for v in effects)
                    losses = sum(v > 1e-12 for v in effects)
                    reliability.append(
                        {
                            "scenario": scenario,
                            "method": method,
                            "n_validation": n,
                            "k": k,
                            "test_mse_mean": float(np.mean([r["test_mse"] for r in subset])),
                            "selected_minus_fixed": g.paired(effects),
                            "wins": wins,
                            "losses": losses,
                            "ties": 40 - wins - losses,
                            "win_ci95": g.wilson(wins, 40),
                            "loss_ci95": g.wilson(losses, 40),
                            "population_oracle_regret_mean": float(
                                np.mean([r["population_oracle_regret"] for r in subset])
                            )
                            if method != "mlp"
                            else None,
                            "validation_optimism_mean": float(
                                np.mean([r["validation_optimism"] for r in subset])
                            )
                            if method != "mlp"
                            else None,
                        }
                    )
    negative = g.paired([u["negative_controls"]["A"]["unscaled"]["spearman"] for u in units])
    if abs(negative["mean"]) > 0.1 and (negative["ci"][0] > 0 or negative["ci"][1] < 0):
        raise ValueError("Label-permutation control failed; mechanistic claims stopped")
    return {
        "primary": primary,
        "reliability": reliability,
        "negative_control_A_unscaled": negative,
        "max_numerical_error": max(u["numerical_max_error"] for u in units),
    }


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "runs/m23"):
        raise ValueError("Use new output under runs/m23")
    output.mkdir(parents=True, exist_ok=False)
    cfg = g.config()
    units = []
    rows = []
    selected = []
    decomposition = []
    failures = []
    for seed in cfg["seeds"]:
        try:
            unit, cases, choices, shift = execute_unit(seed, cfg)
            units.append(unit)
            rows.extend(cases)
            selected.extend(choices)
            decomposition.extend(shift)
            print(
                f"seed {seed}: complete; exact replay, scalar/diagonal, leakage and duplicate-search controls PASS",
                flush=True,
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ) as error:
            failures.append({"seed": seed, "error": f"{type(error).__name__}: {error}"})
            break
    summary = None
    if not failures:
        try:
            summary = summarize(units, rows, selected, cfg)
        except ValueError as error:
            failures.append({"error": str(error)})
    if rows:
        write_csv(output / "candidates.csv", rows)
    if selected:
        write_csv(output / "selections.csv", selected)
    if decomposition:
        write_csv(output / "decomposition.csv", decomposition)
    result = {
        "engineering": "PASS" if not failures and len(units) == 40 else "FAIL",
        "failures": failures,
        "config": cfg,
        "environment": environment(),
        "units": units,
        "summary": summary,
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "src/proteinmod/geometry_study.py",
                "scripts/run_m23.py",
                "src/proteinmod/synthetic_regression.py",
            )
        },
        "artifacts_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.csv")
        },
    }
    (output / "results.json").write_bytes(canonical(result) + b"\n")
    if result["engineering"] != "PASS":
        raise RuntimeError("Registered study stopped; retained failures in results.json")
    print(json.dumps(summary["primary"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
