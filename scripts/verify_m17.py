"""Offline artifact/role/metric checks; optional actual selected-state replay."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from proteinmod.data import checksum, write_json


def verify(root, states=False):
    for p, digest in json.loads(Path("benchmarks/m17/preregistration.json").read_text()).items():
        assert checksum(p) == digest, p
    for name, digest in json.loads((root / "artifact_hashes.json").read_text()).items():
        assert checksum(root / name) == digest, name
    for p, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        assert checksum(p) == digest, p
    for source, digest in json.loads((root / "manifest.json").read_text())["source_sha256"].items():
        assert checksum(source) == digest, source
    runs = json.loads((root / "runs.json").read_text())
    cases = json.loads(Path("benchmarks/m17/split-manifest.json").read_text())["folds"]
    key = lambda r: tuple(str(r[k]) for k in ("study", "fold", "protocol", "seed", "fraction"))
    case_lookup = {key(c): c for c in cases}
    rows = json.loads(Path("benchmarks/m12/rows.json").read_text())
    lookup = {r["id"]: r for r in rows}
    grouped = defaultdict(list)
    with (root / "predictions.csv").open() as f:
        for p in csv.DictReader(f):
            grouped[key(p) + (p["model"],)].append(p)
    for r in runs:
        case = case_lookup[key(r)]
        pp = grouped[key(r) + (r["model"],)]
        assert [p["id"] for p in pp] == case["roles"]["test"]
        assert len(pp) == len({p["id"] for p in pp})
        assert all(float(p["target"]) == lookup[p["id"]]["target"] for p in pp)
        for pid, result in r["per_protein"].items():
            subset = [p for p in pp if p["protein_id"] == pid]
            truth = [float(p["target"]) for p in subset]
            pred = [float(p["prediction"]) for p in subset]
            assert np.isfinite(pred).all()
            score = (
                None
                if np.ptp(pred) == 0 or np.ptp(truth) == 0
                else float(spearmanr(truth, pred).statistic)
            )
            expected = result["spearman_rho"]
            assert (
                score is None
                and expected is None
                or score is not None
                and expected is not None
                and np.isclose(score, expected, atol=1e-12, rtol=0)
            )
    state_result = replay(runs, cases, grouped) if states else None
    print(
        {
            "fits": len(runs),
            "predictions": sum(map(len, grouped.values())),
            "roles_and_metrics": "PASS",
            "state_replay": state_result,
        }
    )
    return state_result


def replay(runs, cases, grouped):
    import torch
    from sklearn.preprocessing import StandardScaler

    from proteinmod.edge_experiments import make_model, topology
    from proteinmod.mechanism_experiments import prepare

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rows, matrices, groups, nulls, _, _ = prepare()
    index = {r["id"]: i for i, r in enumerate(rows)}
    key = lambda r: tuple(str(r[k]) for k in ("study", "fold", "protocol", "seed", "fraction"))
    lookup = {key(c): c for c in cases}
    matrix_names = {
        "structural_ridge": "structural",
        "structural_mlp": "structural",
        "biochemical_ridge": "sequence",
        "esm_local_ridge": "esm_sites",
        "esm_mlp": "esm_sites",
        "esm_unscaled_ridge": "esm_sites",
        "esm_global_ridge": "esm_global",
        "wt_only_ridge": "esm_wt",
        "mpnn_local_ridge": "mpnn_local",
        "mpnn_global_ridge": "mpnn_global",
    }
    settings = json.loads(Path("configs/m17.json").read_text())["training"]
    maximum = 0.0
    for r in runs:
        assert checksum(r["state_path"]) == r["state_sha256"]
        ix = np.array([index[x] for x in lookup[key(r)]["roles"]["test"]])
        if r["model"] in matrix_names:
            state = dict(np.load(r["state_path"], allow_pickle=False))
            matrix = matrices[matrix_names[r["model"]]]
            if "mean" in state:
                scaler = StandardScaler()
                scaler.mean_ = state["mean"]
                scaler.scale_ = state["scale"]
                scaler.n_features_in_ = len(state["mean"])
                matrix = scaler.transform(matrix)
            x = matrix.astype(float)[ix]
            if "coefficient" in state:
                pred = x @ state["coefficient"] + state["intercept"]
            else:
                for i in range(2):
                    x = x @ state[f"weight_{i}"] + state[f"bias_{i}"]
                    if i == 0:
                        x = np.maximum(x, 0)
                pred = x.ravel()
        else:
            gg = nulls if r["model"] == "degree_null" else groups
            mode = "edge_full" if r["model"] == "degree_null" else r["model"]
            model = make_model(next(iter(gg.values())), mode, settings).eval()
            model.load_state_dict(torch.load(r["state_path"], weights_only=True))
            found = {}
            with torch.inference_mode():
                for g in gg.values():
                    local = np.flatnonzero(np.isin(g["indices"], ix))
                    if len(local):
                        values = (
                            model(torch.tensor(g["dynamic"][local]), topology(g, mode, 314159))
                            .numpy()
                            .astype(float)
                        )
                        found.update(zip(g["indices"][local], values, strict=True))
            pred = np.array([found[i] for i in ix])
        old = np.array([float(p["prediction"]) for p in grouped[key(r) + (r["model"],)]])
        error = float(abs(pred - old).max())
        maximum = max(maximum, error)
        assert error <= 1e-12, (r["model"], error)
    result = {"fits": len(runs), "maximum_absolute_prediction_error": maximum}
    write_json("runs/m17/state-replay.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmarks/m17"))
    parser.add_argument("--states", action="store_true")
    args = parser.parse_args()
    verify(args.directory, args.states)
