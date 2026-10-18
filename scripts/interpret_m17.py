"""Apply preregistered descriptive thresholds; no population significance claims."""

import json
from pathlib import Path

import numpy as np

from proteinmod.data import write_json


def interpret():
    assert json.loads(Path("configs/m17.json").read_text())["hypothesis_margin_rho"] == 0.02
    root = Path("runs/m17")
    r = json.loads((root / "summary.json").read_text())["per_family"]
    families = sorted({x["fold"] for x in r if x["study"] == "H1"})

    def value(model, family, fraction=1):
        return next(
            x["rho"]
            for x in r
            if x["study"] == "H1"
            and x["model"] == model
            and x["fold"] == family
            and x["fraction"] == fraction
        )

    def contrast(a, b, fraction=1):
        d = {f: value(a, f, fraction) - value(b, f, fraction) for f in families}
        return {
            "per_family": d,
            "mean": float(np.mean(list(d.values()))),
            "positive_families": sum(v > 0 for v in d.values()),
        }

    changes = {
        f: (value("structural_ridge", f, 0.1) - value("esm_local_ridge", f, 0.1))
        - (value("structural_ridge", f) - value("esm_local_ridge", f))
        for f in families
    }
    h1 = {
        "gap_drop_10_to_100_per_family": changes,
        "mean_drop": float(np.mean(list(changes.values()))),
        "positive_families": sum(v > 0 for v in changes.values()),
        "status": "SUPPORTED_DESCRIPTIVELY"
        if np.mean(list(changes.values())) >= 0.02 and sum(v > 0 for v in changes.values()) >= 3
        else "UNRESOLVED_NOT_UNIFORM",
    }
    diversity = {}
    h2 = []
    pids = {
        p["protein_id"]: p["family"]
        for p in json.loads(Path("benchmarks/m12/rows.json").read_text())
    }
    for model in ("structural_ridge", "esm_local_ridge"):
        effects = {}
        for f in families:
            a = [
                x["rho"]
                for x in r
                if x["study"] == "diversity"
                and x["fold"] == f
                and x["model"] == model
                and x["protocol"].startswith("diverse")
            ]
            b = [
                x["rho"]
                for x in r
                if x["study"] == "diversity"
                and x["fold"] == f
                and x["model"] == model
                and x["protocol"].startswith("narrow")
            ]
            effects[f] = float(np.mean(a) - np.mean(b))
            score = {
                p: float(
                    np.mean(
                        [
                            x["rho"]
                            for x in r
                            if x["study"] == "H2"
                            and pids[x["fold"]] == f
                            and x["model"] == model
                            and x["protocol"] == p
                        ]
                    )
                )
                for p in ("G1", "G2", "G3")
            }
            h2.append(
                {
                    "family": f,
                    "model": model,
                    **score,
                    "G2_minus_G1": score["G2"] - score["G1"],
                    "G3_minus_G1": score["G3"] - score["G1"],
                    "diverse_minus_narrow": effects[f],
                }
            )
        diversity[model] = {
            "per_family": effects,
            "mean": float(np.mean(list(effects.values()))),
            "positive_families": sum(v > 0 for v in effects.values()),
            "status": "SUPPORTED_DESCRIPTIVELY"
            if np.mean(list(effects.values())) >= 0.02 and sum(v > 0 for v in effects.values()) >= 3
            else "NOT_SUPPORTED_UNDER_FIXED_LABEL_BUDGET",
        }
    h3 = {
        str(f): {
            "edge_minus_zero": contrast("edge_full", "zero_message", f),
            "edge_minus_degree_null": contrast("edge_full", "degree_null", f),
            "edge_minus_shuffled_chemistry": contrast("edge_full", "edge_shuffled_chemistry", f),
        }
        for f in (0.25, 1)
    }
    h5 = {
        str(f): {
            "unscaled_minus_scaled_ESM": contrast("esm_unscaled_ridge", "esm_local_ridge", f),
            "global_minus_local_ESM": contrast("esm_global_ridge", "esm_local_ridge", f),
            "local_minus_global_MPNN": contrast("mpnn_local_ridge", "mpnn_global_ridge", f),
            "structural_MLP_minus_ridge": contrast("structural_mlp", "structural_ridge", f),
        }
        for f in (0.1, 0.25, 0.5, 1)
    }
    write_json(
        root / "hypotheses.json",
        {
            "H1": h1,
            "H2": {"paired_family_effects": h2, "diversity": diversity},
            "H3": h3,
            "H4": "No missing physical contacts or canonical heavy atoms in this cohort; coverage loss is not supported as its explanation. Does not establish all relevant chemistry is represented.",
            "H5": h5,
            "independent_confirmation": "BLOCKED",
            "population_CI": "Not estimated, four development families only",
        },
    )


if __name__ == "__main__":
    interpret()
