"""Posthoc source-failure diagnosis; no test outcome changes fitting or features."""

import csv
import json
from pathlib import Path

import numpy as np

from proteinmod.conditioning_features import mutation_features
from proteinmod.data import write_json
from proteinmod.multitask_experiments import inputs
from proteinmod.stability_features import model_inputs as legacy_features

config = json.loads(Path("configs/m8.json").read_text())
m7 = json.loads(Path(config["m7_config"]).read_text())
m5 = json.loads(Path(m7["function_config"]).read_text())
m4 = json.loads(Path(m5["m4_config"]).read_text())
rows, groups, folds, _ = inputs(m7)
features, _ = mutation_features(
    rows["function"],
    groups["function"],
    m4["collection"],
    m4["embedding"],
    Path(config["cache"]) / "wt",
)
m6 = json.loads(Path(m7["stability_config"]).read_text())
sf, _ = mutation_features(
    rows["stability"],
    groups["stability"],
    m6["proteins"],
    m6["embedding"],
    m6["residue_cache"],
    m6["sources"]["measurements"],
)
_, legacy, _ = legacy_features(m6)
feature_replay = {}
for key in sf:
    np.testing.assert_array_equal(sf[key], legacy[key])
    feature_replay[key] = {"dtype": str(sf[key].dtype), "rows": len(sf[key]), "exact": True}
sources = json.loads(Path("runs/m8/source_models.json").read_text())
terms = []
for source in sources:
    if source["source"] != "structural_ridge":
        continue
    w = source["weights"]
    coef, mean, scale = [np.array(w[k]) for k in ("coefficient", "mean", "scale")]
    for protein, group in groups["function"].items():
        i = group["indices"][0]
        x = features["structural_ridge"][i]
        contribution = coef * (x - mean) / scale
        order = np.argsort(np.abs(contribution))[::-1][:6]
        terms.append(
            {
                "function_fold": source["function_fold"],
                "protein_id": protein,
                "variant_id": rows["function"][i]["id"],
                "linear_decomposition_float64_kcal_mol": float(contribution.sum() + w["intercept"]),
                "top_terms": [
                    {
                        "feature_index_zero_based": int(j),
                        "semantic": (
                            "neighborhood B-factor missing fraction; "
                            + {86: "mutation mean", 174: "mutation sum", 262: "mutation max"}[
                                int(j)
                            ]
                        )
                        if j in (86, 174, 262)
                        else "other M3 structural descriptor",
                        "function_value": float(x[j]),
                        "source_train_mean": float(mean[j]),
                        "source_train_scale": float(scale[j]),
                        "standardized_coefficient": float(coef[j]),
                        "contribution_kcal_mol": float(contribution[j]),
                    }
                    for j in order
                ],
            }
        )
with Path("benchmarks/m6/predictions.csv").open() as handle:
    historical = list(csv.DictReader(handle))
replay = []
for f, s, roles in folds:
    for source in [x for x in sources if x["function_fold"] == f]:
        protocol = "S1" if f == "within" else "S2"
        expected = {
            r["id"]: float(r["prediction_ddg_kcal_mol"])
            for r in historical
            if r["protocol"] == protocol and r["fold"] == s and r["model"] == source["source"]
        }
        measured = np.array(
            [expected[rows["stability"][i]["id"]] for i in roles["stability"]["test"]]
        )
        prediction = np.array(source["source_test_predictions"])
        replay.append(
            {
                "protocol": protocol,
                "fold": s,
                "source": source["source"],
                "predictions": len(prediction),
                "max_absolute_difference_kcal_mol": float(np.max(np.abs(prediction - measured))),
            }
        )
        np.testing.assert_allclose(prediction, measured, rtol=0, atol=1e-5)
write_json(
    "runs/m8/source_failure_audit.json",
    {
        "posthoc_only": True,
        "no_model_changes": True,
        "diagnosis": "Nearly constant neighborhood missing-B-factor fractions (float32 summation noise around 1) are scaled by ~6e-8 in M6. Function experimental B-factor availability changes fraction to 0, causing huge OOD linear offsets. No clipping/retraining performed.",
        "feature_indices_zero_based": [86, 174, 262],
        "decomposition": terms,
        "m6_source_prediction_replay": replay,
        "m6_source_feature_replay": feature_replay,
    },
)
print("M6 source replay max difference", max(r["max_absolute_difference_kcal_mol"] for r in replay))
