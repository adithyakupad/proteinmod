"""Offline M12 provenance, roles, predictions and family-balanced metric verification."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, verify_file, write_json
from proteinmod.family_data import audit, collection, folds, similarity
from proteinmod.family_experiments import aggregate, historical
from proteinmod.family_report import summarize
from proteinmod.structural_graph import load_graph


def verify(directory):
    manifest_path = directory / "artifact_hashes.json"
    if manifest_path.exists():
        for name, digest in json.loads(manifest_path.read_text()).items():
            assert checksum(directory / name) == digest, name
    config = json.loads(Path("configs/m12.json").read_text())
    prereg = json.loads((directory / "preregistration.json").read_text())
    assert prereg["config_sha256"] == checksum("configs/m12.json")
    assert prereg["plan_sha256"] == checksum(directory / "PREREGISTRATION.md")
    rows, data = collection(config)
    assert json.loads(json.dumps(rows)) == json.loads((directory / "rows.json").read_text())
    evidence = json.loads((directory / "audit.json").read_text())
    assert data == evidence["data"] and historical() == evidence["historical_sha256"]
    assert similarity(config) == evidence["similarity"]
    mapping = {r["id"]: i for i, r in enumerate(rows)}
    mask = np.array(evidence["structural_mask"])
    roles_lookup = {}
    neighbor_audits = []
    classifications = []
    for spec in config["proteins"]:
        verify_file(spec["structure"]["path"], spec["structure"])
        g = load_graph(spec["structure_artifacts"], spec["reference"]["sequence"])
        assert g["audit"] == evidence["features"][spec["protein_id"]]["graph"]
        manifest = (
            Path(evidence["features"][spec["protein_id"]]["residue_directory"]) / "manifest.json"
        )
        assert (
            checksum(manifest)
            == evidence["features"][spec["protein_id"]]["residue_manifest_sha256"]
        )
        external = json.loads(Path(spec["classification"]["path"]).read_text())
        spans = spec["classification"]["entity_domain_spans"]
        positions = [
            n["reference_position"]
            for n in g["nodes"]
            if any(s["beg_seq_id"] <= n["polymer_position"] <= s["end_seq_id"] for s in spans)
        ]
        assert len(positions) / len(g["nodes"]) >= 0.5
        classifications.append(
            {
                "protein_id": spec["protein_id"],
                "family": spec["family"],
                "family_name": spec["family_name"],
                "pfam_version": spec["classification"]["pfam_version"],
                "entity_domain_spans": spans,
                "matched_assay_positions": positions,
                "assay_fraction_in_domain": len(positions) / len(spec["reference"]["sequence"]),
                "selected_variants_with_outside_domain_sites": sum(
                    any(int(t[1:-1]) not in positions for t in r["mutations"])
                    for r in rows
                    if r["protein_id"] == spec["protein_id"]
                ),
                "classification_sha256": spec["classification"]["sha256"],
                "source": spec["classification"]["url"],
                "sifts_alignments": external.get("rcsb_polymer_entity_align", []),
            }
        )
    for protocol, fold, original in folds(rows, config["family_order"]):
        for cohort in ("sequence_all", "structure_matched"):
            roles = {
                s: ix if cohort == "sequence_all" else ix[mask[ix]] for s, ix in original.items()
            }
            saved = next(
                x
                for x in evidence["roles"]
                if (x["protocol"], x["fold"], x["cohort"]) == (protocol, fold, cohort)
            )
            assert audit(rows, roles, protocol) == saved["roles"]
            roles_lookup[(protocol, fold, cohort)] = roles
        if protocol != "G1":
            train = {rows[i]["protein_id"] for i in original["train"]}
            for pid in sorted({rows[i]["protein_id"] for i in original["test"]}):
                pairs = [
                    p
                    for p in evidence["similarity"]["pairs"]
                    if (p["a"] == pid and p["b"] in train) or (p["b"] == pid and p["a"] in train)
                ]
                nearest = max(pairs, key=lambda p: p["global_alignment"]["identity"])
                neighbor_audits.append(
                    {
                        "protocol": protocol,
                        "fold": fold,
                        "test_protein": pid,
                        "nearest_training_pair": [nearest["a"], nearest["b"]],
                        "global": nearest["global_alignment"],
                        "local": nearest["local_alignment"],
                    }
                )
    with (directory / "predictions.csv").open() as f:
        predictions = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for p in predictions:
        assert float(p["measured"]) == rows[mapping[p["id"]]]["target"]
        assert np.isfinite(float(p["prediction"]))
        for key in ("protein_id", "family", "assay_id"):
            assert p[key] == rows[mapping[p["id"]]][key]
        grouped[(p["protocol"], p["fold"], p["cohort"], p["model"], int(p["seed"]))].append(p)
    records = json.loads((directory / "runs.json").read_text())
    assert (
        len(records) == len(grouped) == 13 * (2 + 3 + len(config["seeds"]) * len(config["modes"]))
    )
    for r in records:
        key = (r["protocol"], r["fold"], r["cohort"], r["model"], r["seed"])
        part = grouped[key]
        roles = roles_lookup[key[:3]]
        assert [p["id"] for p in part] == [rows[i]["id"] for i in roles["test"]]
        metrics = aggregate(rows, roles["test"], [float(p["prediction"]) for p in part])
        for name, value in metrics.items():
            assert value == r[name]
        selection = r["selection"]
        if "checkpoint_path" in selection:
            assert checksum(selection["checkpoint_path"]) == selection["checkpoint_sha256"]
            assert 1 <= selection["epoch"] <= config["training"]["max_epochs"]
        if r["cohort"] == "sequence_all":
            other = grouped[(r["protocol"], r["fold"], "structure_matched", r["model"], r["seed"])]
            if mask.all():
                assert [p["prediction"] for p in part] == [p["prediction"] for p in other]
    summary = json.loads((directory / "summary.json").read_text())
    assert summarize(rows, predictions) == summary["all_test_variants"]
    assert summarize(rows, predictions, True) == summary["common_G1_test_variants"]
    return {
        "verified": True,
        "fits": len(records),
        "held_out_predictions": len(predictions),
        "historical_files_preserved": len(historical()),
        "family_classifications": classifications,
        "nearest_training_neighbors": neighbor_audits,
        "undefined_protein_correlations": sum(
            v["spearman_rho"] is None for r in records for v in r["per_protein"].values()
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--directory", default="benchmarks/m12")
    args = parser.parse_args()
    directory = Path(args.directory)
    result = verify(directory)
    write_json(directory / "validation.json", result)
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("family_classifications", "nearest_training_neighbors")
        }
    )
