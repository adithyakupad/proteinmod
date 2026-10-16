"""Offline M14 evidence verification: actual saved inference, invariances and repeat fits."""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import checksum, write_json
from proteinmod.edge_chemistry import annotate, mapped_atoms
from proteinmod.edge_experiments import historical, make_model, prepare, ridge_predict, topology
from proteinmod.multiprotein_experiments import per_protein
from proteinmod.pretrained_structure import FrozenProteinMPNN
from proteinmod.stability_experiments import scores
from proteinmod.structural_graph import variant_inputs


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def verify(root, repeat=None):
    config = json.loads(Path("configs/m14.json").read_text())
    torch.set_num_threads(config["training"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    audit = json.loads((root / "audit.json").read_text())
    assert config == audit["config"] | {"output": config["output"]}
    assert historical() == audit["history_before"]
    prereg = json.loads((Path("benchmarks/m14") / "historical_hashes.json").read_text())
    assert historical() == prereg
    manifest_path = root / "artifact_hashes.json"
    if manifest_path.exists():
        for name, digest in json.loads(manifest_path.read_text()).items():
            assert checksum(root / name) == digest, name
    source_changes = []
    for p, digest in audit["sources"].items():
        if checksum(p) != digest:
            original = subprocess.check_output(["git", "show", audit["git_revision"] + ":" + p])
            assert hashlib.sha256(original).hexdigest() == digest, p
            source_changes.append(p)
    cohorts, provenance = prepare(config)
    for name in ("family", "native"):
        assert provenance[name] == audit["provenance"][name]
    assert provenance["pretrained"] == audit["provenance"]["pretrained"]
    records = json.loads((root / "metrics.json").read_text())
    saved = read_csv(root / "predictions.csv")
    lookup = defaultdict(list)
    for row in saved:
        lookup[tuple(row[k] for k in ("cohort", "protocol", "fold", "model", "seed"))].append(row)
    expected_roles = []
    for cohort, (rows, _, _, partitions, _) in cohorts.items():
        for protocol, fold, roles in partitions:
            expected_roles.append(
                {
                    "cohort": cohort,
                    "protocol": protocol,
                    "fold": fold,
                    "roles": {role: [rows[i]["id"] for i in ix] for role, ix in roles.items()},
                }
            )
    assert expected_roles == audit["roles"]
    maximum = 0.0
    for record in records:
        cohort = record["cohort"]
        rows, structural, groups, partitions, probe = cohorts[cohort]
        roles = next(
            roles
            for protocol, fold, roles in partitions
            if protocol == record["protocol"] and fold == record["fold"]
        )
        key = tuple(str(record[k]) for k in ("cohort", "protocol", "fold", "model", "seed"))
        found = lookup.pop(key)
        assert [r["id"] for r in found] == [rows[i]["id"] for i in roles["test"]]
        assert [float(r["target"]) for r in found] == [rows[i]["target"] for i in roles["test"]]
        assert checksum(record["state_path"]) == record["state_sha256"]
        if record["model"].endswith("_ridge"):
            state = dict(np.load(record["state_path"]))
            matrix = structural if record["model"] == "structural_ridge" else probe
            prediction = ridge_predict(matrix, state)[0, roles["test"]]
        else:
            model = make_model(next(iter(groups.values())), record["model"], config["training"])
            model.load_state_dict(torch.load(record["state_path"], weights_only=True))
            model.eval()
            values = {}
            with torch.no_grad():
                for group in groups.values():
                    local = np.flatnonzero(np.isin(group["indices"], roles["test"]))
                    for start in range(0, len(local), config["training"]["batch_size"]):
                        ix = local[start : start + config["training"]["batch_size"]]
                        pred = (
                            model(
                                torch.tensor(group["dynamic"][ix]),
                                topology(group, record["model"], config["controls_seed"]),
                            )
                            .numpy()
                            .astype(float)
                        )
                        pred = (
                            pred * record["selection"]["target_scale"]
                            + record["selection"]["target_mean"]
                        )
                        values.update(zip(group["indices"][ix], pred, strict=True))
            prediction = np.array([values[i] for i in roles["test"]])
        recomputed = (
            scores(rows, roles["test"], prediction)["per_protein"]
            if cohort == "native"
            else per_protein(rows, roles["test"], prediction)
        )
        assert recomputed == record["per_protein"]
        error = float(np.max(abs(prediction - np.array([float(r["prediction"]) for r in found]))))
        maximum = max(error, maximum)
        assert error == 0, (key, error)
    assert not lookup
    encoder = FrozenProteinMPNN(config["pretrained"])
    invariance = []
    q = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for cohort, (rows, _, groups, _, _) in cohorts.items():
        source = json.loads(
            Path(config["m12_config" if cohort == "family" else "m6_config"]["path"]).read_text()
        )
        for pid, group in groups.items():
            graph = group["graph"]
            spec = next(p for p in source["proteins"] if p["protein_id"] == pid)
            atoms, _ = mapped_atoms(spec["structure_artifacts"], graph)
            moved = [{k: v @ q + [15, -9, 23] for k, v in a.items()} for a in atoms]
            np.testing.assert_allclose(
                annotate(graph, atoms), annotate(graph, moved), rtol=0, atol=1e-6
            )
            pos = [n["polymer_position"] - 1 for n in graph["nodes"]]
            h1, h2 = encoder.encode(atoms, pos), encoder.encode(moved, pos)
            encoder_error = float(abs(h1 - h2).max())
            assert encoder_error < 1e-3, (pid, encoder_error)
            for i in group["indices"]:
                r = rows[i]
                a = variant_inputs(r["mutations"], spec["reference"]["sequence"], graph)
                b = variant_inputs(
                    list(reversed(r["mutations"])), spec["reference"]["sequence"], graph
                )
                assert not a[2] and not b[2]
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
            model = make_model(group, "edge_full", config["training"]).eval()
            top = topology(group, "edge_full", config["controls_seed"])
            perm = torch.tensor(np.random.default_rng(18).permutation(len(graph["nodes"])))
            inverse = torch.argsort(perm)
            static, src, dst, edge, degree = top
            changed = (static[perm], inverse[src], inverse[dst], edge, degree[:, perm])
            dynamic = torch.tensor(group["dynamic"][:4])
            with torch.no_grad():
                node_error = float(
                    abs(model(dynamic, top) - model(dynamic[:, perm], changed)).max()
                )
            assert node_error < 1e-5
            invariance.append(
                {
                    "cohort": cohort,
                    "protein": pid,
                    "node_permutation_max_error": node_error,
                    "frozen_rigid_transform_max_error": encoder_error,
                    "mutation_order_variants": len(group["indices"]),
                }
            )
    repeat_evidence = None
    if repeat:
        repeated = read_csv(repeat / "predictions.csv")
        assert saved == repeated, "Fresh complete repeat predictions differ"
        runs2 = json.loads((repeat / "metrics.json").read_text())
        assert len(records) == len(runs2)
        for a, b in zip(records, runs2, strict=True):
            for key in ("cohort", "protocol", "fold", "model", "seed", "selection", "per_protein"):
                assert a[key] == b[key], (key, a["model"])
        repeat_evidence = {
            "fits": len(records),
            "prediction_records": len(saved),
            "max_prediction_error": 0,
            "all_selection_traces_identical": True,
            "seconds": json.loads((repeat / "audit.json").read_text())["seconds"],
        }
    result = {
        "fits_replayed": len(records),
        "fitting_git_revision": audit["git_revision"],
        "post_fit_source_changes_verified_against_git": source_changes,
        "prediction_records": len(saved),
        "max_prediction_error": maximum,
        "historical_files_unchanged": len(prereg),
        "roles_identical_to_original_m12_m6": True,
        "extra_exclusions": 0,
        "real_structure_invariances": invariance,
        "fresh_repeat": repeat_evidence,
    }
    write_json(root / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("benchmarks/m14"))
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    verify(args.directory, args.repeat)
