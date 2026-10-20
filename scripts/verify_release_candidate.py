"""Closeout integrity and optional checkpoint replay; never acquire data or fit models."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proteinmod.release_integrity import AMENDABLE, verify_digest

BASE = "14fa4c37fbafda9fbc4bd7f5be482f632062c9d1"


def verify():
    files = subprocess.check_output(["git", "ls-tree", "-r", BASE], text=True).splitlines()
    preserved = 0
    for line in files:
        metadata, path = line.split("\t")
        blob = metadata.split()[2]
        original = subprocess.check_output(["git", "cat-file", "blob", blob])
        verify_digest(path, hashlib.sha256(original).hexdigest())
        preserved += path not in AMENDABLE
    amendments = json.loads(Path("release_candidate/amendments.json").read_text())
    assert set(amendments) == AMENDABLE
    for path, entry in amendments.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == entry["after_sha256"]
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    forbidden = {".pt", ".pth", ".safetensors", ".npy", ".npz", ".zip", ".gz", ".xlsx"}
    assert not [p for p in tracked if Path(p).suffix in forbidden]
    return {"base": BASE, "unamended_original_files": preserved, "amendments": sorted(AMENDABLE)}


def replay():
    import numpy as np

    from proteinmod.data import checksum
    from proteinmod.family_experiments import inputs
    from proteinmod.ridge_geometry import predict

    rows, matrices, mask, _, _ = inputs(json.loads(Path("configs/m12.json").read_text()))
    assert mask.all()
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    cases = {}
    with Path("benchmarks/m19/predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = row["fold"], float(row["fraction"]), int(row["seed"]), row["model"]
            cases.setdefault(key, []).append(row)
    maximum = 0.0
    for record in json.loads(Path("benchmarks/m19/runs.json").read_text()):
        assert checksum(record["state"]) == record["state_sha256"]
        key = record["fold"], record["fraction"], record["seed"], record["model"]
        saved = cases[key]
        with np.load(record["state"]) as state:
            actual = predict(matrices["esm_sites"], state)[[lookup[r["id"]] for r in saved]]
        expected = [float(r["prediction"]) for r in saved]
        np.testing.assert_array_equal(actual, expected)
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    return {
        "records": len(cases),
        "prediction_rows": sum(map(len, cases.values())),
        "max_error": maximum,
        "fits": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--replay-m19", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify()
    if args.replay_m19:
        result["m19"] = replay()
    with Path(args.output).open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result))
