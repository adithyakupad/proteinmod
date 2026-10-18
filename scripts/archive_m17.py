"""Archive small M17 results and verify immutable baseline blobs; no weights/data."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from proteinmod.data import checksum, write_json


def archive():
    root = Path("benchmarks/m17")
    assert not (root / "artifact_hashes.json").exists(), "Preserve archived evidence"
    base = json.loads(Path("configs/m17.json").read_text())["base"]
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", base, "benchmarks", "configs"], text=True
    ).splitlines()
    historical = {}
    for p in names:
        old = subprocess.check_output(["git", "show", base + ":" + p])
        historical[p] = hashlib.sha256(old).hexdigest()
        assert checksum(p) == historical[p], p
    write_json(root / "historical_hashes.json", historical)
    for p in Path("runs/m17").iterdir():
        if p.is_file() and p.suffix in (".json", ".csv", ".png", ".log"):
            shutil.copyfile(p, root / p.name)
    write_json(
        root / "source-qualification.json",
        {
            "base": base,
            "historical_files_exact": len(historical),
            "external_confirmation": "BLOCKED, no external labels used in any fit",
            "sources": {
                str(p): checksum(p)
                for p in [
                    *Path("src/proteinmod").glob("mechanism*.py"),
                    *Path("scripts").glob("*m17.py"),
                ]
            },
        },
    )
    write_json(
        root / "artifact_hashes.json",
        {
            p.name: checksum(p)
            for p in sorted(root.iterdir())
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print("Historical benchmark/config bytes preserved:", len(historical))


if __name__ == "__main__":
    archive()
