"""Freeze small qualification evidence; preserve all prior milestone bytes."""

import json
import shutil
import subprocess
from pathlib import Path

from proteinmod.data import checksum, write_json


def archive():
    source, target = Path("runs/m16.1"), Path("benchmarks/m161")
    if (target / "artifact_hashes.json").exists():
        raise ValueError("Preserve existing archived qualification")
    for name in (
        "portability.json",
        "acquisition.json",
        "license-inventory.json",
        "pr32-verification.json",
        "source-preservation.json",
    ):
        shutil.copyfile(source / name, target / name)
    for device in ("cpu", "mps"):
        for name in (
            "test_predictions.csv",
            "metrics.json",
            "learning_curve_summary.csv",
            "subset_ids.json",
        ):
            shutil.copyfile(source / device / name, target / (device + "-" + name))
    shutil.copyfile(source / "sources/cold-sources.json", target / "cold-sources.json")
    for name in ("commands.json", "corrected.json", "remaining.json"):
        shutil.copyfile(source / "replay" / name, target / ("replay-" + name))
    for n in (10, 11, 13, 14, 15):
        shutil.copyfile(source / "replay" / f"m{n}-corrected.log", target / f"m{n}-replay.log")
    for n in (10, 11):
        shutil.copyfile(
            source / "replay" / f"m{n}-replay.log", target / f"initial-m{n}-failure.log"
        )
    dependencies = (source / "requirements.txt").read_text().splitlines()
    dependencies = [
        "-e ." if line.startswith("-e ") else line
        for line in dependencies
        if not line.startswith("#")
    ]
    (target / "requirements.txt").write_text("\n".join(dependencies) + "\n")
    base = json.loads(Path("configs/m161.json").read_text())["base"]
    old_files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", base, "benchmarks", "configs"], text=True
    ).splitlines()
    historical = {}
    import hashlib

    for name in old_files:
        old = subprocess.check_output(["git", "show", f"{base}:{name}"])
        digest = hashlib.sha256(old).hexdigest()
        assert checksum(name) == digest, name
        historical[name] = digest
    write_json(target / "historical_hashes.json", historical)
    safe = json.loads((source / "safe-stability.json").read_text())
    assert all(
        p["prediction"] is None and p["unit"] is None and not p["thermodynamic_eligible"]
        for p in safe["predictions"]
    )
    write_json(
        target / "safe-stability-audit.json",
        {
            "source_sha256": safe["source_sha256"],
            "rows": len(safe["predictions"]),
            "all_ineligible": True,
            "export_sha256": checksum(source / "safe-stability.json"),
            "raw_historical_values_changed": False,
        },
    )
    write_json(
        target / "sources.json",
        {
            str(p): checksum(p)
            for folder in ("scripts", "src/proteinmod", "configs")
            for p in Path(folder).glob("*")
            if p.is_file()
            and (
                "m16_1" in p.name
                or p.name in ("m161.json", "portability.py", "stability_inference.py")
            )
        },
    )
    write_json(
        target / "artifact_hashes.json",
        {
            p.name: checksum(p)
            for p in sorted(target.iterdir())
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(f"Preserved {len(historical)} historical files")


if __name__ == "__main__":
    archive()
