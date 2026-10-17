"""Refit M12/M13 with unchanged preregistered paths in a fresh canonical workspace."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def refit(milestone):
    if milestone not in (12, 13):
        raise ValueError("Only M12/M13 require the canonical output contract")
    source = Path.cwd()
    workspace = source / f"runs/m16-canonical-m{milestone}"
    output = source / f"runs/m16-m{milestone}"
    if workspace.exists() or output.exists():
        raise ValueError("Existing qualification workspace/output; preserve it")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(workspace)],
        check=True,
    )
    subprocess.run(["git", "-C", str(workspace), "checkout", "--detach", revision], check=True)
    # Copy only required acquisition/activation caches, never historical trained weights.
    for name in ("m12", "m6") if milestone == 13 else ("m12",):
        shutil.copytree(source / "data/cache" / name, workspace / "data/cache" / name)
    checkpoints = workspace / "data/cache/m12/checkpoints"
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    module = "family" if milestone == 12 else "uncertainty"
    subprocess.run(
        [sys.executable, "-m", f"proteinmod.{module}_experiments"],
        cwd=workspace,
        env={**os.environ, "PYTHONPATH": str(workspace / "src")},
        check=True,
    )
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=workspace).strip()
    output.symlink_to(workspace / f"runs/m{milestone}", target_is_directory=True)
    (workspace / "runs/qualification.json").write_text(
        json.dumps(
            {
                "milestone": milestone,
                "source_commit": revision,
                "unchanged_canonical_config": True,
                "fresh_supervised_checkpoints": True,
                "acquisition_and_activation_caches_reused": True,
                "source_checkout_clean_after_fit": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("milestone", type=int, choices=(12, 13))
    refit(parser.parse_args().milestone)
