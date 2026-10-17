"""Check archived qualification evidence and all historical bytes without downloads."""

import json
import subprocess
from pathlib import Path

from proteinmod.data import checksum
from proteinmod.portability import activation_comparison


def verify(root=Path("benchmarks/m161")):
    for name, digest in json.loads((root / "artifact_hashes.json").read_text()).items():
        assert checksum(root / name) == digest, name
    for name, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        assert checksum(name) == digest, name
    results = json.loads((root / "portability.json").read_text())
    assert all(r["exact"] for r in results["same_device"].values())
    assert all(r["roundoff_screen_pass"] for r in results["features"].values())
    assert all(r["pass"] and r["linear_bound_pass"] for r in results["functional"].values())
    assert not all(r["roundoff_screen_pass"] for r in results["layers"].values())
    assert not results["negative_control"]["roundoff_screen_pass"]
    assert not activation_comparison([1.0], [1.01])["roundoff_screen_pass"]
    base = json.loads(Path("configs/m161.json").read_text())["base"]
    assert subprocess.check_output(["git", "merge-base", "--is-ancestor", base, "HEAD"]) == b""
    print("M16.1 evidence and historical byte preservation PASS")


if __name__ == "__main__":
    verify()
