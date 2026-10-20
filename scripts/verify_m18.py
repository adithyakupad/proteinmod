"""Verify the blocked admission package offline without opening phenotype sources."""

import hashlib
import json
from pathlib import Path


def verify():
    root = Path("benchmarks/m18")
    for manifest in ("artifact_hashes.json", "historical_hashes.json"):
        for name, expected in json.loads((root / manifest).read_text()).items():
            path = root / name if manifest == "artifact_hashes.json" else Path(name)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    q = json.loads((root / "qualification.json").read_text())
    assert q["admission"] == "BLOCKED" and not q["targets_accessed"]
    assert q["models_fit"] == q["predictions_generated"] == 0
    index = json.loads((root / "evidence/blind-index.json").read_text())
    assert not index["target_fields_accessed"]
    rows = index["selected"]
    assert len(rows) == len({r["id"] for r in rows}) == 192
    assert all(r["role"] == "test" and "target" not in r for r in rows)
    audit = json.loads((root / "evidence/reference-overlap.json").read_text())
    assert len(audit["pairs"]) == 36
    assert sum(p["review_flag"] for p in audit["pairs"]) == 4
    print("M18: blocked admission verified; 192 blinded candidates, 36 pairs, 4 review flags")


if __name__ == "__main__":
    verify()
