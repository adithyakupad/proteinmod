"""Stdlib-only immutable-evidence checks; no access to target-bearing sources."""

import hashlib
import json
from pathlib import Path


def verify():
    root = Path("benchmarks/m181")
    for manifest in ("artifact_hashes.json", "historical_hashes.json"):
        for name, expected in json.loads((root / manifest).read_text()).items():
            path = root / name if manifest == "artifact_hashes.json" else Path(name)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    alignment = json.loads((root / "alignment-diagnostics.json").read_text())
    assert [(r["a"], r["b"]) for r in alignment["pairs"]] == [
        ("P0A9X9", "BBC1"),
        ("P01053", "BBC1"),
        ("P01053", "1LP1"),
        ("P01053", "1A32"),
    ]
    for row in alignment["pairs"]:
        assert row["original_flag"] and row["global"]["ambiguous_optimum"]
        assert 3 <= row["local"]["aligned"] <= 5
    profiles = json.loads((root / "reciprocal-profiles.json").read_text())
    assert profiles["version"] == "0.12.1" and len(profiles["pairs"]) == 25
    own = {
        "P0A9X9": "PF00313",
        "P01053": "PF00280",
        "BBC1": "PF00018",
        "1LP1": "PF02216",
        "1A32": "PF00312",
    }
    assert len({(r["profile"], r["sequence"]) for r in profiles["pairs"]}) == 25
    for row in profiles["pairs"]:
        path = root / "evidence/profiles" / (row["profile"] + ".hmm")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["profile_sha256"]
        passes = (
            row["score"] is not None
            and row["score"] >= row["GA"][0]
            and any(d["score"] >= row["GA"][1] for d in row["domains"])
        )
        assert row["passes_GA"] == passes == (own[row["sequence"]] == row["profile"])
    mapping = json.loads((root / "structure-profile-mapping.json").read_text())
    for pid, row in mapping.items():
        entity = json.loads((root / "evidence" / (row["pdb"] + "-entity.json")).read_text())
        seq = entity["entity_poly"]["pdbx_seq_one_letter_code_can"]
        start, end = row["reference_start_entity_1based"], row["reference_end_entity_1based"]
        assert seq[start - 1 : end] == profiles["references"][pid]
        for domain in row["CATH"].values():
            assert any(
                m["entity_id"] == 1
                and m["chain_id"] == "A"
                and m["start"]["residue_number"] <= start
                and m["end"]["residue_number"] >= end
                for m in domain["mappings"]
            )
    tem = json.loads((root / "TEM1-qualification.json").read_text())
    assert not tem["target_fields_accessed"] and tem["metadata_rows"] == 5740
    assert tem["single_nonWT_canonical_rows"] == 5434
    assert len(tem["historical_pairs"]) == 18 and not any(
        p["original_flag"] for p in tem["historical_pairs"]
    )
    assert [m["canonical_position"] for m in tem["structure_mismatches"]] == [82, 182]
    q = json.loads((root / "qualification.json").read_text())
    assert q["admission"] == "BLOCKED" and len(q["potential_families"]) == 3
    assert q["models_fit"] == q["predictions_generated"] == 0
    assert not q["confirmatory_candidate_target_cells_accessed"]
    print(
        "M18.1 PASS evidence checks: four retained flags; 25 reciprocal profiles; TEM1 genotype lineage; confirmation BLOCKED"
    )


if __name__ == "__main__":
    verify()
