"""Stdlib-only immutable evidence checks; never open external outcome tables."""

import hashlib
import json
from pathlib import Path


def verify():
    root = Path("benchmarks/m183")
    for name in ("historical_hashes.json", "artifact_hashes.json"):
        for file, expected in json.loads((root / name).read_text()).items():
            path = Path(file) if name == "historical_hashes.json" else root / file
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    data = json.loads((root / "structural-cohort-audit.json").read_text())
    assert data["common_candidates"] == 384
    assert not data["candidate_target_cells_accessed"]
    assert data["models_fit"] == data["predictions_generated"] == 0
    masks = {
        "P0A9X9": list(range(1, 66)),
        "P01053": list(range(1, 64)),
        "TEM1": list(range(24, 287)),
        "GFP": [p for p in range(1, 238) if p not in {63, 79, *range(230, 238)}],
    }
    assert {r["protein_id"] for r in data["families"]} == set(masks)
    seen = set()
    parent = json.loads(Path("benchmarks/m182/evidence/structure-mask-audits.json").read_text())
    for row in data["families"]:
        pid = row["protein_id"]
        assert row["eligible_positions"] == masks[pid]
        assert row["scope"] == "masked-reference-v1" and row["masked_accepted"]
        assert row["eligible_identity"] == row["eligible_CA_coverage"] == 1.0
        assert row["nodes"] == len(masks[pid]) and row["connected_components"] == 1
        assert row["isolated_nodes"] == 0
        assert (
            row["independent_spatial_edges_verified"] and row["raw_atom_site_coordinates_verified"]
        )
        assert (
            row["mutation_order_invariance"]
            and row["confidence_semantics"] == "experimental_bfactor"
        )
        assert row["descriptor_missing_nodes"] == {
            "rsa": 0,
            "contact_degree": 0,
            "nonlocal_contact_degree": 0,
            "bfactor_a2": 0,
            "plddt": row["nodes"],
        }
        assert row["model_input_dimensions"] == {
            "static": 34,
            "dynamic": 25,
            "ridge": 270,
            "edge": 29,
        }
        assert len(row["common_candidate_ids"]) == row["matched_candidates"] == 96
        assert not seen.intersection(row["common_candidate_ids"])
        seen.update(row["common_candidate_ids"])
        if pid in ("TEM1", "GFP"):
            assert row["full_reference_M2_audit"] == next(
                r["audit"] for r in parent if r["pdb"] == row["pdb"]
            )
            assert not row["full_reference_M2_audit"]["accepted"]
    q = json.loads((root / "qualification.json").read_text())
    assert q["admission"] == "BLOCKED" and len(q["families"]) == 4
    assert not q["final_manifest_issued"] and not q["candidate_target_cells_accessed"]
    assert q["models_fit"] == q["predictions_generated"] == 0 and len(q["blockers"]) == 2
    license_audit = json.loads((root / "evidence/GEO-Zenodo-item-audit.json").read_text())
    assert license_audit["archive_directory_entries"] == 3638
    assert (
        not license_audit["equal_size_and_crc_members"]
        and not license_audit["raw_filename_members"]
    )
    assert (
        not license_audit["outcome_values_decoded"]
        and not license_audit["archive_whole_MD5_verified"]
    )
    replay = json.loads((root / "replay.json").read_text())
    assert replay["families_exact"] == 4 and replay["max_tensor_hash_differences"] == 0
    print(
        "M18.3 PASS engineering: four masked graphs, 384 matched blinded candidates, original M2 rejections preserved; confirmation BLOCKED"
    )


if __name__ == "__main__":
    verify()
