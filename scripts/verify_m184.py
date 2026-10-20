"""Offline evidence/integrity checks; never open external outcome tables or coordinate caches."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proteinmod.release_integrity import verify_digest


def verify():
    root = Path("benchmarks/m184")
    for name in ("historical_hashes.json", "artifact_hashes.json"):
        for path, expected in json.loads((root / name).read_text()).items():
            verify_digest(path, expected)
    query = json.loads((root / "evidence/GFP-exact-sequence-audit.json").read_text())
    assert query["entities"] == 732 and not query["exact_segment_entities"]
    assert not query["candidate_target_cells_accessed"]
    evidence = json.loads((root / "GFP-geometry-audit.json").read_text())
    assert evidence["strict_M2"]["accepted"]
    assert evidence["strict_M2"]["exact_compatible_positions"] == 237
    assert evidence["model_input_dimensions"] == {
        "static": 34,
        "dynamic": 25,
        "ridge": 270,
        "edge": 29,
    }
    assert evidence["directed_chemical_edges_finite"] == 2220
    assert evidence["validated_genotypes"] == 96 and evidence["mutation_order_exact_invariance"]
    assert len(evidence["common_reference_positions"]) == 227
    assert evidence["graph_audit"]["descriptor_missing_nodes"]["plddt"] == 237
    assert evidence["graph_audit"]["descriptor_missing_nodes"]["bfactor_a2"] == 237
    assert not evidence["candidate_target_cells_accessed"]
    assert evidence["models_fit"] == evidence["confirmation_predictions"] == 0
    qualification = json.loads((root / "qualification.json").read_text())
    assert qualification["admission"] == "BLOCKED"
    assert not qualification["final_manifest_issued"]
    assert not qualification["candidate_target_cells_accessed"]
    assert qualification["confirmation_runs"] == 0
    assert len(qualification["families"]) == 4
    assert qualification["permission_request"] == "draft_only_unsent"
    assert not (root / "final_manifest.json").exists()
    print("M18.4 PASS evidence/integrity; four-family confirmation BLOCKED, outcomes blinded")


if __name__ == "__main__":
    verify()
