"""Stdlib-only immutable history and blinded-candidate evidence checks; no raw data reads."""

import hashlib
import json
from pathlib import Path


def verify():
    root = Path("benchmarks/m185")
    for name in ["historical_hashes.json", "artifact_hashes.json"]:
        for path, expected in json.loads((root / name).read_text()).items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    metadata = json.loads((root / "evidence/abundance-metadata.json").read_text())
    assert "scoreDistribution" not in metadata and "scoreCalibrations" not in metadata
    assert (
        metadata["urn"] == "urn:mavedb:00001217-a-3" and metadata["license"]["shortName"] == "CC0"
    )
    assert metadata["numVariants"] == 2906
    supplement = json.loads((root / "evidence/DataS2-headers.json").read_text())
    assert supplement["sheet_data_rows_parsed"] == 0
    sheet = next(s for s in supplement["sheets"] if s["sheet"] == "abundance_map_scores_LLRps")
    assert [c["header"] for c in sheet["first_row"]] == ["hgvs_pro"] + metadata["datasetColumns"][
        "scoreColumns"
    ]
    audit = json.loads((root / "structural-genotype-audit.json").read_text())
    assert audit["selected_chain"] == "A"
    for row in audit["strict_full_reference"]:
        assert not row["audit"]["accepted"] and row["audit"]["missing_positions"] == [1]
        assert row["audit"]["exact_compatible_positions"] == 153 and not row["audit"]["differences"]
    assert audit["masked"]["eligible_positions"] == list(range(2, 155))
    assert audit["masked"]["nodes"] == 153 and audit["masked"]["undirected_edges"] == 827
    assert audit["masked"]["connected_components"] == 1 and audit["masked"]["isolated_nodes"] == 0
    assert audit["genotypes"] == audit["assay_and_RefSeq_HGVS_agreement"] == 2906
    assert audit["canonical_structurally_eligible"] == 2632
    assert audit["exclusion_counts"] == {"synonymous_or_WT": 135, "stop": 139}
    assert len(audit["selected_candidates"]) == 96
    assert audit["model_input_dimensions"] == {
        "static": 34,
        "dynamic": 25,
        "ridge": 270,
        "edge": 29,
    }
    assert audit["finite_chemical_edge_rows"] == 1654
    assert not audit["variant_outcome_values_accessed"]
    assert audit["models_fit"] == audit["predictions_generated"] == 0
    similarity = json.loads((root / "similarity-audit.json").read_text())
    assert len(similarity["pairs"]) == 22 and not any(
        r["original_flag"] for r in similarity["pairs"]
    )
    profiles = json.loads((root / "profile-audit.json").read_text())
    assert len(profiles["pairs"]) == 276
    assert [
        (r["profile"], r["sequence"])
        for r in profiles["pairs"]
        if r["passes_GA"] and (r["profile"] == "PF00080" or r["sequence"] == "SOD1")
    ] == [("PF00080", "SOD1")]
    qualification = json.loads((root / "qualification.json").read_text())
    assert qualification["admission"] == "BLOCKED"
    assert qualification["four_family_denominator"] == 4
    assert qualification["aggregate_metadata_mechanically_decoded"]
    assert not qualification["strict_predecode_blinding_preserved"]
    assert not qualification["variant_level_outcomes_accessed"]
    assert qualification["confirmation_runs"] == 0 and not qualification["final_manifest_issued"]
    assert not (root / "final_manifest.json").exists()
    print(
        "SOD1 evidence/history PASS; strict-blinding deviation retained; four-family confirmation BLOCKED"
    )


if __name__ == "__main__":
    verify()
