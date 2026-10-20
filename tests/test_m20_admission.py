"""M20 stop-state and frozen evidence checks; no model/data downloads or outcomes."""

import hashlib
import json
from pathlib import Path


def test_m20_stays_blinded_and_preserves_frozen_history():
    root = Path("benchmarks/m20")
    for path, digest in json.loads((root / "historical_hashes.json").read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    plan = json.loads((root / "admission-plan.json").read_text())
    for path, digest in plan["frozen_inputs"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    status = json.loads((root / "qualification.json").read_text())
    assert status["status"] == "BLOCKED" and len(status["families"]) == 4
    assert len({r["family"] for r in status["families"]}) == 4
    assert status["directional_gate"] == ">=3/4" and status["effect_margin_rho"] == 0.02
    assert not status["final_evaluation_manifest_issued"]
    assert (
        not status["outcome_labels_accessed"]
        and not status["aggregate_distribution_values_accessed"]
    )
    for field in [
        "external_models_fit",
        "external_predictions",
        "external_confirmations",
        "contacts_sent",
    ]:
        assert status[field] == 0
    for hypothesis in status["hypotheses"].values():
        assert hypothesis["status"] == "NOT_TESTED"
        assert hypothesis["per_family_effects"] is hypothesis["per_family_uncertainty"] is None
    assert not (root / "final_manifest.json").exists()
    assert not (root / "predictions.csv").exists()


def test_m20_incident_and_permission_evidence_are_not_self_certification():
    root = Path("benchmarks/m20")
    review = json.loads((root / "incident-review.json").read_text())
    assert not review["reviewer"]["selected_sod1_candidates"]
    assert not review["reviewer"]["external_human_independent_auditor"]
    assert not review["pristine_blinded_replication_allowed"]
    assert not review["dataset_admission_granted"] and not review["external_run_authorized"]
    assert not review["decision_logic_consumption"]["historical_non_consumption_proved"]
    for evidence in review["evidence"]:
        assert hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest() == evidence["sha256"]
    request = json.loads((root / "permission-review.json").read_text())["request"]
    assert not request["sent"] and not request["owner_send_authorization"]
    assert (root / "PERMISSION_REQUEST.md").read_bytes() == Path(
        request["identical_to"]
    ).read_bytes()
