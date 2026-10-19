"""Blinded geometry diagnostics do not weaken structure admission or type confidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("audit_m184_gfp", "scripts/audit_m184_gfp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_rigid_fit_preserves_handedness_and_coordinate_correspondence():
    target = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mobile = target @ rotation + [4, 5, 6]
    fitted, center, target_center = module.align_coordinates(mobile, target)
    np.testing.assert_allclose((mobile - center) @ fitted + target_center, target, atol=1e-12)
    assert np.linalg.det(fitted) == pytest.approx(1)
    mirrored = target * [-1, 1, 1]
    fitted, center, target_center = module.align_coordinates(mirrored, target)
    assert np.linalg.det(fitted) == pytest.approx(1)
    assert not np.allclose((mirrored - center) @ fitted + target_center, target)


@pytest.mark.parametrize(
    "coordinates", [np.zeros((2, 3)), np.zeros((4, 2)), np.full((4, 3), np.nan)]
)
def test_fit_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError):
        module.align_coordinates(coordinates, coordinates)


def test_diagnostic_rejects_label_loader_and_guessed_confidence(tmp_path):
    for config in [
        {"dataset_variants": {"config": "DO_NOT_OPEN"}, "structure": {"confidence": "unknown"}},
        {"structure": {"confidence": "experimental_bfactor"}},
        {"structure": {"confidence": "alphafold_plddt"}},
    ]:
        with pytest.raises(ValueError, match="Genotype-only"):
            module.audit(config, tmp_path / "unused.json")
    assert not list(tmp_path.iterdir())


def test_committed_evidence_keeps_confirmation_blocked():
    spec = importlib.util.spec_from_file_location("verify_m184", "scripts/verify_m184.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify()
    assert not Path("benchmarks/m184/final_manifest.json").exists()
