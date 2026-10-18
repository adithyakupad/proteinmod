"""Negative controls protect the new numeric policy and fail-closed physical output."""

import numpy as np
import pytest

from proteinmod.portability import activation_comparison
from proteinmod.stability_inference import export, physical_stability


def test_numeric_screen():
    a = np.ones((4, 8), dtype=np.float32)
    assert activation_comparison(a, a)["exact"]
    close = activation_comparison(a, a + 2e-6)
    assert close["roundoff_screen_pass"] and not close["legacy_1e-6_pass"]
    assert not activation_comparison(a, a + 0.01)["roundoff_screen_pass"]
    for b in (np.ones(8), a * np.nan, a * np.inf):
        with pytest.raises(ValueError):
            activation_comparison(a, b)
    with pytest.raises(ValueError):
        activation_comparison([], [])


def test_physical_inference_fails_closed(tmp_path):
    row = {
        "id": "x",
        "protein_id": "amylase",
        "source": "esm_sites_ridge",
        "source_weights_sha256": "abc",
        "domain": "out_of_distribution",
    }
    for value in (-1372636.75, 0.2, float("nan"), float("inf")):
        result = physical_stability({**row, "predicted_ddg_kcal_mol": value})
        assert result["prediction"] is None and result["unit"] is None
        assert result["thermodynamic_eligible"] is False
    # A caller cannot promote itself into the physical domain by changing flags.
    assert physical_stability({**row, "domain": "in_domain"})["prediction"] is None
    with pytest.raises(ValueError):
        physical_stability({"id": "x"})
    source = tmp_path / "raw.csv"
    source.write_text("fabricated")
    with pytest.raises(ValueError, match="checksum"):
        export(source, tmp_path / "safe.json")
