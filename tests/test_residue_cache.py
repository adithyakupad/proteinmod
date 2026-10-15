"""Residue caches validate content and ordered sequence identity without network access."""

import numpy as np
import pytest

from proteinmod.residue_cache import residue_cache


class Encoder:
    def __init__(self):
        self.info = {"frozen": True, "device": "fake"}

    def encode(self, sequences):
        return [np.full((len(s), 3), ord(s[0]), dtype=np.float32) for s in sequences]


def test_residue_cache_contract_content_and_order(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("source")
    spec = {"reference": {"sequence": "AC"}, "dataset": {"path": str(raw)}}
    rows = [{"id": "a", "sequence": "AC"}, {"id": "b", "sequence": "VC"}]
    settings = {"dimension": 3, "batch_size": 2}
    tokens, manifest, directory = residue_cache(
        spec, rows, settings, tmp_path / "cache", True, Encoder()
    )
    assert tokens.shape == (2, 2, 3) and tokens[1, 0, 0] == ord("V")
    assert manifest["encoder"]["frozen"]
    with pytest.raises(ValueError, match="Missing"):
        residue_cache(spec, rows[::-1], settings, tmp_path / "cache")
    with open(directory / "residues.npy", "ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        residue_cache(spec, rows, settings, tmp_path / "cache")
