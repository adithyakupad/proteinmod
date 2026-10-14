import copy
import json

import numpy as np
import pytest

from proteinmod.embeddings import (
    cache_location,
    contract,
    extract,
    load_features,
    residue_tokens,
    site_features,
)


class MockEncoder:
    def __init__(self):
        self.info = {"device": "mock", "frozen": True}

    def encode(self, sequences):
        return [
            np.array(
                [
                    [ord(letter), position, len(sequence)]
                    for position, letter in enumerate(sequence)
                ],
                dtype=np.float32,
            )
            for sequence in sequences
        ]


def test_sites_symmetric_and_multi_mutant():
    wt = np.arange(12).reshape(4, 3).astype(np.float32)
    variant = wt + 2
    expected = np.concatenate([variant[[0, 3]].mean(axis=0), [2, 2, 2], [2]])
    np.testing.assert_array_equal(site_features(variant, wt, [3, 0]), expected)
    np.testing.assert_array_equal(site_features(variant, wt, [0, 3, 0]), expected)
    np.testing.assert_array_equal(site_features(wt, wt, []), np.zeros(7))
    with pytest.raises(ValueError, match="outside"):
        site_features(variant, wt, [4])


def test_special_padding_alignment():
    class Tokenizer:
        def convert_tokens_to_ids(self, sequence):
            return [ord(letter) for letter in sequence]

    ids = np.array([[0, 65, 67, 2, 1], [0, 65, 67, 68, 2]])
    attention = np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    special = np.array([[1, 0, 0, 1, 1], [1, 0, 0, 0, 1]])
    hidden = np.array([[[999], [1], [3], [999], [999]], [[999], [2], [4], [6], [999]]])
    residues = residue_tokens(hidden, ids, attention, special, ["AC", "ACD"], Tokenizer())
    assert [matrix.mean() for matrix in residues] == [2, 4]
    with pytest.raises(ValueError, match="alignment"):
        residue_tokens(hidden, ids, attention, special, ["AD", "ACD"], Tokenizer())
    with pytest.raises(ValueError, match="truncation"):
        residue_tokens(hidden, ids, attention, special, ["ACD", "ACD"], Tokenizer())


def test_extraction_cache_and_identity(embedding_config):
    sequences, reference = ["ACDE", "CCDF", "ACDF"], "ACDE"
    directory = extract(embedding_config, sequences, reference, "dataset", encoder=MockEncoder())
    _, identity = cache_location(embedding_config, sequences, reference, "dataset")
    features, metadata = load_features(directory, identity, 3, 3)
    expected = MockEncoder().encode(sequences)
    np.testing.assert_allclose(features["esm_mean"], [row.mean(axis=0) for row in expected])
    np.testing.assert_allclose(features["esm_delta"][0], 0)
    np.testing.assert_allclose(features["esm_sites"][0], 0)
    assert features["esm_sites"][1, -1] == 2
    assert metadata["label_free"] is True
    # A cache hit never loads an encoder or downloads anything.
    assert extract(embedding_config, sequences, reference, "dataset") == directory
    for change in ("revision", "batch_size", "device"):
        changed = copy.deepcopy(embedding_config)
        changed["embedding"][change] = {"revision": "b" * 40, "batch_size": 1, "device": "mps"}[
            change
        ]
        assert cache_location(changed, sequences, reference, "dataset")[0] != directory
    assert cache_location(embedding_config, sequences[::-1], reference, "dataset")[0] != directory
    assert cache_location(embedding_config, sequences, reference, "other dataset")[0] != directory
    (directory / "features.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_features(directory, identity, 3, 3)


def test_contract_rejects_mutable_revision_and_length(embedding_config):
    settings = embedding_config["embedding"]
    settings["revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        contract("data", settings, ["ACDE"], "ACDE")
    settings["revision"] = "a" * 40
    with pytest.raises(ValueError, match="length"):
        contract("data", settings, ["ACD"], "ACDE")


def test_bad_encoder_shape(embedding_config):
    class BrokenEncoder(MockEncoder):
        def encode(self, sequences):
            return [np.zeros((1, 3)) for _ in sequences]

    with pytest.raises(ValueError, match="WT encoder shape"):
        extract(embedding_config, ["ACDE"], "ACDE", "data", encoder=BrokenEncoder())


def test_cache_metadata_mismatch(embedding_config):
    directory = extract(embedding_config, ["ACDE"], "ACDE", "data", encoder=MockEncoder())
    _, identity = cache_location(embedding_config, ["ACDE"], "ACDE", "data")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["contract"]["schema"] = "wrong"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        load_features(directory, identity, 1, 3)


def test_frozen_model_initialization_without_weights(embedding_config, monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from proteinmod.embeddings import FrozenESM

    torch = MagicMock()
    torch.backends.mps.is_available.return_value = False
    torch.cuda.is_available.return_value = False
    transformers = MagicMock()
    model = transformers.EsmModel.from_pretrained.return_value
    model.config = SimpleNamespace(hidden_size=3, max_position_embeddings=1026)
    model.parameters.return_value = [SimpleNamespace(numel=lambda: 123)]
    transformers.AutoTokenizer.from_pretrained.return_value.get_vocab.return_value = {"A": 1}
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr("proteinmod.embeddings.importlib.metadata.version", lambda name: "fixture")
    encoder = FrozenESM(embedding_config["embedding"], local_only=True)
    model.requires_grad_.assert_called_once_with(False)
    model.eval.assert_called_once()
    torch.use_deterministic_algorithms.assert_called_once_with(True)
    assert encoder.device == "cpu"
    kwargs = transformers.EsmModel.from_pretrained.call_args.kwargs
    assert kwargs["revision"] == "a" * 40
    assert kwargs["local_files_only"] is True
    assert kwargs["use_safetensors"] is True
    with pytest.raises(ValueError, match="no truncation"):
        encoder.encode(["A" * 1025])
    embedding_config["embedding"]["device"] = "cuda"
    with pytest.raises(ValueError, match="unavailable"):
        FrozenESM(embedding_config["embedding"])
