import pytest


@pytest.fixture
def embedding_config(tmp_path):
    return {
        "embedding": {
            "model": "mock",
            "revision": "a" * 40,
            "dimension": 3,
            "device": "cpu",
            "batch_size": 2,
            "cache": str(tmp_path / "cache"),
            "model_cache": str(tmp_path / "models"),
        }
    }
