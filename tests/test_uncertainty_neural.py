"""Selected GNN/attention inference preserves roles, tensors and serialization exactly."""

import io

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from test_hybrid_experiments import fixture

from proteinmod.hybrid_experiments import fit
from proteinmod.multiprotein_data import protocols
from proteinmod.uncertainty_models import predict_hybrid


def test_full_role_inference_and_exact_neural_serialization():
    rows, groups = fixture()
    for r in rows:
        j = int(r["id"].split(":")[-1])
        r["partition"] = "train" if j < 3 else "validation" if j < 5 else "test"
    roles = protocols(rows, ["a", "b", "c"])[0][2]
    calibration = roles["train"][:1]
    roles = {**roles, "train": roles["train"][1:], "calibration": calibration}
    training = {k: v for k, v in roles.items() if k != "calibration"}
    settings = {
        "device": "cpu",
        "dimension": 4,
        "heads": 2,
        "layers": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 3,
        "cpu_threads": 1,
    }
    for mode in ("structure_only", "weighted_fusion", "bidirectional"):
        pred, _, state = fit(groups, rows, training, settings, 0, mode)
        full = predict_hybrid(groups, state, settings, mode, len(rows), roles)
        np.testing.assert_array_equal(pred, full[roles["test"]])
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        other = torch.load(buffer, weights_only=True)
        np.testing.assert_array_equal(
            full, predict_hybrid(groups, other, settings, mode, len(rows), roles)
        )
