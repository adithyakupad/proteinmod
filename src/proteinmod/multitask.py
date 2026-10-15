"""M7: two heads, differentiable M6 mapping/readout, explicit unpaired supervision."""

import numpy as np
import torch
from torch import nn

from proteinmod.multiprotein_experiments import fit_targets
from proteinmod.stability_model import StabilityModel

TASKS = ("function", "stability")


class MultitaskModel(nn.Module):
    def __init__(self, graph, embedding_dimension=480, dimension=16, heads=2, layers=2):
        super().__init__()
        self.encoder = StabilityModel(graph, embedding_dimension, dimension, heads, layers)
        self.encoder.stability_head = nn.Identity()
        self.encoder.backbone.requires_grad_(True)
        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(6 * dimension + 4, dimension), nn.ReLU(), nn.Linear(dimension, 1)
                )
                for task in TASKS
            }
        )

    def forward(self, *args, **kwargs):
        # Bypass only M6's no-grad decorator; reuse its mapping and missingness validation.
        h = StabilityModel.represent.__wrapped__(self.encoder, *args, **kwargs)
        return torch.stack([self.heads[t](h).squeeze(-1) for t in TASKS], -1)


def masked_loss(prediction, target, observed, weights=(1.0, 1.0)):
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Task tensor dimensions disagree")
    if observed.dtype != torch.bool or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("Two explicit boolean task masks required")
    if not observed.any():
        raise ValueError("All supervised outcomes are missing")
    w = torch.as_tensor(weights, dtype=prediction.dtype, device=prediction.device)
    if w.shape != (2,) or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Positive finite task weights required")
    losses, active = [], []
    for j in range(2):
        mask = observed[:, j]
        if mask.any():
            # Index before subtraction: absent NaN labels must have exactly zero gradient.
            delta = prediction[mask, j] - target[mask, j]
            if not torch.isfinite(delta).all():
                raise ValueError("Nonfinite observed task outcome/prediction")
            losses.append(delta.square().mean() * w[j])
            active.append(w[j])
    return torch.stack(losses).sum() / torch.stack(active).sum()


def calibrate(rows, train, task):
    if task not in TASKS:
        raise ValueError("Unknown task")
    y = fit_targets(rows, train) if task == "function" else np.full(len(rows), np.nan)
    if task == "stability":
        for i in train:
            r = rows[i]
            if (r["target_unit"], r["target_convention"]) != ("kcal/mol", "positive_destabilizing"):
                raise ValueError("Stability target units/sign mismatch")
            y[i] = r["target"]
    mean, scale = float(y[train].mean()), float(y[train].std())
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError("Nonconstant finite training targets required")
    return (y - mean) / scale, {"mean": mean, "scale": scale}
