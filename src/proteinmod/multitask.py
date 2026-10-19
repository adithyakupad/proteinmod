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
        shared_features = StabilityModel.represent.__wrapped__(self.encoder, *args, **kwargs)
        return torch.stack([self.heads[task](shared_features).squeeze(-1) for task in TASKS], -1)


def masked_loss(prediction, target, observed, weights=(1.0, 1.0)):
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("Task tensor dimensions disagree")
    if observed.dtype != torch.bool or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("Two explicit boolean task masks required")
    if not observed.any():
        raise ValueError("All supervised outcomes are missing")
    task_weights = torch.as_tensor(weights, dtype=prediction.dtype, device=prediction.device)
    if (
        task_weights.shape != (2,)
        or not torch.isfinite(task_weights).all()
        or (task_weights <= 0).any()
    ):
        raise ValueError("Positive finite task weights required")
    losses, active = [], []
    for task_index in range(2):
        mask = observed[:, task_index]
        if mask.any():
            # Index before subtraction: absent NaN labels must have exactly zero gradient.
            task_errors = prediction[mask, task_index] - target[mask, task_index]
            if not torch.isfinite(task_errors).all():
                raise ValueError("Nonfinite observed task outcome/prediction")
            losses.append(task_errors.square().mean() * task_weights[task_index])
            active.append(task_weights[task_index])
    return torch.stack(losses).sum() / torch.stack(active).sum()


def calibrate(rows, train, task):
    if task not in TASKS:
        raise ValueError("Unknown task")
    targets = fit_targets(rows, train) if task == "function" else np.full(len(rows), np.nan)
    if task == "stability":
        for i in train:
            observation = rows[i]
            if (observation["target_unit"], observation["target_convention"]) != (
                "kcal/mol",
                "positive_destabilizing",
            ):
                raise ValueError("Stability target units/sign mismatch")
            targets[i] = observation["target"]
    mean, scale = float(targets[train].mean()), float(targets[train].std())
    if not np.isfinite([mean, scale]).all() or scale <= 0:
        raise ValueError("Nonconstant finite training targets required")
    return (targets - mean) / scale, {"mean": mean, "scale": scale}
