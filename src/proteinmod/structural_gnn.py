"""Small shared-topology residue GNN; no graph construction or pretrained encoder."""

import copy
import time

import numpy as np
import torch
from torch import nn

from proteinmod.models import metrics


class ResidueGNN(nn.Module):
    def __init__(self, static, edge_index, edge_features, hidden=16, layers=2):
        super().__init__()
        self.register_buffer("static", torch.as_tensor(static, dtype=torch.float32))
        self.register_buffer("source", torch.as_tensor(edge_index[0], dtype=torch.long))
        self.register_buffer("target", torch.as_tensor(edge_index[1], dtype=torch.long))
        self.register_buffer("edge_features", torch.as_tensor(edge_features, dtype=torch.float32))
        degree = np.bincount(edge_index[1], minlength=len(static))
        self.register_buffer(
            "degree", torch.as_tensor(np.maximum(degree, 1), dtype=torch.float32)[None, :, None]
        )
        self.input = nn.Linear(static.shape[1] + 25, hidden)
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(layers))
        self.edge_gates = nn.ModuleList(nn.Linear(3, hidden) for _ in range(layers))
        self.updates = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def encode(self, dynamic, graph=None):
        static, source, target, edge_features, degree = (
            (self.static, self.source, self.target, self.edge_features, self.degree)
            if graph is None
            else graph
        )
        static = static.unsqueeze(0).expand(len(dynamic), -1, -1)
        node_hidden = torch.relu(self.input(torch.cat([static, dynamic], dim=-1)))
        for message, gate, update in zip(self.messages, self.edge_gates, self.updates, strict=True):
            # Shared topology avoids a graph framework and duplicated batch edge storage.
            edge_messages = (
                message(node_hidden)[:, source] * torch.sigmoid(gate(edge_features))[None]
            )
            aggregate = torch.zeros_like(node_hidden)
            aggregate.index_add_(1, target, edge_messages)
            node_hidden = node_hidden + torch.relu(
                update(torch.cat([node_hidden, aggregate / degree], dim=-1))
            )
        return node_hidden

    def forward(self, dynamic, graph=None):
        node_hidden = self.encode(dynamic, graph)
        mutation_mask = dynamic[:, :, :1]
        mutation_count = mutation_mask.sum(dim=1)
        mutation_sum = (node_hidden * mutation_mask).sum(dim=1)
        return self.head(
            torch.cat(
                [
                    mutation_sum,
                    mutation_sum / mutation_count.clamp(min=1),
                    node_hidden.mean(dim=1),
                    mutation_count,
                ],
                dim=-1,
            )
        ).squeeze(-1)


def train_gnn(graph, dynamic, targets, train, validation, test, settings, seed, layers):
    """Select weight decay and epoch using validation only; return selected train-only fit."""
    started = time.perf_counter()
    if settings["device"] != "cpu":
        raise ValueError(
            "M3 uses deterministic CPU execution; MPS index aggregation is nondeterministic"
        )
    if settings["max_epochs"] < 1 or settings["batch_size"] < 1 or settings["patience"] < 1:
        raise ValueError("Epochs, batch size and patience must be positive")
    if layers not in (0, 2, 3) or settings["hidden"] < 1 or settings["learning_rate"] <= 0:
        raise ValueError("Invalid GNN architecture or learning rate")
    if not settings["weight_decays"] or any(
        w < 0 or not np.isfinite(w) for w in settings["weight_decays"]
    ):
        raise ValueError("Weight decays must be finite and nonnegative")
    if any(len(batch_indices) == 0 for batch_indices in (train, validation, test)) or any(
        set(a) & set(b) for a, b in ((train, validation), (train, test), (validation, test))
    ):
        raise ValueError("GNN partitions must be nonempty and disjoint")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(settings["device"])
    mutation_inputs = torch.as_tensor(dynamic, dtype=torch.float32, device=device)
    target_mean, target_scale = (
        float(np.mean(targets[train])),
        max(float(np.std(targets[train])), 1e-8),
    )
    scaled_targets = torch.as_tensor(
        (targets - target_mean) / target_scale, dtype=torch.float32, device=device
    )
    batch_size = settings["batch_size"]

    def predict(model, indices):
        model.eval()
        with torch.no_grad():
            values = torch.cat(
                [
                    model(mutation_inputs[batch])
                    for batch in np.array_split(
                        indices, max(1, int(np.ceil(len(indices) / batch_size)))
                    )
                ]
            )
        return values.cpu().numpy().astype(np.float64) * target_scale + target_mean

    candidates, winner = [], None
    for decay in sorted(set(settings["weight_decays"])):
        torch.manual_seed(seed)
        random_generator = np.random.default_rng(seed)
        model = ResidueGNN(
            graph["static"], graph["edge_index"], graph["edge_features"], settings["hidden"], layers
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"], weight_decay=decay
        )
        best, stale_epochs, trace = None, 0, []
        for epoch in range(1, settings["max_epochs"] + 1):
            model.train()
            order = random_generator.permutation(train)
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                batch_indices = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(
                    (model(mutation_inputs[batch_indices]) - scaled_targets[batch_indices]) ** 2
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite GNN training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total_loss += loss.item() * len(batch_indices)
            validation_rmse = metrics(targets[validation], predict(model, validation))["rmse"]
            trace.append(
                {
                    "epoch": epoch,
                    "online_train_rmse": float(np.sqrt(total_loss / len(train)) * target_scale),
                    "validation_rmse": validation_rmse,
                }
            )
            if best is None or validation_rmse < best[0]:
                best = validation_rmse, epoch, copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= settings["patience"]:
                break
        candidates.append(
            {
                "weight_decay": decay,
                "best_epoch": best[1],
                "validation_rmse": best[0],
                "epochs_run": epoch,
                "trace": trace,
            }
        )
        if winner is None or best[0] < winner[0]:
            winner = best[0], decay, best[1], best[2]
    model.load_state_dict(winner[3])
    result = {
        "selection": {
            "weight_decay": winner[1],
            "epoch": winner[2],
            "candidates": candidates,
            "rule": "minimum validation RMSE; ties smallest weight decay then earliest epoch",
        },
        "train": metrics(targets[train], predict(model, train)),
        "validation": metrics(targets[validation], predict(model, validation)),
        "test": metrics(targets[test], predict(model, test)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "target_mean": target_mean,
        "target_std": target_scale,
        "seed": seed,
        "layers": layers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return (
        result,
        predict(model, test),
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "target_mean": target_mean,
            "target_std": target_scale,
            "settings": settings,
            "layers": layers,
            "seed": seed,
        },
    )
