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

    def forward(self, dynamic):
        static = self.static.unsqueeze(0).expand(len(dynamic), -1, -1)
        h = torch.relu(self.input(torch.cat([static, dynamic], dim=-1)))
        for message, gate, update in zip(self.messages, self.edge_gates, self.updates, strict=True):
            # Shared topology avoids a graph framework and duplicated batch edge storage.
            m = message(h)[:, self.source] * torch.sigmoid(gate(self.edge_features))[None]
            aggregate = torch.zeros_like(h)
            aggregate.index_add_(1, self.target, m)
            h = h + torch.relu(update(torch.cat([h, aggregate / self.degree], dim=-1)))
        mask = dynamic[:, :, :1]
        count = mask.sum(dim=1)
        pooled = (h * mask).sum(dim=1)
        return self.head(
            torch.cat([pooled, pooled / count.clamp(min=1), h.mean(dim=1), count], dim=-1)
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
    if any(len(ix) == 0 for ix in (train, validation, test)) or any(
        set(a) & set(b) for a, b in ((train, validation), (train, test), (validation, test))
    ):
        raise ValueError("GNN partitions must be nonempty and disjoint")
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    device = torch.device(settings["device"])
    x = torch.as_tensor(dynamic, dtype=torch.float32, device=device)
    mean, scale = float(np.mean(targets[train])), max(float(np.std(targets[train])), 1e-8)
    y = torch.as_tensor((targets - mean) / scale, dtype=torch.float32, device=device)
    batch_size = settings["batch_size"]

    def predict(model, indices):
        model.eval()
        with torch.no_grad():
            values = torch.cat(
                [
                    model(x[batch])
                    for batch in np.array_split(
                        indices, max(1, int(np.ceil(len(indices) / batch_size)))
                    )
                ]
            )
        return values.cpu().numpy().astype(np.float64) * scale + mean

    candidates, winner = [], None
    for decay in sorted(set(settings["weight_decays"])):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ResidueGNN(
            graph["static"], graph["edge_index"], graph["edge_features"], settings["hidden"], layers
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"], weight_decay=decay
        )
        best, stale, trace = None, 0, []
        for epoch in range(1, settings["max_epochs"] + 1):
            model.train()
            order = rng.permutation(train)
            total = 0.0
            for start in range(0, len(order), batch_size):
                ix = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(x[ix]) - y[ix]) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite GNN training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                total += loss.item() * len(ix)
            score = metrics(targets[validation], predict(model, validation))["rmse"]
            trace.append(
                {
                    "epoch": epoch,
                    "online_train_rmse": float(np.sqrt(total / len(train)) * scale),
                    "validation_rmse": score,
                }
            )
            if best is None or score < best[0]:
                best = score, epoch, copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= settings["patience"]:
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
        "target_mean": mean,
        "target_std": scale,
        "seed": seed,
        "layers": layers,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return (
        result,
        predict(model, test),
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "target_mean": mean,
            "target_std": scale,
            "settings": settings,
            "layers": layers,
            "seed": seed,
        },
    )
