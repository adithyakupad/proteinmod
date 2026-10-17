"""Compact learned edge-conditioned messages; topology remains an external input."""

import torch
from torch import nn


class EdgeGNN(nn.Module):
    def __init__(self, static_width=34, edge_width=29, hidden=16, layers=2, zero_message=False):
        super().__init__()
        self.zero_message = zero_message
        self.input = nn.Linear(static_width + 25, hidden)
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden + edge_width, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )
            for _ in range(layers)
        )
        self.updates = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, dynamic, graph):
        static, source, target, features, degree = graph
        if dynamic.ndim != 3 or dynamic.shape[1:] != (len(static), 25):
            raise ValueError("Expected batch x residues x 25 mutation input")
        h = torch.relu(
            self.input(torch.cat([static[None].expand(len(dynamic), -1, -1), dynamic], -1))
        )
        for message, update in zip(self.messages, self.updates, strict=True):
            m = message(
                torch.cat(
                    [h[:, source], h[:, target], features[None].expand(len(dynamic), -1, -1)], -1
                )
            )
            aggregate = torch.zeros_like(h).index_add(1, target, m)
            if self.zero_message:
                aggregate = aggregate * 0
            h = h + torch.relu(update(torch.cat([h, aggregate / degree], -1)))
        count = dynamic[:, :, :1].sum(1)
        pooled = (h * dynamic[:, :, :1]).sum(1)
        return self.head(
            torch.cat([pooled, pooled / count.clamp(min=1), h.mean(1), count], -1)
        ).squeeze(-1)
