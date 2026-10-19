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
        node_hidden = torch.relu(
            self.input(torch.cat([static[None].expand(len(dynamic), -1, -1), dynamic], -1))
        )
        for message, update in zip(self.messages, self.updates, strict=True):
            edge_messages = message(
                torch.cat(
                    [
                        node_hidden[:, source],
                        node_hidden[:, target],
                        features[None].expand(len(dynamic), -1, -1),
                    ],
                    -1,
                )
            )
            aggregate = torch.zeros_like(node_hidden).index_add(1, target, edge_messages)
            if self.zero_message:
                aggregate = aggregate * 0
            node_hidden = node_hidden + torch.relu(
                update(torch.cat([node_hidden, aggregate / degree], -1))
            )
        mutation_count = dynamic[:, :, :1].sum(1)
        mutation_sum = (node_hidden * dynamic[:, :, :1]).sum(1)
        return self.head(
            torch.cat(
                [
                    mutation_sum,
                    mutation_sum / mutation_count.clamp(min=1),
                    node_hidden.mean(1),
                    mutation_count,
                ],
                -1,
            )
        ).squeeze(-1)
