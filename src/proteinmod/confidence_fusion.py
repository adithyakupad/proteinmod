"""Frozen bidirectional encoder, typed structural inputs and learned reliability fusion."""

import numpy as np
import torch
from torch import nn

from proteinmod.stability_model import StabilityModel
from proteinmod.structure_reliability import FIELDS

MODES = (
    "unconditioned",
    "confidence_input",
    "weighted_late",
    "gate",
    "combined",
    "gate_zero",
    "input_shuffled",
    "combined_shuffled",
)


def frozen_representation(encoder_state, graph, residues, dynamic, mutation_mask):
    """No stability prediction is called; missing geometry has a safe inert graph slot."""
    if np.ndim(residues) != 3 or not np.isfinite(residues).all():
        raise ValueError("Finite ESM residue tensors required")
    available = graph is not None
    if not available:
        graph = {
            "static": np.zeros((1, 34), np.float32),
            "edge_index": np.empty((2, 0), np.int64),
            "edge_features": np.empty((0, 3), np.float32),
            "nodes": [{"reference_position": 1}],
        }
        dynamic = np.zeros((len(residues), 1, 25), np.float32)
    model = StabilityModel(graph, encoder_state=encoder_state).requires_grad_(False).eval()
    positions = torch.tensor([n["reference_position"] - 1 for n in graph["nodes"]])
    with torch.no_grad():
        return model.represent(
            torch.as_tensor(residues, dtype=torch.float32),
            torch.as_tensor(dynamic, dtype=torch.float32),
            None,
            positions,
            torch.as_tensor(mutation_mask, dtype=torch.bool),
            torch.tensor([[True, available]] * len(residues)),
        ).numpy()


class ReliabilityFusion(nn.Module):
    def __init__(self, dimension=16, hidden=16):
        super().__init__()
        self.width = 3 * dimension
        width = self.width
        self.concat = nn.Sequential(
            nn.Linear(2 * width + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.sequence = nn.Sequential(nn.Linear(width + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.structure = nn.Sequential(
            nn.Linear(width + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * width + len(FIELDS), hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, representation, reliability, mode="combined", diagnostics=False):
        if (
            mode not in MODES
            or representation.ndim != 2
            or representation.shape[1] != 2 * self.width + 4
            or reliability.shape != (len(representation), len(FIELDS))
        ):
            raise ValueError("Invalid confidence fusion dimensions/mode")
        if not torch.isfinite(representation).all() or not torch.isfinite(reliability).all():
            raise ValueError("Nonfinite fusion inputs")
        s, g = representation[:, : self.width], representation[:, self.width : 2 * self.width]
        count = representation[:, -4:-3]
        available = representation[:, -2].bool()
        # Recompute missing-structure sequence features upstream: no structural attention keys.
        sequence = self.sequence(torch.cat((s, count), -1)).squeeze(-1).sigmoid()
        structure = self.structure(torch.cat((g, count), -1)).squeeze(-1).sigmoid()
        if mode in ("gate", "combined", "gate_zero", "combined_shuffled"):
            confidence = torch.zeros_like(reliability) if mode in ("gate_zero",) else reliability
            gate = self.gate(torch.cat((s, g, confidence), -1)).squeeze(-1).sigmoid()
        else:
            gate = self.weight.sigmoid().expand(len(s))
        gate = torch.where(available, gate, 0)
        if mode in ("unconditioned", "confidence_input", "input_shuffled"):
            prediction = self.concat(representation).squeeze(-1).sigmoid()
        else:
            prediction = (1 - gate) * sequence + gate * structure
        return (prediction, gate, sequence, structure) if diagnostics else prediction
