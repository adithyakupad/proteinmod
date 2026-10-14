"""Mutation-query sequence/structure attention; graph construction remains in M2."""

import math

import torch
from torch import nn

from proteinmod.structural_gnn import ResidueGNN

MODES = (
    "sequence_only",
    "structure_only",
    "weighted_fusion",
    "concat_fusion",
    "seq_queries_struct",
    "struct_queries_seq",
    "bidirectional",
)


class CrossAttention(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        if dimension < 1 or heads < 1 or dimension % heads:
            raise ValueError("Attention dimension must be divisible by heads")
        self.heads = heads
        self.q, self.k, self.v, self.out = (
            nn.Linear(dimension, dimension, bias=False) for _ in range(4)
        )

    def forward(self, query, context, query_mask, key_mask):
        if (
            query.ndim != 3
            or context.ndim != 3
            or query.shape[0] != context.shape[0]
            or query.shape[-1] != context.shape[-1]
            or query.shape[-1] != self.q.in_features
            or query_mask.shape != query.shape[:2]
            or key_mask.shape != context.shape[:2]
            or query_mask.dtype != torch.bool
            or key_mask.dtype != torch.bool
            or context.shape[1] == 0
        ):
            raise ValueError("Invalid Q/K/V dimensions or boolean masks")
        query = query.masked_fill(~query_mask[..., None], 0)
        context = context.masked_fill(~key_mask[..., None], 0)
        b, n, d = query.shape

        def split(x):
            return x.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)

        scores = (
            split(self.q(query))
            @ split(self.k(context)).transpose(-1, -2)
            / math.sqrt(d // self.heads)
        )
        keys = key_mask[:, None, None, :]
        weights = scores.masked_fill(~keys, -torch.finfo(scores.dtype).max).softmax(-1) * keys
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-12)
        weights = weights * query_mask[:, None, :, None]
        result = (weights @ split(self.v(context))).transpose(1, 2).reshape(b, n, d)
        return self.out(result), weights


def pooled(query, mask, context, context_mask):
    query = query.masked_fill(~mask[..., None], 0)
    count = mask.sum(1, keepdim=True)
    total = query.sum(1)
    context = context.masked_fill(~context_mask[..., None], 0)
    global_mean = context.sum(1) / context_mask.sum(1, keepdim=True).clamp(min=1)
    return torch.cat((total, total / count.clamp(min=1), global_mean), -1)


class HybridRegressor(nn.Module):
    def __init__(
        self, graph, embedding_dimension, dimension=16, heads=2, layers=2, mode="bidirectional"
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown hybrid mode")
        self.mode = mode
        if mode != "structure_only":
            self.sequence = nn.Sequential(
                nn.Linear(embedding_dimension, dimension), nn.LayerNorm(dimension)
            )
        if mode != "sequence_only":
            self.structure = ResidueGNN(
                graph["static"], graph["edge_index"], graph["edge_features"], dimension, layers
            )
            self.structure.head = nn.Identity()
            self.structure_norm = nn.LayerNorm(dimension)
        if mode in ("seq_queries_struct", "bidirectional"):
            self.s_to_g = CrossAttention(dimension, heads)
            self.s_norm = nn.LayerNorm(dimension)
        if mode in ("struct_queries_seq", "bidirectional"):
            self.g_to_s = CrossAttention(dimension, heads)
            self.g_norm = nn.LayerNorm(dimension)
        if mode == "weighted_fusion":
            self.weight = nn.Parameter(torch.zeros(()))
        single = mode in ("sequence_only", "structure_only", "weighted_fusion")
        self.head = nn.Sequential(
            nn.Linear((3 if single else 6) * dimension + 1, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1),
        )

    def forward(
        self,
        residues,
        dynamic,
        topology,
        residue_positions,
        sequence_mask=None,
        graph_mask=None,
        mutation_order=None,
        diagnostics=False,
    ):
        b, length, _ = residues.shape
        n = dynamic.shape[1]
        if dynamic.shape != (b, n, 25) or residue_positions.shape != (n,):
            raise ValueError("Invalid graph/mutation dimensions")
        if len(set(residue_positions.tolist())) != n or bool(
            ((residue_positions < 0) | (residue_positions >= length)).any()
        ):
            raise ValueError("Graph/reference correspondence must be unique and in range")
        sm = torch.ones((b, length), dtype=torch.bool) if sequence_mask is None else sequence_mask
        gm = torch.ones((b, n), dtype=torch.bool) if graph_mask is None else graph_mask
        if (
            sm.shape != (b, length)
            or gm.shape != (b, n)
            or sm.dtype != torch.bool
            or gm.dtype != torch.bool
        ):
            raise ValueError("Invalid sequence/structure masks")
        mutation_mask = dynamic[:, :, 0].bool()
        if bool((mutation_mask & ~gm).any()) or bool(
            (mutation_mask & ~sm[:, residue_positions]).any()
        ):
            raise ValueError("Mutation at a masked or unmapped residue")
        # Padded query slots are masked; WT has one inert slot and retains global context.
        width = max(1, int(mutation_mask.sum(1).max()))
        query_indices = torch.zeros((b, width), dtype=torch.long)
        qm = torch.zeros((b, width), dtype=torch.bool)
        for i in range(b):
            ix = torch.where(mutation_mask[i])[0]
            if mutation_order is not None:
                ix = ix[mutation_order[: len(ix)]]
            query_indices[i, : len(ix)] = ix
            qm[i, : len(ix)] = True
        batch = torch.arange(b)[:, None]
        weights = {}
        if self.mode != "structure_only":
            s = self.sequence(residues.masked_fill(~sm[..., None], 0))
            sq = s[batch, residue_positions[query_indices]]
        if self.mode != "sequence_only":
            g = self.structure_norm(self.structure.encode(dynamic, topology))
            gq = g[batch, query_indices]
        if hasattr(self, "s_to_g"):
            update, weights["seq_queries_struct"] = self.s_to_g(sq, g, qm, gm)
            new_sq = self.s_norm(sq + update)
        if hasattr(self, "g_to_s"):
            update, weights["struct_queries_seq"] = self.g_to_s(gq, s, qm, sm)
            gq = self.g_norm(gq + update)
        if hasattr(self, "s_to_g"):
            sq = new_sq
        if self.mode != "structure_only":
            sp = pooled(sq, qm, s, sm)
        if self.mode != "sequence_only":
            gp = pooled(gq, qm, g, gm)
        if self.mode == "sequence_only":
            merged = sp
        elif self.mode == "structure_only":
            merged = gp
        elif self.mode == "weighted_fusion":
            w = self.weight.sigmoid()
            merged = w * sp + (1 - w) * gp
        else:
            merged = torch.cat((sp, gp), -1)
        prediction = self.head(torch.cat((merged, qm.sum(1, keepdim=True)), -1)).squeeze(-1)
        return (prediction, weights) if diagnostics else prediction
