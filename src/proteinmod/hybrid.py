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
        batch_size, query_count, feature_width = query.shape

        def split(x):
            return x.reshape(batch_size, -1, self.heads, feature_width // self.heads).transpose(
                1, 2
            )

        scores = (
            split(self.q(query))
            @ split(self.k(context)).transpose(-1, -2)
            / math.sqrt(feature_width // self.heads)
        )
        valid_keys = key_mask[:, None, None, :]
        weights = (
            scores.masked_fill(~valid_keys, -torch.finfo(scores.dtype).max).softmax(-1) * valid_keys
        )
        weights = weights / weights.sum(-1, keepdim=True).clamp(min=1e-12)
        weights = weights * query_mask[:, None, :, None]
        result = (
            (weights @ split(self.v(context)))
            .transpose(1, 2)
            .reshape(batch_size, query_count, feature_width)
        )
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
        batch_size, length, _ = residues.shape
        node_count = dynamic.shape[1]
        if dynamic.shape != (batch_size, node_count, 25) or residue_positions.shape != (
            node_count,
        ):
            raise ValueError("Invalid graph/mutation dimensions")
        if len(set(residue_positions.tolist())) != node_count or bool(
            ((residue_positions < 0) | (residue_positions >= length)).any()
        ):
            raise ValueError("Graph/reference correspondence must be unique and in range")
        sequence_valid = (
            torch.ones((batch_size, length), dtype=torch.bool)
            if sequence_mask is None
            else sequence_mask
        )
        graph_valid = (
            torch.ones((batch_size, node_count), dtype=torch.bool)
            if graph_mask is None
            else graph_mask
        )
        if (
            sequence_valid.shape != (batch_size, length)
            or graph_valid.shape != (batch_size, node_count)
            or sequence_valid.dtype != torch.bool
            or graph_valid.dtype != torch.bool
        ):
            raise ValueError("Invalid sequence/structure masks")
        mutation_mask = dynamic[:, :, 0].bool()
        if bool((mutation_mask & ~graph_valid).any()) or bool(
            (mutation_mask & ~sequence_valid[:, residue_positions]).any()
        ):
            raise ValueError("Mutation at a masked or unmapped residue")
        # Padded query slots are masked; WT has one inert slot and retains global context.
        width = max(1, int(mutation_mask.sum(1).max()))
        query_indices = torch.zeros((batch_size, width), dtype=torch.long)
        query_valid = torch.zeros((batch_size, width), dtype=torch.bool)
        for i in range(batch_size):
            mutation_indices = torch.where(mutation_mask[i])[0]
            if mutation_order is not None:
                mutation_indices = mutation_indices[mutation_order[: len(mutation_indices)]]
            query_indices[i, : len(mutation_indices)] = mutation_indices
            query_valid[i, : len(mutation_indices)] = True
        batch = torch.arange(batch_size)[:, None]
        weights = {}
        if self.mode != "structure_only":
            sequence_hidden = self.sequence(residues.masked_fill(~sequence_valid[..., None], 0))
            sequence_queries = sequence_hidden[batch, residue_positions[query_indices]]
        if self.mode != "sequence_only":
            graph_hidden = self.structure_norm(self.structure.encode(dynamic, topology))
            graph_queries = graph_hidden[batch, query_indices]
        if hasattr(self, "s_to_g"):
            update, weights["seq_queries_struct"] = self.s_to_g(
                sequence_queries, graph_hidden, query_valid, graph_valid
            )
            updated_sequence_queries = self.s_norm(sequence_queries + update)
        if hasattr(self, "g_to_s"):
            update, weights["struct_queries_seq"] = self.g_to_s(
                graph_queries, sequence_hidden, query_valid, sequence_valid
            )
            graph_queries = self.g_norm(graph_queries + update)
        if hasattr(self, "s_to_g"):
            sequence_queries = updated_sequence_queries
        if self.mode != "structure_only":
            sequence_pooled = pooled(sequence_queries, query_valid, sequence_hidden, sequence_valid)
        if self.mode != "sequence_only":
            graph_pooled = pooled(graph_queries, query_valid, graph_hidden, graph_valid)
        if self.mode == "sequence_only":
            merged = sequence_pooled
        elif self.mode == "structure_only":
            merged = graph_pooled
        elif self.mode == "weighted_fusion":
            fusion_weight = self.weight.sigmoid()
            merged = fusion_weight * sequence_pooled + (1 - fusion_weight) * graph_pooled
        else:
            merged = torch.cat((sequence_pooled, graph_pooled), -1)
        prediction = self.head(torch.cat((merged, query_valid.sum(1, keepdim=True)), -1)).squeeze(
            -1
        )
        return (prediction, weights) if diagnostics else prediction
