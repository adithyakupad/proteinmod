"""Bounded last-block ESM LoRA; original weights frozen, mutation-set readout symmetric."""

import copy
import hashlib
import math

import torch
from torch import nn


def tensor_digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(x.shape), x.dtype)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


class LowRank(nn.Module):
    def __init__(self, width, rank, alpha, dropout=0.0):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA rank/alpha/dropout")
        self.a = nn.Linear(width, rank, bias=False)
        self.b = nn.Linear(rank, width, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.b(self.a(self.dropout(x))) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if base.in_features != base.out_features:
            raise ValueError("M15 adapts square ESM query/value projections only")
        self.base = base.requires_grad_(False)
        self.adapter = LowRank(base.in_features, rank, alpha, dropout)

    def forward(self, x):
        return self.base(x) + self.adapter(x)


def site_representation(mutant, wt, mutation_mask):
    if mutant.ndim != 3 or wt.shape != mutant.shape or mutation_mask.shape != mutant.shape[:2]:
        raise ValueError("Mutant/WT/mutation mask shape mismatch")
    count = mutation_mask.sum(1, keepdim=True)
    if torch.any(count < 1) or not torch.all((mutation_mask == 0) | (mutation_mask == 1)):
        raise ValueError("At least one valid unique mutation site required")
    mask = mutation_mask[:, :, None]
    return torch.cat(
        [(mutant * mask).sum(1) / count, ((mutant - wt) * mask).sum(1) / count, count], -1
    )


class MutationHead(nn.Module):
    def __init__(self, width=480, hidden=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(2 * width + 1, elementwise_affine=False),
            nn.Linear(2 * width + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class AdaptedESM(nn.Module):
    def __init__(self, backbone, settings, hidden=16, mode="lora", seed=0):
        super().__init__()
        if (
            mode not in ("lora", "frozen_head", "matched_head")
            or settings["layers"] != [len(backbone.encoder.layer) - 1]
            or settings["projections"] != ["query", "value"]
        ):
            raise ValueError("M15 supports only final-block query/value LoRA")
        self.mode = mode
        backbone.requires_grad_(False).eval()
        width = backbone.config.hidden_size
        # Initialize the common head before adapter construction to match it across controls.
        torch.manual_seed(seed)
        self.head = MutationHead(width, hidden)
        self.control = nn.ModuleList()
        self.tail = copy.deepcopy(backbone.encoder.layer[-1]).requires_grad_(False)
        self.norm = copy.deepcopy(backbone.encoder.emb_layer_norm_after).requires_grad_(False)
        if mode == "lora":
            attention = self.tail.attention.self
            for name in settings["projections"]:
                setattr(
                    attention,
                    name,
                    LoRALinear(
                        getattr(attention, name),
                        settings["rank"],
                        settings["alpha"],
                        settings["dropout"],
                    ),
                )
        elif mode == "matched_head":
            self.control.extend(
                LowRank(width, settings["rank"], settings["alpha"], settings["dropout"])
                for _ in range(2)
            )
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # Evaluation-mode backbone dropout is independent of gradient computation.
        self.tail.eval()
        self.norm.eval()
        return self

    def frozen_digest(self):
        return tensor_digest({n: p for n, p in self.named_parameters() if not p.requires_grad})

    def selected_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_selected(self, state):
        expected = {n for n, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError("Adapter/head checkpoint schema mismatch")
        params = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != params[name].shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid adapter/head checkpoint tensor")
                params[name].copy_(value)

    def features(self, prefix, extended_mask, mutation_mask):
        # Last row is WT, shared by this same-protein batch; gradients flow through both paths.
        if prefix.shape[0] != len(mutation_mask) + 1:
            raise ValueError("Prefix batch must contain mutants and one WT")
        valid = extended_mask[:-1, 0, 0, 1:-1] == 0
        if valid.shape != mutation_mask.shape or torch.any((mutation_mask > 0) & ~valid):
            raise ValueError("Mutation mask selects padding or invalid tokens")
        hidden = self.norm(self.tail(prefix, attention_mask=extended_mask))[:, 1:-1]
        return site_representation(
            hidden[:-1], hidden[-1:].expand(len(hidden) - 1, -1, -1), mutation_mask
        )

    def forward(self, features=None, prefix=None, extended_mask=None, mutation_mask=None):
        if self.mode == "lora":
            features = self.features(prefix, extended_mask, mutation_mask)
        if self.mode == "matched_head":
            width = (features.shape[-1] - 1) // 2
            features = torch.cat(
                [
                    features[:, :width] + self.control[0](features[:, :width]),
                    features[:, width : 2 * width]
                    + self.control[1](features[:, width : 2 * width]),
                    features[:, -1:],
                ],
                -1,
            )
        return self.head(features)
