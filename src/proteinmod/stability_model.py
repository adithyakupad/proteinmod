"""Independent continuous ΔΔG head over a frozen M5 encoder; no function objective."""

import hashlib

import torch
from torch import nn

from proteinmod.hybrid import HybridRegressor, pooled

BUFFER_NAMES = {
    "structure." + name for name in ("static", "source", "target", "edge_features", "degree")
}


def encoder_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key in BUFFER_NAMES or key.startswith("head."):
            continue
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class StabilityModel(nn.Module):
    """Frozen pretrained GNN/QKV; only stability_head is optimized, output kcal/mol."""

    def __init__(
        self, graph, embedding_dimension=480, dimension=16, heads=2, layers=2, encoder_state=None
    ):
        super().__init__()
        self.backbone = HybridRegressor(
            graph, embedding_dimension, dimension, heads, layers, "bidirectional"
        )
        if encoder_state is not None:
            selected = {
                k: v
                for k, v in encoder_state.items()
                if k in dict(self.backbone.named_parameters()) and not k.startswith("head.")
            }
            expected = {
                k for k in dict(self.backbone.named_parameters()) if not k.startswith("head.")
            }
            if set(selected) != expected:
                raise ValueError("Incomplete pretrained encoder parameter set")
            self.backbone.load_state_dict(selected, strict=False)
        self.backbone.head = nn.Identity()
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.stability_head = nn.Sequential(
            nn.Linear(6 * dimension + 4, dimension), nn.ReLU(), nn.Linear(dimension, 1)
        )
        self.register_buffer("target_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("target_scale", torch.ones((), dtype=torch.float64))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def represent(self, residues, dynamic, topology, positions, mutation_mask, modality_mask=None):
        b, length, _ = residues.shape
        n = dynamic.shape[1]
        if (
            mutation_mask.shape != (b, length)
            or mutation_mask.dtype != torch.bool
            or positions.shape != (n,)
        ):
            raise ValueError("Invalid reference mutation mask or graph correspondence")
        if (
            dynamic.shape != (b, n, 25)
            or len(set(positions.tolist())) != n
            or bool(((positions < 0) | (positions >= length)).any())
        ):
            raise ValueError("Invalid structural tensor dimensions/correspondence")
        available = torch.ones((b, 2), dtype=torch.bool) if modality_mask is None else modality_mask
        if available.shape != (b, 2) or available.dtype != torch.bool or not available.any(1).all():
            raise ValueError("At least one explicit available modality required")
        if bool(
            ((dynamic[:, :, 0].bool() != mutation_mask[:, positions]) & available[:, 1, None]).any()
        ):
            raise ValueError("Sequence and structure mutation masks disagree")
        sm = available[:, :1].expand(b, length)
        gm = available[:, 1:].expand(b, n)
        s = self.backbone.sequence(residues.masked_fill(~sm[..., None], 0))
        g = self.backbone.structure_norm(
            self.backbone.structure.encode(dynamic.masked_fill(~gm[..., None], 0), topology)
        )
        width = max(1, int(mutation_mask.sum(1).max()))
        sites = torch.zeros((b, width), dtype=torch.long)
        qm = torch.zeros((b, width), dtype=torch.bool)
        for i in range(b):
            ix = torch.where(mutation_mask[i])[0]
            sites[i, : len(ix)] = ix
            qm[i, : len(ix)] = True
        lookup = torch.full((length,), -1, dtype=torch.long)
        lookup[positions] = torch.arange(n)
        graph_sites = lookup[sites]
        sqmask = qm & available[:, :1]
        gqmask = qm & (graph_sites >= 0) & available[:, 1:]
        batch = torch.arange(b)[:, None]
        sq, gq = s[batch, sites], g[batch, graph_sites.clamp(min=0)]
        su, _ = self.backbone.s_to_g(sq, g, sqmask, gm)
        gu, _ = self.backbone.g_to_s(gq, s, gqmask, sm)
        sp = pooled(self.backbone.s_norm(sq + su), sqmask, s, sm)
        gp = pooled(self.backbone.g_norm(gq + gu), gqmask, g, gm)
        count = qm.sum(1, keepdim=True)
        coverage = gqmask.sum(1, keepdim=True) / count.clamp(min=1)
        return torch.cat((sp, gp, count, available, coverage), -1)

    def predict_features(self, features):
        return (
            self.stability_head(features).squeeze(-1).double() * self.target_scale
            + self.target_mean
        )

    def load_head(self, state):
        if state.get("encoder_tensor_sha256") != encoder_digest(self.backbone.state_dict()):
            raise ValueError("Stability head belongs to a different frozen encoder")
        if state.get("feature_schema") != "m6-frozen-hybrid-v1":
            raise ValueError("Stability feature schema mismatch")
        mean, scale = float(state["target_mean"]), float(state["target_scale"])
        if not torch.isfinite(torch.tensor([mean, scale])).all() or scale <= 0:
            raise ValueError("Invalid physical target calibration")
        self.stability_head.load_state_dict(state["stability_head"])
        self.target_mean.fill_(mean)
        self.target_scale.fill_(scale)
        self.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.predict_features(self.represent(*args, **kwargs))
