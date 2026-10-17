"""Frozen official ProteinMPNN backbone encoder, before its sequence decoder."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import download, verify_file


class FrozenProteinMPNN:
    def __init__(self, specification, acquire=False):
        for spec in specification["files"]:
            if acquire and not Path(spec["path"]).exists():
                download(spec)
            verify_file(spec["path"], spec)
        source, weights, _ = specification["files"]
        spec = importlib.util.spec_from_file_location("proteinmod_pinned_mpnn", source["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.gather = module.gather_nodes
        checkpoint = torch.load(weights["path"], map_location="cpu", weights_only=True)
        self.model = module.ProteinMPNN(
            21,
            128,
            128,
            128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=checkpoint["num_edges"],
            augment_eps=0,
            dropout=0,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval().requires_grad_(False)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.encoder_parameters = sum(
            p.numel()
            for name, p in self.model.named_parameters()
            if name.startswith(("features.", "W_e.", "encoder_layers."))
        )

    def encode(self, atoms, positions):
        names = ("N", "CA", "C", "O")
        mask = np.array([all(n in a for n in names) for a in atoms])
        if not mask.all():
            raise ValueError(
                "ProteinMPNN comparison requires complete mapped backbone; match cohort first"
            )
        x = torch.tensor(np.array([[a[n] for n in names] for a in atoms]), dtype=torch.float32)[
            None
        ]
        mask = torch.tensor(mask, dtype=torch.float32)[None]
        positions = torch.tensor(positions, dtype=torch.long)[None]
        with torch.inference_mode():
            edges, indices = self.model.features(x, mask, positions, torch.ones_like(positions))
            h = torch.zeros_like(edges[:, :, 0])
            e = self.model.W_e(edges)
            attend = mask[:, :, None] * self.gather(mask[:, :, None], indices).squeeze(-1)
            for layer in self.model.encoder_layers:
                h, e = layer(h, e, indices, mask, attend)
        return h[0].numpy().copy()


def site_probe(embeddings, dynamic):
    result = []
    for d in dynamic:
        mutated = d[:, 0].astype(bool)
        if not mutated.any():
            raise ValueError("At least one mapped mutation required")
        sites = np.c_[embeddings[mutated], d[mutated, 1:]]
        result.append(
            np.r_[sites.mean(0), sites.sum(0), sites.max(0), embeddings.mean(0), mutated.sum()]
        )
    return np.asarray(result)
