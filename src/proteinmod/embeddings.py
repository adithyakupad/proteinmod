"""Versioned, label-free frozen ESM-2 extraction; heavyweight imports are lazy."""

import hashlib
import importlib.metadata
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, mutations, write_json

FEATURE_NAMES = ("esm_mean", "esm_delta", "esm_sites")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def contract(dataset_sha, settings, sequences, reference):
    if not re.fullmatch(r"[0-9a-f]{40}", settings["revision"]):
        raise ValueError("ESM revision must be an immutable 40-character commit SHA")
    if settings["batch_size"] < 1:
        raise ValueError("Embedding batch_size must be positive")
    for sequence in sequences:
        mutations(reference, sequence)
    return {
        "schema": "proteinmod-esm2-v1",
        "dataset_sha256": dataset_sha,
        "settings": settings,
        "sequences_sha256": fingerprint(sequences),
        "reference_sha256": fingerprint(reference),
        "extractor_sha256": checksum(__file__),
    }


def site_features(residues, wt_residues, positions):
    """Symmetric mean mutant / mean delta at changed sites, plus site count.

    positions are zero-based and unique; WT (empty set) has zero site features.
    """
    positions = sorted(set(positions))
    if residues.shape != wt_residues.shape or residues.ndim != 2:
        raise ValueError("WT and mutant residue representations must have identical shapes")
    if positions and (positions[0] < 0 or positions[-1] >= len(residues)):
        raise ValueError("Mutation site outside residue representation")
    if not positions:
        return np.zeros(2 * residues.shape[1] + 1, dtype=np.float32)
    return np.concatenate(
        [
            residues[positions].mean(axis=0),
            (residues[positions] - wt_residues[positions]).mean(axis=0),
            [len(positions)],
        ]
    ).astype(np.float32)


def residue_tokens(hidden, input_ids, attention_mask, special_mask, sequences, tokenizer):
    """Mask specials and padding, then verify each residue token in sequence order."""
    result = []
    for row, sequence in enumerate(sequences):
        keep = attention_mask[row].astype(bool) & ~special_mask[row].astype(bool)
        ids = input_ids[row][keep].tolist()
        expected = tokenizer.convert_tokens_to_ids(list(sequence))
        if len(ids) != len(sequence) or ids != expected:
            raise ValueError("Residue-token alignment mismatch or unintended truncation")
        residues = hidden[row][keep]
        if not np.isfinite(residues).all():
            raise ValueError("Encoder returned nonfinite representations")
        result.append(residues)
    return result


class FrozenESM:
    def __init__(self, settings, local_only=False):
        import torch
        from transformers import AutoTokenizer, EsmModel

        requested = settings["device"]
        available = {
            "cpu": True,
            "mps": torch.backends.mps.is_available(),
            "cuda": torch.cuda.is_available(),
        }
        self.device = (
            ("cuda" if available["cuda"] else "mps" if available["mps"] else "cpu")
            if requested == "auto"
            else requested
        )
        if not available.get(self.device, False):
            raise ValueError(f"Requested device {self.device} unavailable; select cpu or auto")
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(settings.get("cpu_threads", 4))
        kwargs = {
            "revision": settings["revision"],
            "cache_dir": settings["model_cache"],
            "local_files_only": local_only,
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(settings["model"], **kwargs)
        self.model = EsmModel.from_pretrained(
            settings["model"], add_pooling_layer=False, use_safetensors=True, **kwargs
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device=self.device, dtype=torch.float32)
        if self.model.config.hidden_size != settings["dimension"]:
            raise ValueError("Configured embedding dimension differs from checkpoint")
        self.max_residues = self.model.config.max_position_embeddings - 2
        self.info = {
            "device": self.device,
            "dtype": "float32",
            "frozen": True,
            "eval": True,
            "deterministic_algorithms": True,
            "layer": "last_hidden_state",
            "tokenizer": type(self.tokenizer).__name__,
            "tokenizer_revision": settings["revision"],
            "vocabulary_sha256": fingerprint(self.tokenizer.get_vocab()),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "numpy", "safetensors", "huggingface-hub")
            },
        }

    def encode(self, sequences):
        import torch

        if any(len(sequence) > self.max_residues for sequence in sequences):
            raise ValueError(
                f"Sequence exceeds checkpoint limit {self.max_residues}; no truncation"
            )
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = tokens.pop("special_tokens_mask").numpy()
        ids, mask = tokens["input_ids"].numpy(), tokens["attention_mask"].numpy()
        try:
            with torch.inference_mode():
                hidden = self.model(**{k: v.to(self.device) for k, v in tokens.items()})
                hidden = hidden.last_hidden_state.cpu().numpy()
        except RuntimeError as error:
            raise RuntimeError(
                f"ESM inference failed on {self.device}; reduce batch_size or use CPU. "
                f"Original error: {error}"
            ) from error
        return residue_tokens(hidden, ids, mask, special, sequences, self.tokenizer)


def cache_location(config, sequences, reference, dataset_sha):
    identity = contract(dataset_sha, config["embedding"], sequences, reference)
    directory = Path(config["embedding"]["cache"]) / fingerprint(identity)
    return directory, identity


def load_features(directory, identity, count, dimension):
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["contract"] != identity or metadata["sha256"] != checksum(
        directory / "features.npz"
    ):
        raise ValueError("Embedding cache provenance/checksum mismatch; re-extract")
    with np.load(directory / "features.npz", allow_pickle=False) as stored:
        features = {name: stored[name] for name in FEATURE_NAMES}
    for name, matrix in features.items():
        width = 2 * dimension + 1 if name == "esm_sites" else dimension
        if matrix.shape != (count, width) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid cache shape/values for {name}")
    return features, metadata


def extract(config, sequences, reference, dataset_sha, encoder=None, local_only=False):
    directory, identity = cache_location(config, sequences, reference, dataset_sha)
    dimension = config["embedding"]["dimension"]
    if (directory / "manifest.json").exists():
        load_features(directory, identity, len(sequences), dimension)
        print(f"Verified embedding cache: {directory}", flush=True)
        return directory
    started = time.perf_counter()
    encoder = encoder or FrozenESM(config["embedding"], local_only=local_only)
    directory.mkdir(parents=True, exist_ok=True)
    wt = encoder.encode([reference])[0]
    if wt.shape != (len(reference), dimension):
        raise ValueError("WT encoder shape mismatch")
    means, sites = [], []
    batch_size = config["embedding"]["batch_size"]
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = encoder.encode(batch)
        if len(encoded) != len(batch):
            raise ValueError("Encoder returned incorrect batch size")
        for sequence, residues in zip(batch, encoded, strict=True):
            if residues.shape != wt.shape or not np.isfinite(residues).all():
                raise ValueError("Mutant encoder residue shape/values mismatch")
            positions = [i for i, (old, new) in enumerate(zip(reference, sequence)) if old != new]
            means.append(residues.mean(axis=0))
            sites.append(site_features(residues, wt, positions))
        if start == 0 or (start // batch_size) % 10 == 0 or start + batch_size >= len(sequences):
            print(
                f"Embedded {min(start + batch_size, len(sequences))}/{len(sequences)} "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    mean = np.asarray(means, dtype=np.float32)
    # Use float64 subtraction so mean and delta are a precise constant translation in ridge.
    delta = mean.astype(np.float64) - wt.mean(axis=0).astype(np.float64)
    temporary = directory / "features.tmp.npz"
    np.savez_compressed(
        temporary,
        esm_mean=mean,
        esm_delta=delta,
        esm_sites=np.asarray(sites, dtype=np.float32),
        wt_residues=wt,
    )
    temporary.replace(directory / "features.npz")
    write_json(
        directory / "manifest.json",
        {
            "contract": identity,
            "encoder": encoder.info,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "sha256": checksum(directory / "features.npz"),
            "pooling": "mean over residues only; no BOS/EOS/pad; mutant-minus-WT delta",
            "sites": "concat(mean mutant at changed sites, mean mutant-minus-WT at sites, site count); WT zeros",
            "label_free": True,
        },
    )
    load_features(directory, identity, len(sequences), dimension)
    return directory
