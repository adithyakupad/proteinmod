"""Versioned, observable assay metadata; no outcomes or held-out vocabularies."""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

CATEGORIES = (
    "family",
    "endpoint",
    "technology",
    "context",
    "substrate",
    "unit",
    "direction",
    "assay_id",
    "protein_proxy",
)
STATUSES = ("observed", "unavailable", "not_applicable")
MODES = ("zero", "structured", "with_assay_id", "shuffled_train", "protein_proxy")


@dataclass(frozen=True)
class Condition:
    value: float | None
    status: str

    def __post_init__(self):
        if self.status not in STATUSES or (self.status == "observed") != (self.value is not None):
            raise ValueError("Condition value/status disagree")
        if self.value is not None and not np.isfinite(self.value):
            raise ValueError("Nonfinite condition")


@dataclass(frozen=True)
class AssayMetadata:
    assay_id: str
    family: str
    endpoint: str
    technology: str
    context: str | None
    substrate: str | None
    unit: str
    direction: str
    temperature_c: Condition
    ph: Condition
    provenance: list[str]

    @classmethod
    def from_dict(cls, raw):
        try:
            result = cls(
                **{
                    **raw,
                    "temperature_c": Condition(**raw["temperature_c"]),
                    "ph": Condition(**raw["ph"]),
                }
            )
        except (TypeError, KeyError) as error:
            raise ValueError(
                "Unknown/missing metadata field; labels and derived statistics forbidden"
            ) from error
        if not result.provenance or result.direction != "higher_score_better":
            raise ValueError("Provenance and explicit processed-score direction required")
        for field in CATEGORIES[:-1]:
            value = getattr(result, field)
            if value is not None and (not isinstance(value, str) or not value or len(value) > 200):
                raise ValueError("Invalid categorical metadata")
        return result

    def to_dict(self):
        return asdict(self)


class MetadataVocabulary:
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, metadata, proteins):
        vocabularies = {}
        for field in CATEGORIES:
            values = proteins if field == "protein_proxy" else [getattr(m, field) for m in metadata]
            tokens = sorted({"__missing__" if v is None else v for v in values})
            vocabularies[field] = {v: j + 1 for j, v in enumerate(tokens)}
        numeric = {}
        for field in ("temperature_c", "ph"):
            values = [
                getattr(m, field).value for m in metadata if getattr(m, field).status == "observed"
            ]
            numeric[field] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "scale": max(float(np.std(values)), 1e-6) if values else 1.0,
                "observed_training_records": len(values),
            }
        return cls(
            {"schema": "m9-assay-metadata-v1", "vocabularies": vocabularies, "numeric": numeric}
        )

    def transform(self, metadata, proteins):
        cat = np.zeros((len(metadata), len(CATEGORIES)), dtype=np.int64)
        num = np.zeros((len(metadata), 8), dtype=np.float32)
        for i, (m, protein) in enumerate(zip(metadata, proteins, strict=True)):
            for j, field in enumerate(CATEGORIES):
                value = protein if field == "protein_proxy" else getattr(m, field)
                cat[i, j] = self.state["vocabularies"][field].get(
                    "__missing__" if value is None else value, 0
                )
            for j, field in enumerate(("temperature_c", "ph")):
                value = getattr(m, field)
                if (
                    value.status == "observed"
                    and self.state["numeric"][field]["observed_training_records"]
                ):
                    stats = self.state["numeric"][field]
                    num[i, j] = (value.value - stats["mean"]) / stats["scale"]
                num[i, 2 + j * 3 + STATUSES.index(value.status)] = 1
        return cat, num


def training_shuffle(metadata, proteins, train, seed):
    """Shuffle whole assay records in training only, across endpoints within protein."""
    result = list(metadata)
    rng = np.random.default_rng(seed)
    mapping = {}
    for protein in sorted({proteins[i] for i in train}):
        assays = sorted({metadata[i].assay_id for i in train if proteins[i] == protein})
        if len(assays) < 2:
            raise ValueError("Assay shuffle requires repeated endpoints per protein")
        shift = int(rng.integers(1, len(assays)))
        mapping.update({a: assays[(j + shift) % len(assays)] for j, a in enumerate(assays)})
    by_id = {m.assay_id: m for m in metadata}
    for i in train:
        result[i] = by_id[mapping[metadata[i].assay_id]]
    return result, mapping


class AssayHead(nn.Module):
    def __init__(self, representation_dimension, vocabulary, hidden=16):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(len(vocabulary.state["vocabularies"][f]) + 1, hidden, padding_idx=0)
            for f in CATEGORIES
        )
        self.numeric = nn.Linear(8, hidden, bias=False)
        nn.init.zeros_(self.numeric.weight)
        self.head = nn.Sequential(
            nn.Linear(representation_dimension + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, representation, categorical, numeric, mode="structured"):
        if (
            mode not in MODES
            or categorical.shape != (len(representation), len(CATEGORIES))
            or numeric.shape != (len(representation), 8)
        ):
            raise ValueError("Invalid metadata mode/tensor dimensions")
        if not torch.isfinite(representation).all() or not torch.isfinite(numeric).all():
            raise ValueError("Nonfinite inference features")
        context = torch.zeros(
            (len(representation), self.numeric.out_features), device=representation.device
        )
        for j, embedding in enumerate(self.embeddings):
            enabled = (
                (j < 7 or (j == 7 and mode == "with_assay_id"))
                if mode != "protein_proxy"
                else j == 8
            )
            if enabled and mode != "zero":
                context = context + embedding(categorical[:, j])
        if mode not in ("zero", "protein_proxy"):
            context = context + self.numeric(numeric)
        return self.head(torch.cat((representation, context), -1)).squeeze(-1).sigmoid()
