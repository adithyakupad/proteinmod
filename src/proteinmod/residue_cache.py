"""Label-free frozen residue tokens, independently cached without changing M1/M4."""

import json
import time
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, write_json
from proteinmod.embeddings import FrozenESM, fingerprint
from proteinmod.multiprotein_data import collection_rows


def residue_cache(spec, rows, settings, root, prepare=False, encoder=None):
    identity = {
        "schema": "m5-residue-tokens-v1",
        "reference": spec["reference"],
        "dataset_sha256": checksum(spec["dataset"]["path"]),
        "ordered_ids": [r["id"] for r in rows],
        "sequences_sha256": fingerprint([r["sequence"] for r in rows]),
        "embedding": settings,
        "extractor_sha256": checksum(__file__),
        "frozen_encoder_sha256": checksum(Path(__file__).with_name("embeddings.py")),
    }
    directory = Path(root) / fingerprint(identity)
    manifest_path = directory / "manifest.json"
    shape = (len(rows), len(spec["reference"]["sequence"]), settings["dimension"])
    if not manifest_path.exists():
        if not prepare:
            raise ValueError("Missing M5 residue cache; run prepare first")
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        encoder = encoder or FrozenESM(settings, local_only=True)
        temporary = directory / "residues.tmp.npy"
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=shape)
        size = settings["batch_size"]
        for start in range(0, len(rows), size):
            encoded = encoder.encode([r["sequence"] for r in rows[start : start + size]])
            if len(encoded) != len(rows[start : start + size]):
                raise ValueError("Incorrect encoder batch length")
            for offset, values in enumerate(encoded):
                if values.shape != shape[1:] or not np.isfinite(values).all():
                    raise ValueError("Invalid residue tokens")
                array[start + offset] = values
        array.flush()
        del array
        temporary.replace(directory / "residues.npy")
        write_json(
            manifest_path,
            {
                "contract": identity,
                "sha256": checksum(directory / "residues.npy"),
                "encoder": encoder.info,
                "elapsed_seconds": time.perf_counter() - started,
                "alignment": "BOS/EOS/padding removed; every residue ID checked by FrozenESM.encode",
            },
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["contract"] != identity
        or checksum(directory / "residues.npy") != manifest["sha256"]
    ):
        raise ValueError("Residue cache contract/hash mismatch")
    result = np.load(directory / "residues.npy", mmap_mode="r", allow_pickle=False)
    if result.shape != shape or result.dtype != np.float32:
        raise ValueError("Residue cache dimensions/dtype mismatch")
    return result, manifest, directory


def prepare(config, m4):
    encoder = FrozenESM(m4["embedding"], local_only=True)
    for spec in m4["collection"]:
        rows, _ = collection_rows(spec, m4["experiment"]["cap_per_partition"])
        _, manifest, _ = residue_cache(
            spec, rows, m4["embedding"], config["residue_cache"], True, encoder
        )
        print(spec["protein_id"], manifest["elapsed_seconds"], flush=True)
