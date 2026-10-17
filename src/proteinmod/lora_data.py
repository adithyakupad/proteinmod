"""Immutable label-free prefix cache and nested, protein-stratified training subsets."""

import json
import time
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, mutations, write_json
from proteinmod.embeddings import fingerprint, residue_tokens


def history():
    return {
        str(p): checksum(p)
        for root in ("benchmarks", "configs")
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
        and not any(
            part.startswith("m")
            and part[1:].split(".")[0].isdigit()
            and int(part[1:].split(".")[0]) >= 15
            for part in p.parts
        )
    }


def subset(rows, roles, fraction, salt):
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0,1]")
    train = []
    for pid in sorted({rows[i]["protein_id"] for i in roles["train"]}):
        ix = [i for i in roles["train"] if rows[i]["protein_id"] == pid]
        ix.sort(key=lambda i: fingerprint([salt, rows[i]["id"]]))
        train.extend(ix[: max(1, int(len(ix) * fraction))])
    return roles | {"train": np.array(sorted(train), dtype=int)}


def prefix_contract(settings, checkpoint_files, ids, sequences, boundary):
    return {
        "schema": "m15-frozen-prefix-v1",
        "settings": settings,
        "checkpoint_files": checkpoint_files,
        "ids": ids,
        "sequences_sha256": fingerprint(sequences),
        "boundary": boundary,
        "extractor_sha256": checksum(__file__),
        "encoder_sha256": checksum(Path(__file__).with_name("embeddings.py")),
    }


def prefix_cache(encoder, rows, reference, config, prepare=False):
    import torch

    sequences = [r["sequence"] for r in rows] + [reference]
    ids = [r["id"] for r in rows] + ["WT"]
    boundary = config["lora"]["layers"][0]
    identity = prefix_contract(
        config["embedding"], config["checkpoint_files"], ids, sequences, boundary
    )
    directory = Path(config["cache"]) / "prefixes" / fingerprint(identity)
    manifest_path = directory / "manifest.json"
    shape = (len(sequences), len(reference) + 2, config["embedding"]["dimension"])
    m = encoder.model
    if not manifest_path.exists():
        if not prepare:
            raise ValueError("Missing M15 prefix cache; run --prepare")
        directory.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        array = np.lib.format.open_memmap(
            directory / "prefix.tmp.npy", mode="w+", dtype="float32", shape=shape
        )
        maximum = 0.0
        for start in range(0, len(sequences), config["training"]["batch_size"]):
            seq = sequences[start : start + config["training"]["batch_size"]]
            if any(len(s) > encoder.max_residues for s in seq):
                raise ValueError("ESM length limit; mutation sites may not be truncated")
            tokens = encoder.tokenizer(
                seq,
                padding=True,
                truncation=False,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = tokens.pop("special_tokens_mask").numpy()
            with torch.inference_mode():
                extended = m.get_extended_attention_mask(
                    tokens["attention_mask"], tokens["input_ids"].shape
                )
                h = m.embeddings(input_ids=tokens["input_ids"])
                for layer in m.encoder.layer[:boundary]:
                    h = layer(h, attention_mask=extended)
                tail = h
                for layer in m.encoder.layer[boundary:]:
                    tail = layer(tail, attention_mask=extended)
                tail = m.encoder.emb_layer_norm_after(tail)
                whole = m(**tokens).last_hidden_state
                error = float(abs(tail - whole).max())
                if error > 1e-6:
                    raise ValueError("Cached prefix/tail diverges from full ESM")
                maximum = max(error, maximum)
            residue_tokens(
                tail.numpy(),
                tokens["input_ids"].numpy(),
                tokens["attention_mask"].numpy(),
                special,
                seq,
                encoder.tokenizer,
            )
            array[start : start + len(seq)] = h.numpy()
        array.flush()
        del array
        (directory / "prefix.tmp.npy").replace(directory / "prefix.npy")
        write_json(
            manifest_path,
            {
                "contract": identity,
                "sha256": checksum(directory / "prefix.npy"),
                "seconds": time.perf_counter() - started,
                "full_forward_max_error": maximum,
                "residue_tokens_verified": True,
                "supervised_labels_used": False,
            },
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest["contract"] != identity or checksum(directory / "prefix.npy") != manifest["sha256"]:
        raise ValueError("Prefix cache contract/checksum mismatch")
    prefix = np.load(directory / "prefix.npy", mmap_mode="r", allow_pickle=False)
    if prefix.shape != shape or prefix.dtype != np.float32 or not np.isfinite(prefix).all():
        raise ValueError("Invalid prefix cache shape/dtype/values")
    masks = []
    for row in rows:
        if set(row["mutations"]) != set(mutations(reference, row["sequence"])):
            raise ValueError("Mutation metadata differs from sequence")
        mask = np.zeros(len(reference), np.float32)
        positions = [int(t[1:-1]) - 1 for t in row["mutations"]]
        if (
            not positions
            or len(positions) != len(set(positions))
            or any(p < 0 or p >= len(reference) for p in positions)
        ):
            raise ValueError("Invalid unique mutation positions")
        for p in positions:
            if row["sequence"][p] == reference[p]:
                raise ValueError("Mutation mask does not identify a substituted residue")
        mask[positions] = 1
        masks.append(mask)
    return prefix, np.array(masks), manifest, str(directory)
