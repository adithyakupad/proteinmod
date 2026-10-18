"""M17 outcome-blind controlled roles and graph nulls; historical APIs unchanged."""

import hashlib
import math
from itertools import combinations, product

import numpy as np


def ordered(rows, indices, seed):
    return sorted(
        indices, key=lambda i: hashlib.sha256(f"m17:{seed}:{rows[i]['id']}".encode()).hexdigest()
    )


def balanced(rows, indices, fraction, seed):
    result = []
    for pid in sorted({rows[i]["protein_id"] for i in indices}):
        ix = [i for i in indices if rows[i]["protein_id"] == pid]
        result.extend(ordered(rows, ix, seed)[: math.ceil(len(ix) * fraction)])
    return np.array(sorted(result), dtype=int)


def study_roles(rows, families):
    proteins = {f: sorted({r["protein_id"] for r in rows if r["family"] == f}) for f in families}

    def indices(pids, partition):
        return np.array(
            [
                i
                for i, r in enumerate(rows)
                if r["protein_id"] in pids and r["partition"] == partition
            ],
            dtype=int,
        )

    result = []
    for j, family in enumerate(families):
        val_family = families[(j + 1) % len(families)]
        other = [f for f in families if f not in (family, val_family)]
        train_pids = sum([proteins[f] for f in other], [])
        base = {
            "train": indices(train_pids, "train"),
            "validation": indices(proteins[val_family], "validation"),
            "test": indices(proteins[family], "test"),
        }
        result.append(("H1", family, "G3", base))
        for pid in proteins[family]:
            partner = next(p for p in proteins[family] if p != pid)
            for protocol, replacement in (("G1", pid), ("G2", partner), ("G3", None)):
                pp = train_pids if replacement is None else [replacement] + train_pids[1:]
                rr = {
                    "train": balanced(rows, indices(pp, "train"), 0.5, 0),
                    "validation": base["validation"],
                    "test": indices([pid], "test"),
                }
                result.append(("H2", pid, protocol, rr))
        for family2 in other:
            result.append(
                (
                    "diversity",
                    family,
                    "narrow:" + family2,
                    {**base, "train": indices(proteins[family2], "train")},
                )
            )
        for pair in product(*(proteins[f] for f in other)):
            result.append(
                (
                    "diversity",
                    family,
                    "diverse:" + "+".join(pair),
                    {**base, "train": indices(pair, "train")},
                )
            )
    for _, _, _, roles in result:
        if any(not len(x) for x in roles.values()):
            raise ValueError("Empty roles")
        if any(set(roles[a]) & set(roles[b]) for a, b in combinations(roles, 2)):
            raise ValueError("Variant leakage")
    return result


def degree_null(edges, seed):
    """Undirected degree-preserving double-edge swaps, without loops/duplicates."""
    edges = np.asarray(edges)
    keys = {tuple(sorted(e)) for e in edges.T}
    if edges.shape[0] != 2 or len(keys) * 2 != edges.shape[1] or any(a == b for a, b in keys):
        raise ValueError("Need simple bidirectional graph")
    rng = np.random.default_rng(seed)
    keys = set(keys)
    swaps = 0
    for _ in range(10 * len(keys)):
        pool = sorted(keys)
        ix = rng.choice(len(pool), 2, replace=False)
        a, b = pool[ix[0]]
        c, d = pool[ix[1]]
        if len({a, b, c, d}) != 4:
            continue
        if rng.integers(2):
            c, d = d, c
        left, right = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if left in keys or right in keys:
            continue
        keys.remove(pool[ix[0]])
        keys.remove(pool[ix[1]])
        keys.update((left, right))
        swaps += 1
    result = np.array(
        [(a, b) for a, b in sorted(keys) for a, b in ((a, b), (b, a))], dtype=np.int64
    ).T
    if not np.array_equal(
        np.bincount(edges[1]), np.bincount(result[1], minlength=int(edges.max()) + 1)
    ):
        raise ValueError("Degree changed")
    return result, swaps
