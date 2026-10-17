"""Measured GB1 count-based fitness, label-blind cohorts and factorial contrasts."""

import csv
import itertools
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from proteinmod.data import download, verify_file
from proteinmod.embeddings import fingerprint
from proteinmod.stability_data import mutant_sequence


@dataclass(frozen=True)
class LandscapeVariant:
    id: str
    sequence: str
    mutations: tuple[str, ...]
    target: float
    raw_fitness: float
    published_fitness: float
    count_input: int
    count_selected: int
    source_row: int
    cohort: bool


def acquire(config):
    spec = config["dataset"]
    archive = Path(spec["archive"]["path"])
    if archive.exists():
        verify_file(archive, spec["archive"])
    else:
        download(spec["archive"])
    path = Path(spec["path"])
    if path.exists():
        verify_file(path, spec)
    else:
        with zipfile.ZipFile(archive) as z:
            info = z.getinfo(spec["archive_member"])
            if info.file_size > 64 * 1024**2:
                raise ValueError("Unexpected GB1 member size")
            raw = z.read(info)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        verify_file(path, spec)


def source_rows(config):
    spec = config["dataset"]
    verify_file(spec["path"], spec)
    reference = config["reference"]["sequence"]
    full = config["full_reference"]["sequence"]
    positions = config["positions"]
    wt = "".join(reference[p - 1] for p in positions)
    raw = {}
    with Path(spec["path"]).open() as handle:
        for index, r in enumerate(csv.DictReader(handle), 1):
            genotype = r["Variants"]
            if len(genotype) != len(positions) or genotype in raw:
                raise ValueError("Duplicate/invalid genotype")
            tokens = tuple(
                f"{reference[p - 1]}{p}{a}"
                for p, a in zip(positions, genotype, strict=True)
                if reference[p - 1] != a
            )
            sequence = mutant_sequence(reference, tokens)
            if r["sequence"] != sequence + full[len(reference) :] or int(r["HD"]) != len(tokens):
                raise ValueError("GB1 source/reference mutation mapping differs")
            count_input, count_selected = int(r["Count input"]), int(r["Count selected"])
            if count_input < 10 or count_selected < 0:
                raise ValueError("Counts violate measured-data filter")
            published = float(r["Fitness"])
            if not np.isfinite(published) or published < 0:
                raise ValueError("Invalid fitness")
            raw[genotype] = {
                "tokens": tokens,
                "sequence": sequence,
                "input": count_input,
                "selected": count_selected,
                "published": published,
                "row": index,
            }
    if wt not in raw or raw[wt]["selected"] <= 0:
        raise ValueError("Measured WT required")
    denominator = raw[wt]["selected"] / raw[wt]["input"]
    floor = config["fitness_floor"]
    if not 0 < floor < 1:
        raise ValueError("Explicit positive detection floor required")
    rows = {}
    for genotype, r in raw.items():
        w = r["selected"] / r["input"] / denominator
        rows[genotype] = asdict(
            LandscapeVariant(
                genotype,
                r["sequence"],
                r["tokens"],
                float(np.log(max(w, floor))),
                w,
                r["published"],
                r["input"],
                r["selected"],
                r["row"],
                False,
            )
        )
        if len(r["tokens"]) != 1 and not np.isclose(w, r["published"], rtol=1e-8, atol=1e-10):
            raise ValueError("Non-single fitness does not reproduce published counts")
    if len(rows) != config["expected_source_rows"]:
        raise ValueError("Pinned source row count changed")
    return rows


def constituent_keys(genotype, positions, wildtype):
    changed = [j for j, (a, b) in enumerate(zip(genotype, wildtype, strict=True)) if a != b]
    for a, b in itertools.combinations(changed, 2):
        bg = "".join(wildtype[j] if j in (a, b) else c for j, c in enumerate(genotype))
        ga = "".join(bg[j] if j == b else c for j, c in enumerate(genotype))
        gb = "".join(bg[j] if j == a else c for j, c in enumerate(genotype))
        yield {
            "target": genotype,
            "a": ga,
            "b": gb,
            "background": bg,
            "position_pair": f"{positions[a]}-{positions[b]}",
            "background_mutations": len(changed) - 2,
        }


def collection(config):
    source = source_rows(config)
    selected = []
    for count, cap in sorted((int(k), v) for k, v in config["caps"].items()):
        candidates = sorted(
            [g for g, r in source.items() if len(r["mutations"]) == count],
            key=lambda g: fingerprint([config["split_salt"], g]),
        )
        selected.extend(candidates if cap is None else candidates[:cap])
    selected = set(selected)
    wt = "".join(config["reference"]["sequence"][p - 1] for p in config["positions"])
    if wt not in selected:
        raise ValueError("WT must be in training cohort")
    contrasts = []
    rejected = Counter()
    needed = set(selected)
    for genotype in sorted(selected):
        for c in constituent_keys(genotype, config["positions"], wt):
            keys = [c[k] for k in ("target", "a", "b", "background")]
            if any(g not in source for g in keys):
                rejected["missing_measured_constituent"] += 1
                continue
            if any(source[g]["raw_fitness"] < config["fitness_floor"] for g in keys):
                rejected["censored_constituent"] += 1
                continue
            needed.update(keys)
            contrasts.append(c)
    rows = [{**source[g], "cohort": g in selected} for g in sorted(needed)]
    return (
        rows,
        contrasts,
        {
            "source_count": len(source),
            "source_mutation_counts": dict(Counter(len(r["mutations"]) for r in source.values())),
            "cohort_counts": dict(Counter(len(source[g]["mutations"]) for g in selected)),
            "probe_only_rows": len(needed - selected),
            "eligible_measured_contrasts": len(contrasts),
            "contrast_exclusions": dict(rejected),
            "fitness_scale": "ln(max(same-assay count enrichment normalized to WT, 0.01))",
            "direct_epistasis_scale": "ln(wAB)-ln(wA)-ln(wB)+ln(wBG), all four >= floor",
            "published_single_column_used_as_target": False,
            "single_published_vs_count_median_absolute_difference": float(
                np.median(
                    [
                        abs(r["raw_fitness"] - r["published_fitness"])
                        for r in source.values()
                        if len(r["mutations"]) == 1
                    ]
                )
            ),
            "sampling": "all measured WT/singles/doubles; smallest label-blind SHA for triples/quads; evaluation probes never fitted",
        },
    )


def protocols(rows, config):
    candidates = [i for i, r in enumerate(rows) if r["cohort"]]
    anchors = [i for i in candidates if len(rows[i]["mutations"]) < 2]
    within = {p: [] for p in ("train", "validation", "test")}
    within["train"].extend(anchors)
    for count in (2, 3, 4):
        ordered = sorted(
            [i for i in candidates if len(rows[i]["mutations"]) == count],
            key=lambda i: fingerprint([config["split_salt"], "within", rows[i]["id"]]),
        )
        a, b = round(len(ordered) * 0.7), round(len(ordered) * 0.85)
        for p, ix in zip(within, (ordered[:a], ordered[a:b], ordered[b:]), strict=True):
            within[p].extend(ix)
    unseen = {p: [] for p in within}
    for i in candidates:
        positions = {int(t[1:-1]) for t in rows[i]["mutations"]}
        role = (
            "test"
            if set(config["test_position_pair"]) <= positions
            else "validation"
            if set(config["validation_position_pair"]) <= positions
            else "train"
        )
        unseen[role].append(i)
    result = [("C1", within), ("C2", unseen)]
    for protocol, roles in result:
        sets = [set(v) for v in roles.values()]
        if (
            any(not s for s in sets)
            or any(a & b for a, b in itertools.combinations(sets, 2))
            or set.union(*sets) != set(candidates)
        ):
            raise ValueError("Invalid combinatorial role partition")
        if any(i not in roles["train"] for i in anchors):
            raise ValueError("Single/WT anchors escaped training")
        seq = [{rows[i]["sequence"] for i in v} for v in roles.values()]
        if any(a & b for a, b in itertools.combinations(seq, 2)):
            raise ValueError("Repeated-sequence leakage")
        if protocol == "C2":
            for key in ("train", "validation"):
                if any(
                    set(config["test_position_pair"])
                    <= {int(t[1:-1]) for t in rows[i]["mutations"]}
                    for i in roles[key]
                ):
                    raise ValueError("Unseen pairing leakage")
        roles = {p: np.array(sorted(v), dtype=int) for p, v in roles.items()}
        yield protocol, roles


def additive(rows, train):
    wt = next(
        (r["target"] for i, r in enumerate(rows) if i in set(train) and not r["mutations"]), None
    )
    if wt is None:
        raise ValueError("Measured training WT required")
    singles = {
        r["mutations"][0]: r["target"] - wt
        for i, r in enumerate(rows)
        if i in set(train) and len(r["mutations"]) == 1
    }
    pred = np.array([wt + sum(singles.get(t, 0.0) for t in r["mutations"]) for r in rows])
    missing = [sum(t not in singles for t in r["mutations"]) for r in rows]
    return pred, {
        "wt": wt,
        "deltas": singles,
        "missing_single_counts": missing,
        "missing_single_policy": "zero delta, explicit missing count; never fabricated measurement",
    }
