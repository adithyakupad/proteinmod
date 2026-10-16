"""M12 bounded fresh-fit G1/G2/G3 experiments; no historical supervised weights."""

import argparse
import importlib.metadata
import json
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from proteinmod.data import checksum, download, verify_file, write_json
from proteinmod.embeddings import FrozenESM, fingerprint, site_features
from proteinmod.family_data import audit, collection, folds, similarity
from proteinmod.multiprotein_data import sequence_features
from proteinmod.multiprotein_experiments import per_protein, ridge_fit
from proteinmod.range_zip import selected_members
from proteinmod.residue_cache import residue_cache
from proteinmod.structural_graph import load_graph, variant_inputs
from proteinmod.structure_pipeline import run_structure, write_csv


def historical():
    return {
        str(p): checksum(p)
        for root in ("benchmarks", "configs")
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
        and not any(
            part.startswith("m")
            and part[1:].split(".")[0].isdigit()
            and int(part[1:].split(".")[0]) >= 12
            for part in p.parts
        )
    }


def acquire(config):
    missing = []
    for spec in config["proteins"]:
        d = spec["dataset"]
        if not Path(d["path"]).exists():
            missing.append(d)
        else:
            verify_file(d["path"], d)
        for name in ("classification", "structure"):
            d = spec[name]
            if Path(d["path"]).exists():
                verify_file(d["path"], d)
            else:
                download(d)
    evidence = {}
    if missing:
        blobs, evidence = selected_members(
            config["archive"]["url"], [s["archive_member"] for s in missing]
        )
        for spec in missing:
            from hashlib import sha256

            blob = blobs[spec["archive_member"]]
            if sha256(blob).hexdigest() != spec["sha256"]:
                raise ValueError("Dataset checksum mismatch")
            p = Path(spec["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
    return evidence


def inputs(config, prepare=False):
    rows, data = collection(config)
    matrices = {
        "sequence": np.array(
            [
                sequence_features(
                    r,
                    next(
                        p["reference"]["sequence"]
                        for p in config["proteins"]
                        if p["protein_id"] == r["protein_id"]
                    ),
                )
                for r in rows
            ]
        ),
        "esm_sites": np.zeros((len(rows), 961), np.float32),
        "structural": np.zeros((len(rows), 270), np.float32),
    }
    mask = np.zeros(len(rows), bool)
    groups = {}
    evidence = {}
    encoder = None
    for spec in config["proteins"]:
        pid, ref = spec["protein_id"], spec["reference"]["sequence"]
        settings = {
            "structure": spec["structure"],
            "reference": spec["reference"],
            "mapping": spec["mapping"],
            "geometry": {"contact_cutoff_a": 8.0, "sasa_points": 960, "dssp": "off"},
            "structure_run": {"output": spec["structure_artifacts"]},
        }
        manifest_path = Path(spec["structure_artifacts"]) / "manifest.json"
        if not manifest_path.exists():
            if not prepare:
                raise ValueError("Run --prepare first")
            run_structure(settings, render=False)
        if json.loads(manifest_path.read_text())["config"] != settings:
            raise ValueError("Graph configuration changed")
        graph = load_graph(spec["structure_artifacts"], ref)
        ix = np.array([i for i, r in enumerate(rows) if r["protein_id"] == pid])
        token_rows = [rows[i] for i in ix] + [{"id": pid + ":WT", "sequence": ref}]
        try:
            tokens, manifest, directory = residue_cache(
                spec, token_rows, config["embedding"], Path(config["cache"]) / "residues"
            )
        except ValueError as error:
            if not prepare or "Missing M5 residue cache" not in str(error):
                raise
            encoder = encoder or FrozenESM(config["embedding"], local_only=True)
            tokens, manifest, directory = residue_cache(
                spec,
                token_rows,
                config["embedding"],
                Path(config["cache"]) / "residues",
                True,
                encoder,
            )
        dynamic = []
        valid = []
        excluded = []
        for j, i in enumerate(ix):
            r = rows[i]
            positions = [int(t[1:-1]) - 1 for t in r["mutations"]]
            matrices["esm_sites"][i] = site_features(tokens[j], tokens[-1], positions)
            d, features, reasons = variant_inputs(r["mutations"], ref, graph)
            if reasons:
                excluded.append({"id": r["id"], "reasons": reasons})
            else:
                mask[i] = True
                valid.append(j)
                dynamic.append(d)
                matrices["structural"][i] = features
        valid = np.array(valid, dtype=int)
        groups[pid] = {
            "graph": graph,
            "dynamic": np.array(dynamic),
            "indices": ix[valid],
            "token_indices": valid,
            "residues": tokens,
        }
        evidence[pid] = {
            "graph": graph["audit"],
            "excluded": excluded,
            "residue_manifest": manifest,
            "residue_directory": str(directory),
            "residue_manifest_sha256": checksum(directory / "manifest.json"),
        }
        print("features", pid, "valid", len(valid), "excluded", len(excluded), flush=True)
    if any(not np.isfinite(v).all() for v in matrices.values()):
        raise ValueError("Nonfinite model features")
    return rows, matrices, mask, groups, {"data": data, "features": evidence}


def aggregate(rows, indices, pred):
    proteins = per_protein(rows, indices, pred)
    families = defaultdict(list)
    for pid, v in proteins.items():
        family = next(r["family"] for r in rows if r["protein_id"] == pid)
        families[family].append(v["spearman_rho"])
    family_scores = {
        f: float(np.mean([v if v is not None else 0 for v in values]))
        for f, values in sorted(families.items())
    }
    return {
        "per_protein": proteins,
        "per_family": family_scores,
        "family_macro_spearman": float(np.mean(list(family_scores.values()))),
    }


def run(config):
    import torch

    from proteinmod.hybrid_experiments import fit

    started = time.perf_counter()
    before = historical()
    output = Path(config["output"])
    if (
        Path("benchmarks").resolve() in output.resolve().parents
        or output.resolve() == Path("benchmarks").resolve()
    ):
        raise ValueError("Run outputs must not overwrite archived benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    prereg = json.loads(Path("benchmarks/m12/preregistration.json").read_text())
    if prereg["config_sha256"] != checksum("configs/m12.json"):
        raise ValueError("Preregistered config changed")
    similarities = similarity(config)
    rows, matrices, mask, groups, evidence = inputs(config)
    roles_evidence = []
    records = []
    predictions = []
    cache = Path(config["cache"]) / "checkpoints" / fingerprint(config)
    cache.mkdir(parents=True, exist_ok=True)

    def record(protocol, fold, cohort, roles, model, seed, pred, selection, elapsed):
        result = dict(
            protocol=protocol,
            fold=fold,
            cohort=cohort,
            model=model,
            seed=seed,
            selection=selection,
            elapsed_seconds=elapsed,
            **aggregate(rows, roles["test"], pred),
        )
        records.append(result)
        predictions.extend(
            {
                "protocol": protocol,
                "fold": fold,
                "cohort": cohort,
                "model": model,
                "seed": seed,
                "id": rows[i]["id"],
                "protein_id": rows[i]["protein_id"],
                "assay_id": rows[i]["assay_id"],
                "family": rows[i]["family"],
                "measured": rows[i]["target"],
                "prediction": float(v),
            }
            for i, v in zip(roles["test"], pred, strict=True)
        )
        write_json(output / "runs.json", records)
        write_csv(output / "predictions.csv", predictions)
        print(
            protocol,
            fold,
            cohort,
            model,
            seed,
            round(result["family_macro_spearman"], 4),
            round(elapsed, 2),
            flush=True,
        )

    for protocol, fold, original in folds(rows, config["family_order"]):
        for cohort in ("sequence_all", "structure_matched"):
            roles = {
                s: ix if cohort == "sequence_all" else ix[mask[ix]] for s, ix in original.items()
            }
            evidence_roles = audit(rows, roles, protocol)
            roles_evidence.append(
                {"protocol": protocol, "fold": fold, "cohort": cohort, "roles": evidence_roles}
            )
            # Balanced validation proteins per family make the existing macro selector family-balanced.
            counts = [
                len(v)
                for v in (
                    {
                        r["family"]: {
                            rows[i]["protein_id"]
                            for i in roles["validation"]
                            if rows[i]["family"] == r["family"]
                        }
                        for r in [rows[i] for i in roles["validation"]]
                    }
                ).values()
            ]
            if len(set(counts)) != 1:
                raise ValueError("Validation family weights unequal")
            for name, x in matrices.items():
                if name == "structural" and cohort == "sequence_all":
                    continue
                start = time.perf_counter()
                pred, selection = ridge_fit(x, rows, roles, config["alphas"])
                record(
                    protocol,
                    fold,
                    cohort,
                    roles,
                    name + "_ridge",
                    0,
                    pred,
                    selection,
                    time.perf_counter() - start,
                )
            if cohort == "sequence_all":
                continue
            for seed in config["seeds"]:
                for mode in config["modes"]:
                    start = time.perf_counter()
                    pred, selection, state = fit(
                        groups, rows, roles, config["training"], seed, mode
                    )
                    path = cache / f"{protocol}-{fold}-{mode}-{seed}.pt"
                    torch.save(state, path)
                    selection.update(checkpoint_path=str(path), checkpoint_sha256=checksum(path))
                    record(
                        protocol,
                        fold,
                        cohort,
                        roles,
                        mode,
                        seed,
                        pred,
                        selection,
                        time.perf_counter() - start,
                    )
    if before != historical():
        raise ValueError("M0-M11 historical artifacts changed")
    write_json(output / "rows.json", rows)
    write_json(
        output / "audit.json",
        dict(
            **evidence,
            roles=roles_evidence,
            similarity=similarities,
            historical_sha256=before,
            structural_mask=mask.tolist(),
            supervised_pretraining="None; every fold starts fresh. ESM-2 frozen unsupervised exposure unknown.",
            target_handling="Training-only per-assay midranks. Held-out measured labels used only for Spearman; no calibration to test scale.",
        ),
    )
    write_json(
        output / "environment.json",
        {
            "config_sha256": checksum("configs/m12.json"),
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "biopython")
            },
            "device": "deterministic CPU training; frozen ESM MPS when available",
        },
    )
    from proteinmod.family_report import report

    report(config, output)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/m12.json")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.acquire:
        print(acquire(config))
    elif args.prepare:
        inputs(config, True)
    elif args.report:
        from proteinmod.family_report import report

        report(config, Path(config["output"]))
    else:
        run(config)


if __name__ == "__main__":
    main()
