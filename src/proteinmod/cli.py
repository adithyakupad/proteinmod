"""Explicit data/model extraction; validation and regression operate offline."""

import argparse
import json
import tomllib
from pathlib import Path

from proteinmod.benchmark import run
from proteinmod.data import download, load_dataset, verified_reference, verify_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "download",
            "validate",
            "run",
            "extract",
            "run-m1",
            "run-m3",
            "structure-download",
            "structure-run",
        ),
    )
    parser.add_argument("--config", default="configs/amylase.toml")
    parser.add_argument("--data", help="Exact local file; configured checksum still enforced")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--baseline-only", action="store_true", help="M3 ridge comparisons only")
    parser.add_argument(
        "--local-only", action="store_true", help="Extract using cached model files only"
    )
    args = parser.parse_args()
    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)
    if args.data and args.command != "run-m3":
        section = "structure" if args.command.startswith("structure-") else "dataset"
        config[section]["path"] = args.data
    if args.output:
        section = (
            "learning_curve"
            if args.command in ("extract", "run-m1", "structure-download", "structure-run")
            else "run"
        )
        if args.command.startswith("structure-"):
            section = "structure_run"
        if args.command == "run-m3":
            section = "m3"
        config[section]["output"] = args.output
    try:
        if args.command == "run-m3":
            from proteinmod.structural_experiments import run_structural

            if args.data:
                raise ValueError("M3 dataset is pinned by its sequence_config; use that config")
            print(json.dumps(run_structural(config, args.baseline_only), indent=2, allow_nan=False))
        elif args.command == "structure-download":
            from proteinmod.structure_sources import acquire_structure

            print(acquire_structure(config["structure"]))
        elif args.command == "structure-run":
            from proteinmod.structure_pipeline import run_structure

            print(json.dumps(run_structure(config), indent=2, allow_nan=False))
        elif args.command == "download":
            print(download(config["dataset"]))
        elif args.command == "validate":
            verify_file(config["dataset"]["path"], config["dataset"])
            _, qc = load_dataset(
                config["dataset"]["path"], verified_reference(config.get("reference"))
            )
            output = Path(config["run"]["output"])
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "qc.json", qc)
            print(json.dumps(qc, indent=2))
        elif args.command == "extract":
            from proteinmod.embeddings import extract

            verify_file(config["dataset"]["path"], config["dataset"])
            reference = verified_reference(config.get("reference"))
            if reference is None:
                raise ValueError("M1 mutation-aware extraction requires a verified WT")
            rows, _ = load_dataset(config["dataset"]["path"], reference)
            from proteinmod.data import checksum

            print(
                extract(
                    config,
                    [row["sequence"] for row in rows],
                    reference,
                    checksum(config["dataset"]["path"]),
                    local_only=args.local_only,
                )
            )
        elif args.command == "run-m1":
            from proteinmod.experiments import run_experiments

            print(json.dumps(run_experiments(config), indent=2, allow_nan=False))
        else:
            print(json.dumps(run(config), indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"proteinmod: {error}\n")


if __name__ == "__main__":
    main()
