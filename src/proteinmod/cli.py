"""Download is always explicit; validate and run operate offline."""

import argparse
import json
import tomllib
from pathlib import Path

from proteinmod.benchmark import run
from proteinmod.data import download, load_dataset, verified_reference, verify_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "validate", "run"))
    parser.add_argument("--config", default="configs/amylase.toml")
    parser.add_argument("--data", help="Exact local file; configured checksum still enforced")
    parser.add_argument("--output", help="Override output directory")
    args = parser.parse_args()
    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)
    if args.data:
        config["dataset"]["path"] = args.data
    if args.output:
        config["run"]["output"] = args.output
    try:
        if args.command == "download":
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
        else:
            print(json.dumps(run(config), indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(1, f"proteinmod: {error}\n")


if __name__ == "__main__":
    main()
