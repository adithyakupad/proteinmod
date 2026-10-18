"""Independent small structure/native-label source regeneration, no warm cache reads."""

import json
import time
import tomllib
from pathlib import Path

from verify_structures import compare_formats

from proteinmod.data import download, write_json
from proteinmod.stability_data import load_stability
from proteinmod.structure_pipeline import run_structure
from proteinmod.structure_sources import acquire_structure


def main():
    root = Path("data/cache/m16.1/sources")
    output = Path("runs/m16.1/sources")
    root.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    structures = {}
    for name in ("ubiquitin_m2", "ubiquitin_m2_pdb"):
        config = tomllib.loads(Path(f"configs/{name}.toml").read_text())
        config["structure"]["path"] = str(root / Path(config["structure"]["path"]).name)
        config["structure_run"]["output"] = str(output / name)
        acquire_structure(config["structure"])
        structures[name] = run_structure(config, render=False)
    comparison = compare_formats(output / "ubiquitin_m2", output / "ubiquitin_m2_pdb")
    config = json.loads(Path("configs/m6.json").read_text())
    for source in config["sources"].values():
        source["path"] = str(root / Path(source["path"]).name)
        download(source)
    rows, audit = load_stability(config)
    assert len(rows) == 1800
    write_json(
        output / "cold-sources.json",
        {
            "structures": structures,
            "format_comparison": comparison,
            "stability_rows": len(rows),
            "stability_audit": audit,
            "sources": config["sources"],
            "seconds": time.perf_counter() - started,
            "scope": "Cold source adapter, not native transfer-head retraining",
        },
    )


if __name__ == "__main__":
    main()
