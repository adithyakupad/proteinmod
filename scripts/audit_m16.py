"""Offline release inventory: immutable evidence, local links, license and binary hygiene."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

BASELINE = "7cc4dbbe9c47355d1afa46f6f46f502394e9e999"
SECRET = re.compile(
    rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)(?![A-Za-z0-9])"
)


def local_links(document):
    missing = []
    for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
        if target.startswith(("https://", "http://", "#", "mailto:")):
            continue
        path = document.parent / unquote(target.split("#")[0])
        if not path.exists():
            missing.append(target)
            continue
        # macOS existence alone accepts case errors that fail on Linux/GitHub.
        normalized = Path(os.path.abspath(path))
        parent = Path(normalized.anchor)
        for part in normalized.parts[1:]:
            if part not in {p.name for p in parent.iterdir()}:
                missing.append(target)
                break
            parent /= part
    return missing


def audit():
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    tracked = [Path(p) for p in tracked if p]
    old = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", BASELINE, "--", "benchmarks", "configs"]
        )
        .decode()
        .splitlines()
    )
    historical = {}
    for name in old:
        original = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"])
        assert Path(name).read_bytes() == original, name
        historical[name] = hashlib.sha256(original).hexdigest()
    manifests = {}
    for manifest in sorted(Path("benchmarks").glob("*/artifact_hashes.json")):
        values = json.loads(manifest.read_text())
        for name, digest in values.items():
            assert hashlib.sha256((manifest.parent / name).read_bytes()).hexdigest() == digest
        manifests[str(manifest)] = len(values)
    documents = [Path("README.md"), Path("docs/M16.md"), Path("docs/RELEASE_NOTES_PROPOSED.md")]
    links = {str(p): local_links(p) for p in documents}
    assert not any(links.values()), links
    forbidden = {".pt", ".pth", ".npz", ".npy", ".safetensors", ".zip", ".gz"}
    binaries = [str(p) for p in tracked if p.suffix in forbidden]
    secrets = [str(p) for p in tracked if SECRET.search(p.read_bytes())]
    assert not binaries and not secrets, (binaries, secrets)
    paths = [str(p) for p in documents if re.search(r"/(?:Users|home)/[^/\s]+/", p.read_text())]
    assert not paths, paths
    return {
        "baseline": BASELINE,
        "historical_files_exact": len(historical),
        "historical_sha256": historical,
        "artifact_manifests_verified": manifests,
        "release_document_links_missing": links,
        "tracked_weight_array_archives": binaries,
        "credential_pattern_matches": secrets,
        "release_document_private_paths": paths,
        "source_license_present": any(
            Path(p).is_file() for p in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")
        ),
        "limits": "Pattern scan is not a complete security audit. External URL health and source-license approval are separate gates.",
    }


if __name__ == "__main__":
    result = audit()
    output = Path("runs/m16/source-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "historical_sha256"})
