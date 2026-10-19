import hashlib
import json
import subprocess
from pathlib import Path

root = Path("data/cache/m182/source")
out = []
repo = "jackievaleri/BioAutoMATED"
rev = "63e7b52c3b61435080d5e6449a3f6ae32dc6ce5d"
for name, original, expected in [
    ("GFP-reference.fa", "avGFP_reference_sequence.fa", "c1af632d736837b1d1518cb9aae53859"),
    (
        "GFP-primary.tsv",
        "amino_acid_genotypes_to_brightness.tsv",
        "bfff10393b2bedd687738885de79d76d",
    ),
]:
    path = "clean_data/raw/avgfp/3102154/" + original
    try:
        b = subprocess.check_output(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github.raw+json",
                f"repos/{repo}/contents/{path}?ref={rev}",
            ],
            stderr=subprocess.PIPE,
        )
        digest = hashlib.md5(b).hexdigest()
        record = {
            "name": name,
            "repo": repo,
            "revision": rev,
            "path": path,
            "bytes": len(b),
            "md5": digest,
            "expected_md5": expected,
            "sha256": hashlib.sha256(b).hexdigest(),
            "matches_primary": digest == expected,
        }
        if digest == expected:
            (root / name).write_bytes(b)
        out.append(record)
    except (OSError, subprocess.CalledProcessError) as e:
        out.append({"name": name, "status": "ERROR", "error": str(e)})
(root / "mirror-check.json").write_text(json.dumps(out, indent=2) + "\n")
print(out)

assert len(out) == 2 and all(r.get("matches_primary") for r in out), (
    "Primary-file identity not established; original cache preserved"
)
