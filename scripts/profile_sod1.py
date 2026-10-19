"""Same published GA thresholds and filter-disabled search as the historical profile audit."""

import argparse
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import pyhmmer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("benchmarks/m185/profile-audit.json"))
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError("Choose a new profile output; preserve frozen qualification evidence")
panel = json.loads(Path("benchmarks/m182/reference-panel.json").read_text())
refs = {r["id"]: r["sequence"] for r in panel["historical"] + panel["external_candidates"]}
refs["SOD1"] = json.loads(Path("configs/m185.json").read_text())["reference"]
profile_paths = sorted(Path("benchmarks/m181/evidence/profiles").glob("*.hmm")) + sorted(
    Path("benchmarks/m182/evidence/profiles").glob("*.hmm")
)
profile_paths.append(Path("benchmarks/m185/evidence/PF00080.hmm"))
assert len(refs) == 23 and len(profile_paths) == 12
alphabet = pyhmmer.easel.Alphabet.amino()
seqs = [
    pyhmmer.easel.TextSequence(name=k.encode(), sequence=v).digitize(alphabet)
    for k, v in refs.items()
]
out = []
t = time.monotonic()
for profile_path in profile_paths:
    pf = profile_path.stem
    data = profile_path.read_bytes()
    with pyhmmer.plan7.HMMFile(io.BytesIO(data)) as f:
        hmm = f.read()
    ga = tuple(hmm.cutoffs.gathering)
    hits = next(
        pyhmmer.hmmer.hmmsearch(
            hmm,
            seqs,
            cpus=1,
            E=1e9,
            domE=1e9,
            incE=1e9,
            incdomE=1e9,
            F1=1,
            F2=1,
            F3=1,
            bias_filter=False,
        )
    )
    byname = {h.name.decode() if isinstance(h.name, bytes) else h.name: h for h in hits}
    for name in refs:
        h = byname.get(name)
        dom = []
        if h:
            for d in h.domains:
                a = d.alignment
                dom.append(
                    {
                        "score": d.score,
                        "hmm_from": a.hmm_from,
                        "hmm_to": a.hmm_to,
                        "target_from": a.target_from,
                        "target_to": a.target_to,
                        "i_evalue": d.i_evalue,
                    }
                )
        out.append(
            {
                "profile": pf,
                "accession": str(hmm.accession),
                "length": hmm.M,
                "GA": ga,
                "sequence": name,
                "score": h.score if h else None,
                "evalue": h.evalue if h else None,
                "domains": dom,
                "passes_GA": bool(h and h.score >= ga[0] and any(d["score"] >= ga[1] for d in dom)),
                "profile_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
result = {
    "tool": "pyhmmer",
    "version": pyhmmer.__version__,
    "platform": platform.platform(),
    "seconds": time.monotonic() - t,
    "cpus": 1,
    "references": refs,
    "search": "23 references against12 explicit family profiles; filters disabled; published GA cutoffs unchanged; not an exhaustive all-Pfam scan",
    "pairs": out,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
for r in out:
    if r["passes_GA"]:
        print(r["profile"], r["sequence"], r["score"], r["passes_GA"])
