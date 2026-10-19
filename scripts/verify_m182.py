"""Stdlib-only blinded admission verification, never opening candidate phenotypes."""

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def verify():
    root = Path("benchmarks/m182")
    for manifest in ("artifact_hashes.json", "historical_hashes.json"):
        for name, expected in json.loads((root / manifest).read_text()).items():
            path = root / name if manifest == "artifact_hashes.json" else Path(name)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
    panel = json.loads((root / "reference-panel.json").read_text())
    assert len(panel["historical"]) == 18 and len(panel["external_candidates"]) == 4
    refs = {r["id"]: r["sequence"] for r in panel["historical"] + panel["external_candidates"]}
    assert len(set(refs.values())) == 22
    pairs = json.loads((root / "similarity-audit.json").read_text())["pairs"]
    assert len(pairs) == 78 and sum(r["original_flag"] for r in pairs) == 4
    assert all(not r["original_flag"] for r in pairs if r["a"] in ("GFP", "TEM1"))
    profiles = json.loads((root / "reciprocal-profiles.json").read_text())
    assert profiles["references"] == refs and len(profiles["pairs"]) == 242
    assert len({(r["profile"], r["sequence"]) for r in profiles["pairs"]}) == 242
    own = {"GFP": "PF01353", "TEM1": "PF13354", "P0A9X9": "PF00313", "P01053": "PF00280"}
    external_profiles = set(own.values())
    for row in profiles["pairs"]:
        p = root / "evidence/profiles" / (row["profile"] + ".hmm")
        if not p.exists():
            p = Path("benchmarks/m181/evidence/profiles") / p.name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["profile_sha256"]
        passes = (
            row["score"] is not None
            and row["score"] >= row["GA"][0]
            and any(d["score"] >= row["GA"][1] for d in row["domains"])
        )
        assert row["passes_GA"] == passes
        if row["sequence"] in own:
            assert passes == (own[row["sequence"]] == row["profile"])
        elif row["profile"] in external_profiles:
            assert not passes
    audits = json.loads((root / "evidence/structure-mask-audits.json").read_text())
    assert len(audits) == 7 and all(not a["audit"]["accepted"] for a in audits)
    masks = {}
    for pid, pdb, expected in [("GFP", "1GFL", 227), ("TEM1", "1XPB", 263)]:
        audit = next(a for a in audits if a["pdb"] == pdb)
        masks[pid] = {
            r["reference_position"]
            for r in audit["residues"]
            if r["status"] == "mapped"
            and r["has_ca"]
            and not r["noncanonical"]
            and r["reference_aa"] == r["structure_aa"]
        }
        assert len(masks[pid]) == expected
    tem = next(a for a in audits if a["pdb"] == "1XPB")
    assert tem["audit"]["differences"] == [] and masks["TEM1"] == set(range(24, 287))
    for name, pid in [("GFP-genotype-audit.json", "GFP"), ("TEM1-common-mask-audit.json", "TEM1")]:
        index = json.loads((root / name).read_text())
        assert not index["target_cells_accessed"] and len(index["selected_candidates"]) == 96
        seen = set()
        for row in index["selected_candidates"]:
            assert row["candidate_role"] == "test_only_if_admitted" and "target" not in row
            assert row["id"] not in seen
            seen.add(row["id"])
            seq = list(refs[pid])
            for token in row["mutations"]:
                pos = int(token[1:-1])
                assert pos in masks[pid] and refs[pid][pos - 1] == token[0]
                seq[pos - 1] = token[-1]
            assert "".join(seq) == row["sequence"]
    meta = json.loads((root / "evidence/PMC4032126.1.json").read_text())
    link = next(u for u in meta["media_urls"] if "Data_S1-S4.xlsx" in u)
    assert meta["is_pmc_openaccess"] and meta["license_code"] == "CC BY-NC"
    assert parse_qs(urlsplit(link).query)["md5"] == ["abc10a058a2adaba6ec7672fdda76ea7"]
    item = json.loads((root / "evidence/TEM1-licensed-item-linkage.json").read_text())
    assert (
        item["exact_item_linkage_verified"]
        and item["cached_workbook_md5"] == item["supplement_md5"]
    )
    q = json.loads((root / "qualification.json").read_text())
    assert q["admission"] == "BLOCKED" and len(q["potential_families"]) == 4
    assert q["models_fit"] == q["predictions_generated"] == 0
    assert not q["candidate_target_cells_accessed"] and not q["final_folds_issued"]
    print(
        "M18.2 evidence PASS: fourth primary family, 78 alignments, 242 profiles, exact TEM1 licensed-item linkage; structural admission BLOCKED"
    )


if __name__ == "__main__":
    verify()
