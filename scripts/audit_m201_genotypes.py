"""Genotype/reference/structure compatibility only; no outcome columns are available."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from proteinmod.structural_graph import load_graph, variant_inputs


def main():
    root = Path("benchmarks/m201")
    audits = json.loads((root / "evidence/member-audit.json").read_text())
    tables = [r for r in audits if "genotype_projection" in r]
    assert len(tables) == 3
    assert len({r["member_sha256"] for r in tables}) == 1
    for record in tables:
        path = Path(record["genotype_projection"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["projection_sha256"]
    rows = json.loads(Path(tables[0]["genotype_projection"]).read_text())
    references = json.loads(Path("benchmarks/m18/evidence/author-references.json").read_text())
    old = json.loads(Path("benchmarks/m18/evidence/blind-index.json").read_text())["selected"]
    results = []
    # Author Fig2 code: zero-based core positions, translated assay constructs.
    for pid, prefix, positions in [
        ("P0A9X9", "MT", [4, 16, 25, 27, 46, 48, 62]),
        ("P01053", "KT", [11, 18, 27, 45, 47, 49, 55]),
    ]:
        reference = references[pid]
        graph = load_graph(f"data/cache/m18/graphs/{pid}", reference)
        mapped = {n["reference_position"] for n in graph["nodes"]}
        eligible = {}
        exclusions = Counter()
        burdens = Counter()
        candidates = [r for r in rows if r["aa_seq"].startswith(prefix)]
        for row in candidates:
            sequence = row["aa_seq"]
            if len(sequence) != len(reference) or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                exclusions["frame_or_noncanonical"] += 1
                continue
            changed = [i for i, (a, b) in enumerate(zip(reference, sequence)) if a != b]
            if not changed:
                exclusions["WT"] += 1
                continue
            if not set(changed) <= set(positions):
                exclusions["outside_author_core"] += 1
                continue
            assert all(i + 1 in mapped for i in changed)
            mutations = [f"{reference[i]}{i + 1}{sequence[i]}" for i in changed]
            if sequence in eligible:
                exclusions["duplicate_AA"] += 1
                continue
            eligible[sequence] = mutations
            burdens[len(changed)] += 1
        selected = [r for r in old if r["protein_id"] == pid]
        matched = [r for r in selected if r["sequence"] in eligible]
        widths = set()
        for row in matched:
            dynamic, features, reasons = variant_inputs(row["mutations"], reference, graph)
            reverse, reversed_features, errors = variant_inputs(
                list(reversed(row["mutations"])), reference, graph
            )
            assert not reasons and not errors
            np.testing.assert_array_equal(dynamic, reverse)
            np.testing.assert_array_equal(features, reversed_features)
            widths.add((dynamic.shape[1], len(features)))
        results.append(
            {
                "protein": pid,
                "reference": reference,
                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                "author_core_positions_one_based": [p + 1 for p in positions],
                "source_genotypes_prefix": len(candidates),
                "eligible_unique_nonWT": len(eligible),
                "exclusions": dict(exclusions),
                "mutation_burden": dict(burdens),
                "missing_nt_seq": sum(r["nt_seq"] in {"NA", "", "nan"} for r in candidates),
                "prior_selected_ids": len(selected),
                "prior_selected_genotypes_present": len(matched),
                "feature_widths": sorted(widths),
                "mutation_order_exact": True,
                "mapped_reference_residues": len(mapped),
                "new_selected_set_issued": False,
                "genotype_set_sha256": hashlib.sha256(
                    "\n".join(sorted(eligible)).encode()
                ).hexdigest(),
            }
        )
    output = {
        "scope": "Source compatibility audit, not final study admission or score equivalence",
        "all_three_processed_member_bytes_identical": True,
        "genotype_rows_total": len(rows),
        "families": results,
        "outcome_or_flag_values_decoded": False,
        "models_fit": 0,
        "predictions": 0,
    }
    with (root / "genotype-structure-audit.json").open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
