# M18 external qualification: BLOCKED before label access

Baseline: `73f0666a87560dadffa76f34f5255856d2ce7459` (latest main at start).
Prospective admission and evaluation freeze: `c83f2a7`; original M17 hypotheses remain unchanged.
No external target values were inspected or parsed numerically. No models were fitted,
no predictions generated, no performance or independent-validation claim is made.

## Source-to-label evidence

The primary authors' [repository](https://github.com/lehner-lab/combinatorialcores/tree/f978897ef3a278796acc0e87e5cd82d4d0d10082)
identifies the Science study DOI 10.1126/science.adq3948 and
[GEO GSE266299](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE266299).
The GEO processing metadata explicitly identifies `fitness_raw` and `sigma_raw` as direct
DiMSum outputs of the newly constructed core-mutagenesis abundancePCA experiment.
`fitness_scaled` is a separate 0–1 transformation. These are yeast growth/abundance proxy
readouts, not folding ΔΔG or kcal/mol. Higher fitness is favorable. The precise raw numerical
normalization should be resolved before cross-assay calibration; no log base is assumed.
The biological system is yeast BY4741, methotrexate abundancePCA selection.
No per-row censoring bounds are supplied. Reported error is not automatically aleatoric variance.

The raw gzip SHA256 is
`4cd224d56a0004be70f1b33f055cb2a0d2846065497c94e4fcb8bb289a167b84`
(1,189,372 bytes). Only identifiers, genotype/core, mutation count and reference flag were
projected. The file remains ignored in `data/cache/m18/source/primary-scores.csv.gz`.
Notebook **code cells only**, never outputs, supplied literal WT DNA and separated the new
experiment from Tsuboyama comparison data. A mixed archive alone establishes neither
contamination nor independence. Zenodo record 11175470 declares CC-BY-4.0; its archive
MD5 is 716ab20afae843c6db145bbb72063453. This does not automatically grant a separate
license to every linked GEO file. GEO redistribution terms still require review.

| Candidate | Label lineage | Cohort decision |
|---|---|---|
| Primary GEO raw abundancePCA | ACCEPT lineage: direct experimental DiMSum output | BLOCKED: family similarity flags unresolved |
| FLIP2 Hydro target | BLOCKED: exact primary-column transformation not located | BLOCKED; matching counts/genotypes do not prove label mapping |
| FYN P06241 | New assay, but SH3 PF00018 already represented | REJECT family novelty |
| CspA P0A9X9 | New assay; CSD PF00313 | BLOCKED inherited similarity review |
| CI-2A P01053 | New assay; inhibitor PF00280 | BLOCKED inherited similarity review |

## Outcome-blind structural and family checks

Author WT DNA translation, not a variant consensus, gives CspA65aa and CI2 63aa.
Primary metadata counts: FYN9,972; CI2 6,463; CspA8,500 (24,935 total).
After excluding native WT, 8,499 CspA mutant genotypes remain; CI2 has6,463.
There are no duplicate eligible genotypes, noncanonical residues, outside-core mutations,
or mutation-count/core-string inconsistencies. Frozen hash ordering selects96/protein,
192 test-only **candidate** variants; this is not an admitted evaluation split.
PDB1MJC and3CI2 chainA map at reference identity1.0/coverage1.0 with65/63 graph nodes,
zero removed reference sites and no selected mutation mapping failures. Experimental
B-factors are not pLDDT, predictive uncertainty, or mutant rearrangements.

All18 unique reference sequences extracted from baseline configurations were compared:
36 external/historical pairs, unchanged M12 thresholds. Four global review flags:

| External | Historical reference | Identity | Both-coverage minimum | Ambiguous optimum |
|---|---|---:|---:|---|
| CspA | BBC1 SH3, configs/m12.json/proteins/0 | .416667 | .738462 | yes |
| CI2 | BBC1 SH3 | .428571 | .765625 | yes |
| CI2 | 1LP1, configs/m6.json/proteins/0 | .413043 | .730159 | yes |
| CI2 | 1A32, configs/m6.json/proteins/1 | .446809 | .746032 | yes |

Local alignments span only3–5 residues; these flags may reflect short-domain alignment
artifacts, not ancestry. Different Pfam annotations are supporting evidence, not sufficient
to waive the unchanged similarity gate. No score threshold was relaxed. Resolution requires
an outcome-blind profile/fold/evolutionary review of these exact pairs, or a different verified
cohort under a new prospective admission freeze. Do not treat unresolved flags as proven
contamination. Two candidate families also cannot satisfy M17's original >=3/4-family claim.

## Frozen experiments and stop decision

ADMISSION_PROTOCOL.md specifies H1–H5, source-only training/validation/calibration rotations,
seeds0–4, nested fractions, existing model classes, outcome-blind exclusions and metrics.
Phase B exact model/split admission is intentionally **not issued**: independence has not
passed. Test labels stay blinded. No low-label curves, paired-family differences, calibration,
selective prediction or H1–H5 confirmation can be reported for this external cohort.
M17 results and negative findings remain development evidence. Any later accepted study
must distinguish simultaneous assay, mutation-burden and family shifts. Native-unit conformal
coverage cannot transfer from stability-derived scores to abundance units without justification.

## Qualification matrix

| Gate | Status | Evidence |
|---|---|---|
| Latest-main baseline, hypothesis/admission freeze | PASS | commits above; ADMISSION_PROTOCOL.md |
| Primary raw-score experimental lineage | PASS | committed GEO metadata and WT DNA projection |
| Exact FLIP target lineage | BLOCKED | preprocessing/column transform unavailable |
| Reference and selected mutation structural mapping | PASS | blind-index.json;1MJC/3CI2 |
| New family independence | BLOCKED | four unresolved historical alignment flags |
| External experiments/calibration | NOT RUN | admission prerequisite failed; targets blinded |
| Historical preservation | PASS if verifier passes | historical_hashes.json covers baseline benchmarks/configs |
| Independent cold-source reproduction | PARTIAL | metadata evidence offline; reacquisition/graphs need network/BioPython/cache |
| Data redistribution licensing | PARTIAL | Zenodo CC-BY-4.0; GEO-specific terms unresolved |
| Release qualification | NOT QUALIFIED | external gate unresolved; no tag |

## Reproduction

From this branch, install `.[dev]`, then run:

```
ruff check .
ruff format --check .
pytest
python scripts/audit_m16.py
python scripts/verify_m17.py
python scripts/verify_m18.py
```

The last check needs no external data, model weights, network or target file.
To reconstruct the blind audit, place primary source metadata/WT references from committed
`evidence/` into `data/cache/m18/source/`; fetch the gzip from
`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE266nnn/GSE266299/suppl/GSE266299_sparse_DTS_core_mutagenesis_scores.csv.gz`
and verify the SHA above. Install `.[structure]`, run `python scripts/index_m18.py`
(network downloads pinned structures and produces geometry), then `python scripts/audit_m18_overlap.py`.
The committed blind index includes coordinate checksums. This is a documented source recipe,
not a claim that a clean independent operator has reproduced the source acquisition.
No target-unblinding runner is provided while admission remains blocked.
