# M18.2: fourth-family provenance and TEM1 admission audit

**Outcome: BLOCKED before confirmation, without candidate label access.**
Start: verified PR40 `256c2518bda891e5037376cfff351a852ec22010` (12/12 CI).
Prospective genotype/mask protocol: `4e20e52`, committed before the new file inspections.
M17 H1–H5, effect criterion0.02rho, original >=3/4-family criterion, seeds, controls,
model budgets and M0–M18.1 artifacts/negative findings remain unchanged. No architecture,
fit, prediction, external performance result, release or tag was added.

## Fourth measured family: GFP, primary column and immutable file identity

The author's [Figshare version1](https://figshare.com/articles/dataset/Local_fitness_landscape_of_the_green_fluorescent_protein/3102154/1)
DOI10.6084/m9.figshare.3102154.v1 explicitly links the
[Sarkisyan et al.2016 primary experiment](https://doi.org/10.1038/nature17995), raw sequencing
BioProject PRJNA282342, reference DNA, processed protein genotypes and score definitions.
It carries a **dataset-level CC-BY4.0 license**, not merely an article/code license.

Exact label lineage: barcode sequencing and four fluorescence-sorting replicates →
`barcodes_to_brightness.tsv` → grouping by amino-acid genotype →
`amino_acid_genotypes_to_brightness.tsv`, column **medianBrightness**. Column `std` is
across-barcode log-brightness variation, not predictive uncertainty or automatically
aleatoric variance; `uniqueBarcodes` is replication metadata. The metadata's file overview
uses the generic word averaged, while the column definition says median; no alternative
mean-based ProteinGym target is substituted. We would evaluate the deposited column
unchanged, in reported log-brightness units, higher favorable. A log base is not inferred
from the inspected metadata. Instrument gate values and author filtering procedures are
measurement definitions, not new score-dependent filters. No native ΔΔG or common scale
with abundancePCA/ampicillin resistance is claimed. Low-fluorescence assay floors and
processing noise would limit interpretation; no phenotype distribution was inspected.

Official download endpoints failed (500 responses; another returned empty bytes), but
metadata/file manifests remained available. We recovered a **byte-identity-verified mirror**
at BioAutoMATED commit63e7b52c3b61435080d5e6449a3f6ae32dc6ce5d. Both files match the
primary author's published file sizes and MD5s; local SHA256s are also fixed:

| Primary file | Bytes | Author-deposited MD5 | Local SHA256 |
|---|---:|---|---|
| reference FASTA |740|c1af632d736837b1d1518cb9aae53859|2559ee1afe9d0f41e03ea75db96448a34178c2dd597c78e08f01b1ee4ab5e248|
| protein-level TSV |2,334,128|bfff10393b2bedd687738885de79d76d|5581230b9888f9960708a4769ce1c9b4453561f0b5f0374525445b4a984d54d5|

The mirror supplies identical source bytes, not a new study, independent replication,
or permission to override Figshare's license. Failure logs and pinned mirror coordinates
are retained. Raw phenotype-bearing TSV is ignored, never committed or numerically parsed.

## Genotype-only mapping and reference conventions

Author DNA translation gives a237aa reference segment followed by termination. The FASTA
omits the vector/initiator methionine; we neither invent it nor call this segment a proven
complete expression construct. All212,388 source mutation tokens match its declared
zero-based WT positions. The decoder adds1 for internal positions in this segment; it
does **not** silently reuse ProteinGym's different initiator-inclusive numbering.

The54,025 primary rows contain1 WT,2,310 stop-substitution variants and51,714 unique
canonical non-WT sequences. All WT tokens are validated even in variants excluded for a
stop; no phenotype or barcode-count criterion is used. Mutation burden is recorded as
metadata, not balanced after outcomes. The proposed1GFL common mask retains43,964 genotypes,
excludes7,750 otherwise canonical genotypes, and produces96 deterministically hash-ordered
**candidate** test IDs. These are not admitted folds or evaluated observations.

GFP's official profile is PF01353.28 (clanCL0069), and the experimental GFP fold is distinct
from the historical families. It is a fourth independently **measured potential family**,
not a fourth fully admitted multimodal cohort. Existing encoder-pretraining exposure is
unknown; independent measurements do not imply unseen pretraining sequences.

## Full blinded homology audit

Compared each of four external references against all18 distinct historical references,
plus all six external/external pairs: **78** unchanged global/local diagnostics. Only the
four historical CspA/CI2 flags recur; GFP and TEM1 create no new original review flag.
M18/M18.1 original flags and bounded profile/fold adjudications are preserved, not relabeled.
The historical inventory includes structural references such as ubiquitin, not just M17's
supervised families. No GFP/Sarkisyan target lineage or exact reference alias is present in
historical benchmark/configuration evidence.

Ran **242** actual searches:22 exact references ×11 official family HMMs, PyHMMER0.12.1,
one CPU, prefilters disabled, published sequence AND domain GA cutoffs unchanged.
Each external candidate passes only its own profile in this panel:

| Candidate | Profile | Sequence bit score | GA sequence/domain |
|---|---|---:|---:|
| CspA |PF00313.29|113.2883|27/27|
| CI2 |PF00280.25|93.6350|24.5/24.5|
| TEM1 |PF13354.13|149.8618|22/22|
| GFP |PF01353.28|213.8306|25/25|

All40 non-own candidate/profile comparisons fail GA, and no external family profile
passes a historical construct. Historical-family profiles include SH3, chromo, homeobox,
WW, SpA, S15 and ubiquitin controls. This is an explicit11-profile panel, not exhaustive
Pfam/HHsearch or proof of zero remote ancestry. Full profile scores, domain spans,
versions, reference aliases and source hashes are archived; small-panel E-values are not
advertised as full-database significance.

## TEM1: sequence mismatch resolved, common mask defined

PDB**1XPB chainA** is an exact263aa mature match to canonical assay reference24–286.
It has no82/182 differences and all263 matching sites have Cα coordinates. By contrast,
1ZG4 declares the full286aa sequence but retains V82I/A182V and lacks signal-peptide
coordinates; merely having a full deposited SEQRES is not full experimental coverage.
Other sequence candidates1AXB/1FQG also fail exact-reference matching; their metadata
and the original1BTL evidence remain recorded.

The prospective1XPB mask excludes canonical1–23, including the start-codon translation
ambiguity. It leaves **4,997** of the5,434 table-encoded canonical non-WT substitutions.
Primary S2 numbering is mapped through the author's WT-site column; skipped Ambler
positions are not contiguous sequence offsets. Exclusions are437 signal-region substitutions,
286 WT-symbol rows and20 terminal-site rows. Positions82/182 can be exact in1XPB and are
not silently approved as mismatches in1BTL. No native signal-peptide geometry or mutant
rearrangement is invented. The full286aa assay reference is retained in metadata/model
context; it is not silently replaced by a mature reference.

**Remaining structural gate:** the unchanged strict M2 full-reference checks still reject
1XPB at coverage263/286=.919580, with required coverage1.0 and identity1.0. A necessary
263-site eligibility mask is not automatically an accepted full-reference extraction.
The committed protocol explicitly says not to pass off rejected full-reference M2 artifacts
as accepted graphs. No audit flag was flipped and no allowlist was added.

This is an admission/interface limitation under this protocol, not evidence that those
4,997 variants lack usable structural context. A separately reviewed masked-extraction
contract could validate coverage over eligible sites and provide compatible graph/descriptor
provenance for all models; that scope is not represented as if already accepted here.

## GFP structure search and mask limitations

Official RCSB exact-identity search returned2G2S/2G3D/2G16/2G5Z entities. Inspection showed
these entity1 sequences span only66 residues, so their search score is **not** evidence of
a whole237aa assay-reference match. Attempted full-chain M2 mappings fail the unique-alignment
check;2WUR and2B3P also fail it. These unsuccessful candidates are retained.

PDB**1GFL chainA** maps uniquely but differs at segment L63F/Q79R and lacks coordinates
at230–237. The strict extraction rejects both the unapproved differences and incomplete
coverage. The proposed mask contains227 exact, observed, canonical sites and excludes all
ten affected positions. Its43,964 genotype candidates are a possible shared intersection,
not an accepted model-ready graph. Graph neighborhoods/degree/RSA would also require explicit
masked-feature provenance; excluding only output rows does not erase incompatible graph nodes
or background-geometry differences. Sequence-only51,714 and structure-intersection43,964
counts are reported separately; there are no model comparisons across unmatched cohorts.
Experimental B-factors are not pLDDT, and neither is predictive uncertainty.

## TEM1 permission gate: newly resolved for noncommercial research

The initial workbook properties, shared text, supplement index and supplement XML have
no item-level license statement; those negative checks are preserved. The old PMC OA API
is retired, so404 was not treated as a permission denial. We followed the current
[PMC AWS documentation](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) and README instead.

The current authoritative
[PMC4032126.1 JSON inventory](https://pmc-oa-opendata.s3.amazonaws.com/metadata/PMC4032126.1.json)
states open-access/non-manuscript and `CC BY-NC`, and **explicitly lists the exact workbook**
in `media_urls`, MD5abc10a058a2adaba6ec7672fdda76ea7. That matches the cached XLSX byte-for-byte.
The PMC README documents that media/supplements are distributed when publisher licenses
permit, and the article identifies version3.0. This supplies a licensed-version-to-exact-item
linkage beyond assuming the article-text license alone covers any workbook.

Noncommercial research with attribution and NLM source/version notice passes this gate.
Commercial reuse is not granted. No workbook is redistributed in the PR; no automatic
license is asserted for future learned weights or predictions. Source-code MIT remains
separate. Raw GEO CspA/CI2 item-to-licensed-Zenodo equivalence remains the earlier qualified
licensing edge; this audit does not broaden it. See `permissions-matrix.json` for research,
raw redistribution, commercial use and output decisions separately.

## Final qualification and stop decision

| Gate | Status | Evidence |
|---|---|---|
| Verified PR40 and prospective protocol |PASS|256c2518…;4e20e52 |
| Fourth primary family/column lineage |PASS GFP|Author versioned deposition, MD5-matched source bytes |
| Genotype/reference audit |PASS within explicit author segment|54,025 GFP rows;5,434 TEM1 table substitutions |
| Four-family homology review |PASS bounded evidence|78 alignments;242 profiles; original flags retained |
| TEM1 mature sequence mismatch resolution |PASS|1XPB exact24–286, zero differences |
| TEM1 noncommercial workbook research permission |PASS|Licensed PMC version links exact workbook MD5 |
| TEM1 full-reference M2 graph admission |BLOCKED|263/286 coverage under unchanged strict check |
| GFP full-reference M2 graph admission |BLOCKED|Two differences, eight missing sites; other mappings ambiguous |
| Common graph-feature/row intersection for all frozen models |NOT ADMITTED|Masks supplied; full M2 graph prerequisites fail |
| CspA/CI2 raw GEO file-specific licensing qualification |PARTIAL, unchanged|No new licensed-archive byte-equivalence proof |
| Four fully admitted external families |BLOCKED|Potential families are not four usable multimodal test cohorts |
| Final folds/checkpoints/metrics manifest |NOT ISSUED|No Phase B freeze claimed without admission |
| Confirmation, paired effects, learning curves, calibration |NOT RUN|Targets stay blinded; no false numerical performance |
| Historical invariance, tests/Ruff/CI |See verification evidence|All parent evidence retained |
| Release |NOT QUALIFIED|No tag/release |

The exact blockers are model-ready masked structural admission and the carried-forward raw
GEO license qualification. This outcome does not lower the four-family denominator, alter
M17 hypotheses, invent a sequence-only confirmation or use tests to choose a model. Further
eligibility work must remain blinded, explicitly freeze a valid shared graph/mask contract,
and resolve source-item scope; only then can final roles, checkpoints/grids, IDs, exclusions
and metrics be committed and predictions frozen before any target access.

The [exposure ledger](exposure-ledger.json) retains the prior Mavor methods example exposure
and distinguishes instrument/metadata numbers from candidate outcomes. CspA, CI2, TEM1 and
GFP numerical phenotype/error cells were not inspected, decoded, filtered on or fitted.

## Reproduction and costs

Metadata acquired2026-10-18/19 UTC; URLs/checksums and unsuccessful lookups are retained.
No second independent operator or completely cache-free historical model reproduction is
claimed. CPU profile runtime/platform and exact repeated scores are recorded. No model
training/inference compute or MPS comparison was performed. All four candidate targets
remain quarantined; no target CSV/XLSX/ZIP or model weights are committed.

```
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
python scripts/audit_m16.py
python scripts/verify_m16_1.py
python scripts/verify_m17.py
python scripts/verify_m18.py
python scripts/verify_m181.py
python scripts/verify_m182.py
```

The final verifier is stdlib-only and requires no raw labels, model cache or network.
`python scripts/audit_m182_similarity.py` repeats all78 comparisons offline.
For profile replay, install `pyhmmer==0.12.1` separately, then run `python scripts/profile_m182.py`.
It uses22 committed references and11 exact CC0 HMMs (five preserved M18.1, six new).

Source reconstruction: `python scripts/acquire_m182_gfp.py` retrieves the pinned public
GitHub mirror using `gh`, checks primary MD5s before saving and leaves mismatched cached
files untouched. It needs network/authenticated `gh` access, not candidate outcomes.
To repeat the genotype census, place the checksum-pinned TEM1 workbook from the documented
PMC media URL in `data/cache/m181/source/TEM1-Data-S1-S4.xlsx`, then run
`python scripts/index_m182.py`. To repeat structural mapping, download the seven coordinate
files listed in `evidence/structure-mask-audits.json` from `https://files.rcsb.org/download/ID.cif`
to `data/cache/m182/source/`; the script checks their exact hashes, then run
`python scripts/map_m182.py`. These commands recreate admission evidence, not an admitted
experimental result. All original M17 development findings and M18 blocked records remain
byte-identical rather than being overwritten by the newer interpretation.
