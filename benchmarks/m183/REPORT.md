# M18.3 — masked extraction passes; four-family confirmation remains BLOCKED

Issue [#43](https://github.com/adithyakupad/proteinmod/issues/43), stacked on verified
PR [#42](https://github.com/adithyakupad/proteinmod/pull/42) head
`d9579d6a51eaf200558d6a932e6072cff4cdab32` (12/12 successful checks).
The [preregistered contract](ADMISSION_PROTOCOL.md) was committed as `b794903`
before new qualification results. Research acquisitions occurred on 2026-10-19 UTC.
No new architecture, external target-value parsing, supervised fits, confirmatory
predictions, final folds/checkpoints manifest, tag or release was produced.

## What is implemented

`proteinmod.masked_structure.extract` is a separate `masked-reference-v1` adapter.
It uses the original M2 parser, unique alignment, conformer selection, canonical
identity checks, ASA and contact algorithms. It requires exact canonical identity
and observed CA at **every declared eligible site**; invalid or ambiguous masks
fail closed. Full assay reference numbering and every excluded row are retained.
It preserves the original strict full-reference audit **without changing its
accepted flag**. It does not produce a fake accepted M2 manifest or change M2's
loader. This distinction is covered by offline tests.

The adapter returns the existing in-memory graph/atom interface for M17's original
GNN, structural ridge and M14 edge annotator. No prediction head or training code
changes. Static nodes have 34 features, dynamic mutation inputs 25, structural
ridge 270, and annotated edges 29. Node/edge features describe deposited reference
geometry; no mutant relaxation or WT side-chain reconstruction is implied.

## Actual outcome-blind geometry and matched cohort results

| Family / template chain A | Full reference | Eligible nodes | Undirected edges | Heavy atoms at nodes | Components | Matched candidate IDs |
|---|---:|---:|---:|---:|---:|---:|
| CspA PF00313 / 1MJC | 65 | 65 | 298 | 486 | 1 | 96 |
| CI2 PF00280 / 3CI2 | 63 | 63 | 288 | 505 | 1 | 96 |
| TEM-1 PF13354 / 1XPB | 286 | 263 | 1,355 | 2,027 | 1 | 96 |
| GFP PF01353 / 1GFL | 237 | 227 | 1,145 | 1,798 | 1 | 96 |

All eligible-site identities and CA coverages are 1.0, with zero isolated nodes.
CspA/CI2 full masks reproduce the old M2/M3 input tensors **exactly**. TEM-1's
absent signal residues 1–23 remain in its full reference; the structure represents
only mature positions 24–286. GFP excludes mismatched 63/79 and missing 230–237.
The original full-reference M2 rejections for TEM-1 and GFP remain byte-for-byte
identical to PR #42. No admission threshold changed.

The [machine-readable audit](structural-cohort-audit.json) independently checks
coordinates against raw mmCIF atom-site records, all-pair spatial contacts,
peptide connectivity, edge symmetry, dimensions, finite features, chemical atom
availability and mutation-order invariance. It also reconstructs every selected
sequence from its mutation tokens. All four frozen genotype cohorts pass the
structural interface: **384/384** candidate IDs, zero additional exclusions.
These are candidate IDs, not a final evaluation fold or a claim of four-family
scientific admission. Structural ridge/original GNN/edge GNN share their entire
intersection; sequence-compatible full references remain untruncated. Fresh ESM
embedding extraction and supervised checkpoint generation were deferred behind
the admission gate; no PLM performance is claimed.

CspA/CI2 genotypes retain their M18 SHA-selected IDs. TEM-1/GFP retain M18.2 IDs
from the 4,997/43,964 mask-compatible canonical variants, respectively. Nothing
is selected by phenotype, error, barcode count or observed model performance.
The canonical counts/mapping failures and prior negative records are not rewritten.

ASA uses 960 Shrake–Rupley points, a 1.4 Å probe and the entire selected deposited
model (including other chains/ligands; no water/H/D/symmetry mates), as in M2.
Contacts/degrees are recomputed on the eligible induced graph. CspA, CI2, TEM-1
and GFP RSA maxima are 0.884421, 0.769732, 0.824309 and 1.016179; RSA is not clipped.
All nodes have ASA/RSA, contact degrees and experimental B-factors. All lack pLDDT
by design: experimental B-factors are **not** pLDDT or predictive uncertainty.

GFP reference site 64 is missing a deposited oxygen atom. The existing M14
backbone-polar validity mask marks **12 directed edges invalid**; missing atoms
are never replaced or silently interpreted as negative interactions. All other
examined chemistry validity flags pass. Candidate interactions remain geometric
candidates, not experimentally established hydrogen bonds.

## GFP background remains a scientific blocker

The two excluded mismatched chain-A residues have **20 spatial contacts with 20
eligible residues**. Deleting only these two residues from the deposited model
changed no eligible ASA at this 960-point sampling (maximum absolute difference
0.0 Å²; chain B retained). That is a bounded occlusion sensitivity calculation,
not evidence that L63F/Q79R have no conformational or functional consequences.
The complete list of neighbors and affected candidate IDs is archived in the
audit. Masking removes those contact contributions from model degrees, but cannot
reconstruct the missing WT geometry or demonstrate absence of propagated effects.
Thus GFP's local extraction passes while its WT-background admission stays blocked.

A new 95%-identity sequence query returned 732 polymer-entity hits; 30 ranked hits
were archived and the top 13 entities inspected. None contains the exact complete
assay reference. Top-ranked 5N9O has S64T despite a normalized search score of 1.0;
its actual coordinate mapping fails the unchanged unique-alignment check. Other
high-ranked hits retain L63F and/or S64T, including fusion constructs. This is a
bounded search, not proof that no suitable GFP structure exists. Primary search
request/response, entity sequences, metadata hashes and the actual 5N9O mapping
failure are in [the template audit](evidence/GFP-closer-template-audit.json).
No alternative mask was chosen to improve outcome performance.

## CspA/CI2 GEO item licensing remains unverified

The primary deposited raw item is
[GSE266299_sparse_DTS_core_mutagenesis_scores.csv.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE266nnn/GSE266299/suppl/GSE266299_sparse_DTS_core_mutagenesis_scores.csv.gz),
SHA256 `4cd224d56a0004be70f1b33f055cb2a0d2846065497c94e4fcb8bb289a167b84`.
Its `fitness_raw` column is the previously traced experimental DiMSum abundance
readout. No raw/scaled fitness or error values were decoded in this work.

The author's [Zenodo record 11175470](https://zenodo.org/records/11175470) explicitly
licenses its deposited archive CC-BY-4.0. We retrieved the complete **3,638-entry
ZIP directory** using **487,341 ranged bytes**, rather than downloading 1.33 GB.
No member has the GEO table's uncompressed size **4,091,026 bytes** and CRC32
**3681033130**; no member has the raw GEO filename. The archive contains a distinct
`zenodo/Fig1/Sparse_DTS_Cores_fitness_replicates.txt` (19,073,162 bytes), not an
established byte-identical raw GEO export. Renaming or matching the paper title
cannot establish column-level license lineage. Derived equivalence is still
unverified. The whole archive MD5 is publisher-declared, **not** locally verified;
member contents/replicate outcomes were not opened. The complete directory and
[opaque-byte comparison](evidence/GEO-Zenodo-item-audit.json) are preserved.

The official [NCBI molecular-data policy](https://www.ncbi.nlm.nih.gov/home/about/policies/)
places no NCBI restrictions but explicitly says it cannot transfer third-party
rights or provide unrestricted permission. Direct attempts to obtain the complete
[GEO-specific disclaimer](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html)
returned a reCAPTCHA challenge. A search excerpt says NCBI places no restrictions;
it is not a retrieved complete file-specific license. This is **not a finding of
prohibited research use**, but insufficient evidence for this preregistered
source-permission gate. No license was inferred from ProteinMod MIT or from an
unlinked archive. No message requesting author permission was sent.

| Source | Current noncommercial research admission | Commercial / raw redistribution |
|---|---|---|
| GFP Figshare exact MD5-linked files | PASS, CC-BY-4.0 attribution | Same license; no table redistributed |
| TEM-1 exact PMC-linked workbook | PASS, CC-BY-NC 3.0 and NLM notice | Commercial permission not granted; no workbook redistributed |
| GEO CspA/CI2 raw `fitness_raw` table | PARTIAL: exact item terms/equivalence unresolved | Not asserted; no raw table redistributed |
| Licensed Zenodo archive itself | CC-BY-4.0 deposit verified | Does not automatically license a different GEO file |

To clear this gate requires authoritative terms specifically covering this GEO
item, depositor permission linking the item/columns, or a traceable licensed export
with a verified blind lineage. To clear GFP requires a suitable reference-compatible
structure, or defensible prospective outcome-blind background qualification; local
masking and a zero ASA deletion delta alone do not establish that qualification.
Neither a reduced family denominator nor threshold relaxation is a remedy.

## Qualification matrix and interpretation

| Gate | Result | Evidence / remaining limitation |
|---|---|---|
| Verified PR42 and preregistration before new geometry | PASS | Exact parent and `b794903` contract |
| Original M2 and historical M0–M18.2 evidence preservation | PASS | Parent/source/artifact hashes; strict rejections retained |
| Masked extraction / atoms / contacts / frozen structural dimensions | PASS | Four real coordinate files, 384 matched IDs |
| Mutation-order invariance / missing chemistry / confidence semantics | PASS | All selected genotypes; explicit oxygen/validity masks |
| Four-family provenance and homology adjudication | PARTIAL | Four candidates; earlier 4 numerical flags retained with bounded profile/fold adjudication, no new zero-ancestry claim |
| TEM-1 mature-domain mapping and research permission | PASS | 263 sites; earlier exact workbook licensed-item linkage |
| GFP WT-background compatibility | BLOCKED | Two mismatches still contact eligible environment |
| GEO raw-item permission linkage | BLOCKED | No byte-identical ZIP member; complete item-specific terms unverified |
| Cold-source independent reproduction | PARTIAL | Same operator/environment, pinned existing caches/projections; deterministic local replay |
| External final manifest / once-only confirmation | NOT RUN | Admission blocked; outcomes remain blinded |
| Tests / Ruff / historical guards | PASS | See verification.json and exact-head PR checks |
| Release qualification | BLOCKED | Scientific admission gates; no tag/release |

M17 H1–H5, four-family denominator, >=3/4 family agreement, original similarity
thresholds, fixed model choices, validation-only selection and negative historical
findings remain unchanged. No external Spearman, uncertainty calibration or
independent-validation success can be inferred from this geometry audit. Assay
abundance, ampicillin resistance and fluorescence are not folding ΔΔG. No OOD
stability predictions or fabricated paired thermodynamic labels enter this work.

## Reproduction

From repository root, install the existing `.[dev,m3,esm]` environment. No new
runtime dependency is introduced. First verify archived evidence without touching
any target-bearing source (also runs in CI):

```sh
python scripts/audit_m16.py
python scripts/verify_m16_1.py
python scripts/verify_m17.py
python scripts/verify_m18.py
python scripts/verify_m181.py
python scripts/verify_m182.py
python scripts/verify_m183.py
ruff check .
ruff format --check .
pytest -q
```

For the real CPU geometry replay, retain/acquire the checksum-pinned coordinate
files listed in `structural-cohort-audit.json` at their recorded cache paths.
Primary downloads are `https://files.rcsb.org/download/{1MJC,3CI2,1XPB,1GFL}.cif`;
verify exact recorded SHA256 before use. The two old CspA/CI2 graph manifests and
artifacts are additionally required for tensor equality with M2, as documented by
M18's reproduction instructions. Selected genotype projections are committed;
this geometry script never opens external outcome tables.

```sh
python scripts/audit_m183.py --output /tmp/proteinmod-m183-replay
python scripts/qualify_m183_sources.py --output /tmp/proteinmod-m183-source-replay
```

The second command uses Zenodo HTTP ranges plus the pinned GEO gzip in
`data/cache/m18/source/primary-scores.csv.gz`; it hashes opaque table bytes and
reads only the known header, never phenotype values. It fails if HTTP ranges or
archive metadata change. Compare `families` exactly with the committed geometry
audit; runtime is observational, not a reproducibility key. The committed replay
records tensor equality and maximum hash differences. All computations are CPU,
with versions/runtime recorded in the audit; there is no randomness or new model
seed. M17 seeds/config/checkpoints were not changed or newly selected. The GFP
search request can be posted to the recorded RCSB endpoint; live hit ordering can
change, so retain pinned response/entity/coordinate hashes. This is local replay
with cache dependencies, not a separate operator's cold-source replication.
