# M12 source provenance and family definitions

Experimental measurements: [Tsuboyama et al., Nature 620, 434–444 (2023)](https://www.nature.com/articles/s41586-023-06328-6),
DOI 10.1038/s41586-023-06328-6; [original deposited data](https://zenodo.org/records/7992926).
The cDNA-display proteolysis study experimentally estimates domain stability. These
are experimental measurements with an assay-based inference step, not simulated labels.

M12 consumes the curated [ProteinGym v1.3 substitution archive](https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip).
HTTP range extraction fetched only the eight selected CSV members: **296,298 bytes**
including archive directory/range reads, versus 43,021,128 bytes for the full archive.
ETag `"2907348-6329ff6970941"`; every extracted member has CRC validation and a pinned
SHA256 in `configs/m12.json`. Only the rows used by the benchmark and held-out predictions
are archived; full raw tables and binary features remain ignored caches.

The reference catalog is pinned at ProteinGym Git revision
`144fe22b07dfaeec2b366f2346203a9838a55b4c`, catalog SHA256
`a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`.
Independent assay WT/construct sequences, sequence hashes, source filenames, article
citation, raw phenotype column (`ddG_ML_float`) and directionality (+1) are retained
in the config. All source mutant sequences are verified against the pinned construct,
not a WT inferred from variants. Missing labels: zero in these selected source tables.
Native processed `DMS_score` is preserved, higher-is-better. M12 does **not** export a
calibrated shared ΔΔG prediction or silently reverse sign conventions.

ProteinGym's [MIT license](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/LICENSE)
covers the curation; original experimental authors and dataset citations remain required.
RCSB/wwPDB coordinate data are CC0; cite each deposition. Family annotations are
third-party Pfam annotations distributed by RCSB with source and assignment version,
not new biological classifications claimed by this project.

| Pfam family | Measured constructs | PDB model | Selected single/double variants |
|---|---|---|---:|
| PF00018 SH3 | BBC1_YEAST / MYO3_YEAST | [1TG0](https://www.rcsb.org/structure/1TG0) / [2BTT](https://www.rcsb.org/structure/2BTT) | 54/42; 24/72 |
| PF00385 Chromo | CBX4_HUMAN / SR43C_ARATH | [2K28](https://www.rcsb.org/structure/2K28) / [2N88](https://www.rcsb.org/structure/2N88) | 46/50; 52/44 |
| PF00046 Homeodomain | NKX31_HUMAN / PITX2_HUMAN | [2L9R](https://www.rcsb.org/structure/2L9R) / [2L7M](https://www.rcsb.org/structure/2L7M) | 55/41; 53/43 |
| PF00397 WW | PIN1_HUMAN / SAV1_MOUSE | [1I6C](https://www.rcsb.org/structure/1I6C) / [2YSB](https://www.rcsb.org/structure/2YSB) | 87/9; 67/29 |

Family assignments use **Pfam 34.0 positional domain features on polymer entity 1**,
retrieved from official `https://data.rcsb.org/rest/v1/core/polymer_entity/<PDB>/1`.
The exact JSON responses are checksum-pinned in the config. `validation.json` retains
family/domain spans, assay-reference overlap and SIFTS UniProt crosswalks, so the family
can be reproduced without trusting a hand-written protein name. Different additional
full-protein annotations in those JSON records are not misassigned to the isolated
experimental construct. All assay domains overlap their annotated family by >50%.

Global aligned-column identities within annotated families: BBC1/MYO3 **0.3158**,
CBX4/SR43C **0.5854**, NKX31/PITX2 **0.3846**, PIN1/SAV1 **0.6000**. This demonstrates
why threshold-only clustering cannot replace external remote-family classification.
Global identity, both coverage values, ambiguous optima, local alignment blocks and
three coverage-filtered clustering thresholds are archived in `audit.json`.
`validation.json` also lists each held-out protein's closest training construct.
No unresolved cross-family cluster or partial-domain hit passed the preregistered
screening rules; remote homology below them remains possible.

Seven NMR structures and the BBC1 X-ray structure use model index 0 and explicit author chains. PITX2 is chain P. SAV1
has repeated terminal cloning tags; exact polymer range 7–49 is independently checked
against the full assay reference. All eight graphs have complete compatible reference
Cα coverage and one connected component; no selected variants are structurally excluded.
Unknown NMR confidence is explicitly masked; no NMR B column is labeled pLDDT or
predictive uncertainty. Only reference-construct geometry is represented.

Frozen ESM-2: `facebook/esm2_t12_35M_UR50D`, revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`, 480 residue dimensions. The label-free cached
arrays and manifests bind ordered variant IDs, sequences, reference/source checksums,
encoder revision, device and extraction versions. Total recorded token extraction was
**7.003 s**, using MPS locally; model initialization, download and graph extraction are
separate preparation costs and are not included in that figure. Every supervised model
starts fresh in its fold. Unknown ESM pretraining family exposure prevents a claim that
these protein families were unseen by the language model.

Domain-boundary audit: selected variants with one or more sites outside the annotated
Pfam span (retained as prespecified measured assay-construct variants): BBC1 10/96, CBX4 1/96, MYO3 11/96, NKX31 9/96, PIN1 18/96, PITX2 0/96, SAV1 23/96, SR43C 0/96.
These can include native flanks and construct extensions; absence from a Pfam span is
not proof of a cloning tag. M12 includes the full measured construct and does not
retrospectively exclude sites after seeing results. This limits a strict domain-core
interpretation; all models still use identical genotypes.
