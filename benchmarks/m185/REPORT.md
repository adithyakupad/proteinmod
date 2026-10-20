# SOD1 blinded-candidate admission audit — issue48

**Four-family admission remains BLOCKED. No external ML confirmation, supervised
SOD1 fit, prediction, final manifest, merge or release tag was produced.**
Individual variant scores remain unread. Strict metadata blinding was **not fully
preserved**: the initial API metadata response included an aggregate distribution
that was mechanically decoded before schema inspection. The deviation is recorded
below and in its earlier committed record; no pristine independent-validation
claim is made.

## Frozen starting point and protocol

Base: `research/sod1-blinded-admission` at
`c748ef6924a102ef424dac3212b8b61fad993697`, clean working tree. Dedicated branch:
`codex/sod1-blinded-qualification`. Before acquisition, commit `e6e4b22` chose
abundance urn:mavedb:00001217-a-3 as the primary outcome, the initiator-only mask
policy and original deterministic genotype selection. Activity is context only,
not a score-selected fallback. No activity genotype/outcome rows were acquired; activity column headers are context only.

All **878 parent files** are protected byte-for-byte, including readability
changes, M0–M18.4 evidence, source APIs, checkpoints/configurations, old failures,
CI workflow and MIT source license. New tests run through existing CI; no old
source guard or workflow hash was edited. M17 H1–H5, >=3/4 directional agreement,
four-family denominator, original folds, seeds, budgets, physical/null controls,
model selection and uncertainty rules remain unchanged. SOD1 is a proposed GFP
replacement, not an admitted fifth observation. Personal prior SOD1 PyMOL work is
biological familiarity, not supervised benchmark exposure. Frozen encoder and
ProteinMPNN pretraining exposure are unknown; independence here concerns new
experimental measurements and supervised families, not pretraining independence.

## Source and genotype acquisition

The primary [MaveDB abundance item](https://mavedb.org/score-sets/urn:mavedb:00001217-a-3)
provides the exact 154-aa SOD1 target, 2,906 catalogued variants, score-column
schema and an explicit **CC0** license. This improves the earlier branch's
usage-policy-only evidence. A separate sequence-only UniProt P00441 acquisition
matches the assay reference exactly. Metadata modification/publication date is
2025-06-16. The paper is Axakova et al., *Landscapes of missense variant impact
for human superoxide dismutase 1*, DOI10.1016/j.ajhg.2025.08.019,
PMC12696502.1. BEBlab's distinct 2026 ddPCA publication was not used.

Official server source at `894b63294bdb29e0ac65ac7e99854df658ab2138` and OpenAPI
were inspected before the successful genotype request. Explicit CSV namespaces
replace defaults; CORE contains accession/HGVS only, and `mavedb` adds reference
HGVS/VRS identifiers. Score, count, clinical and calibration namespaces were
excluded. The response header was checked against a finite allowlist **before
reading body cells**. All 2,906 unique variant IDs were acquired, and all 2,906
assay HGVS strings match the corresponding RefSeq-frame protein HGVS exactly.
No individual score, SE, LLR, evidence-weight or count values were read.

Automatic approval review initially rejected a generic full-body CSV download
because namespace exclusion was not yet verified. That command never executed.
The replacement request was approved after official-source verification and
passing tests of a header-first abort boundary; it was not a workaround to fetch
forbidden columns. The API's `/variants` lean view is explicitly avoided because
its schema includes `score`.

| Genotype-only result | Actual count |
|---|---:|
| Provider rows / unique variant URNs |2,906 |
| Canonical, unique, structurally eligible missense sequences |2,632 |
| Synonymous / WT exclusions |135 |
| Stop exclusions |139 |
| Mapping/reference errors |0 |
| Preregistered candidate selection |96 |

Selection uses original SHA256(`m18:external:SOD1` + mutant_sequence) ordering,
without scores. All eligible variants are single substitutions; multiple-mutation
model inputs were tested synthetically, not claimed as measured combinatorial
SOD1 data. The 96 rows are **candidate inputs**, not admitted independent observations.
Provider missing variants were not imputed and no label-based exclusions/replacements
were made. Any eventual outcome missingness requires the original frozen common
exclusion policy, not replacement by favorable candidates.

## Actual strict 2C9V mapping and physical context

The official deposited mmCIF reports X-ray diffraction at **1.07 Å**. Both protein
chains **A and F** have 153 residues matching assay positions2–154 exactly.
For each chain the unchanged strict full-reference mapping gives:

- Accepted: **false**, retained verbatim.
- Coverage: **153/154 =0.9935064935064936**.
- Aligned identity: **1.0**; compatible CA positions: **153**.
- Missing reference position: **1 (initiator Met)**; sequence mismatches: **0**.
- Unique assay-to-polymer alignment; no guessed offset or synthetic substitution.

The predeclared conditional mask2–154 passes the existing masked-reference-v1
extractor. Lexicographically first eligible chainA was selected without outcomes.
This is a prospective masked scope, not retroactive full-M2 acceptance.

| Masked chain-A result | Actual result |
|---|---:|
| Eligible identity / CA coverage |100% /100% |
| Nodes / undirected edges / directed chemical-edge rows |153 /827 /1,654 |
| Connected components / isolated nodes |1 /0 |
| Static / mutation / ridge / edge widths |34 /25 /270 /29 |
| Missing RSA / density / B-factor descriptors |0 |
| Missing pLDDT |153 (correct for experimental geometry) |
| Independently recalculated 8 Å contact set |Exact equality |
| Synthetic multi-mutation order invariance |Exact equality |

Each monomer has a deposited Cu site with alternative positions of occupancy0.90
and0.10 (not two fully occupied Cu ions) and a Zn site of occupancy1.00. The
chain-A Cys57–Cys146 disulfide corresponds to **assay Cys58–Cys147**, SG distance
**2.1765768528 Å**. Experimental B-factors are displacement/refinement quantities,
not pLDDT or predictive uncertainty. No new metal nodes or interaction architecture
were introduced. Existing feature/edge definitions are unchanged.

Reference geometry describes the deposited metallated, disulfide-bonded SOD1
dimer context, including other chains/ligands in ASA. The abundance assay uses a
C-terminal SOD1–GFP fusion in HEK293T cells; yeast activity uses a different
functional-complementation setting. The structure does not establish identical
assay metal occupancy, initiator processing, fusion effects or mutant rearrangements.
The missing Met remains explicit rather than asserting that all assay constructs
undergo the same maturation. These limitations must accompany future predictions.

## Family / homology audit

The unchanged M12 global/partial similarity review finds **0 flags across22
comparisons**: all18 historical reference-panel proteins plus CspA, CI2, TEM1
and GFP. A metadata-only scan of maintained JSON/TOML reference definitions found
no historical sequence absent from that panel. Every parent configuration is
hash-protected. Raw alignment diagnostics retain **22 ambiguous global optima**
and6 ambiguous local optima; none was deleted or reinterpreted as a mapping.
The longest local alignment has14 residues, below the unchanged25-residue
partial-review minimum. Fragmented global alignments can have high identity on
small aligned subsets; raw identity alone is not evidence of close homology.

Using the same filter-disabled HMMER protocol and published GA cutoffs as M18.2,
**23 references ×12 explicit profiles =276 searches** were executed locally with
pyhmmer0.12.3, CPU1. PF00080.26 (Sod_Cu; InterPro IPR001424) matches SOD1:
sequence score **177.332214 bits**, domain score177.156235, published GA28.9 bits,
E-value **2.1863e-55**, domain reference positions15–150. No historical/external
reference passes the SOD_Cu profile, and SOD1 passes no other included profile.
All276 records reproduce exactly on repeat execution. This is a bounded reciprocal
family screen with version/hash-pinned profiles, not an exhaustive all-Pfam search
or proof of unseen-fold/encoder-pretraining generalization. Previous M18 flags and
homology adjudications remain unchanged.

## Exact column lineage, assay handling and permissions

Paper methods and the primary MaveDB method/schema establish:

- Abundance is VAMP-seq in HEK293T landing-pad cells, with two biological replicates;
  author source comments describe three technical library preparations per biological
  replicate. Activity is yeast functional complementation, not the selected endpoint.
- tileseqMut/tileseqMave processing; provider minimum10 nonselected-condition reads;
  nonsense/synonymous distribution anchors define0/1. They are scale definitions,
  not native ΔΔG, and do not justify clipping values into[0,1].
- `score` is the map phenotype, `se` its standard error, `df` degrees of freedom.
  Metadata describe replicate averaging/error propagation and single-replicate
  fallback. Exact reconstruction of those computations was not claimed or run.
- `llr`, `llrCI_lower`, `llrCI_upper`, `evidence_weight` are clinical-derived fields;
  `pos` is a derived position. None is a functional training target or input here.
  Author code links merged unfloored maps and an MaveDB export that separates CI
  columns/removes SD/derives positions; code was never executed.

The paper explicitly links final map scores to Data S2 and MaveDB. Its abundance
URN contains a duplicated `mavedb:` prefix; the valid API item, exact gene target
and assay title are recorded rather than silently treating that typography as a
verified file hash. Data S2's caption identifies `mmc2.xlsx`. Initial header-only ZIP attempts
received HTTP403. An honest research User-Agent made public metadata accessible;
an inherited Excel namespace then caused a first-row parse error, corrected with
a tested namespace-preserving header wrapper. The final bounded read succeeds:
**12 sheet headers, 68,846 transferred range bytes, zero data rows parsed**.
The `abundance_map_scores_LLRps` sheet A–I header is exactly:
`hgvs_pro, score, se, df, llr, llrCI_lower, llrCI_upper, evidence_weight, pos`.
This equals the MaveDB schema's genotype field plus all score-column names in order.
The activity sheet similarly has `hgvs_pro, score, se, df, pos`, but no activity
values or genotypes were acquired. Clinical/reference/other sheets were inspected
only for header metadata, never data rows. The reader rejects numeric first-row
cells and decodes only selected shared-string headers. Whole-sheet CRC and full
workbook SHA were not verified; metadata/shared-string CRC checks are explicit.

**Named-column lineage is PASS** using primary paper/deposit linkage, matching
headers, method definitions and relevant author export source. This is not an
empirical comparison of score values, Data S2 row IDs or numerical equality.
All initial access/parser failures remain archived rather than being erased.

Dataset permissions are stronger than the earlier candidate note: the **specific
MaveDB item is explicitly CC0**, supporting research, redistribution, derived
predictions/weights and commercial reuse insofar as its data copyright is concerned.
That conclusion uses the item license, not generic accessibility or ProteinMod MIT.
The article separately declares CC-BY4.0. Its declaration does not automatically
license all linked GitHub software; only bounded source-lineage excerpts are kept.
Coordinates and downloadable Pfam data have their own CC0 notices. No SOD1-specific
permission email is required by the observed CC0 grant; **CspA/CI2 GEO item rights
remain unresolved** under the unchanged historical gate.

Authority links: [MaveDB licensing](https://www.mavedb.org/docs/mavedb/submitting-data/metadata-guide.html#licenses),
[version retention](https://www.mavedb.org/docs/mavedb/submitting-data/publishing.html),
[wwPDB policy](https://www.wwpdb.org/about/usage-policies),
[InterPro/Pfam terms](https://interpro-documentation.readthedocs.io/en/latest/license.html).

If eventually admitted, use original test-only M17 rotations, assay-local
Spearman and equal-family macro aggregation. Do not pool native assay RMSE or
claim comparable thermodynamic units/conformal guarantees across assays. Family,
assay and mutation-burden shifts are confounded, not isolated causal factors.

## Blinding deviation and qualification matrix

During initial metadata schema inspection, two whole-JSON loads mechanically
decoded an unexpected nonempty `scoreDistribution`. Its values were not printed,
queried, used for selection, committed or supplied to a model. Individual variant
scores were never accessed. **This is still a failure of pre-decode aggregate
exclusion**, not pristine outcome-blind qualification. Commit `d4a0ef2` recorded
the deviation and introduced tested whitelist selection before JSON decoding.
The original quarantined metadata hash is retained without publishing the field.
Independent review must decide whether this aggregate-only incident affects a
future confirmatory claim; this work cannot certify it away.

| Gate | Status | Evidence / limitation |
|---|---|---|
| Historical / readability preservation |PASS |878 parent files exact; old guards pass |
| Genotype-only export |PASS |2,906 rows, validated header, exact RefSeq HGVS agreement |
| Strict full-reference 2C9V mapping |FAIL retained |Only missing initiator Met; no threshold changes |
| Prospective mature-reference graph |PASS |153 nodes,827 edges,96 matched candidate inputs |
| Bounded family/similarity checks |PASS |Sod_Cu GA match;0 original flags; all ambiguities retained |
| Specific MaveDB dataset license |PASS |Explicit CC0 item statement |
| Data S2 named-column linkage |PASS |Primary sheet header matches exact MaveDB columns; no score values compared |
| Initial pre-decode aggregate blinding |FAIL / DEVIATION |Distribution mechanically decoded; not displayed/used |
| Subsequent exclusion guards |PASS |Poison-field/header/body tests; no individual scores read |
| Individual variant label blinding |PASS |No score endpoint, score column or clinical values acquired |
| Four-family final admission |BLOCKED |CspA/CI2 permissions and deviation review |
| External ML confirmation / release |NOT RUN |No final read/fold/checkpoint manifest; no tag |

## Reproduction and software checks

See `verification.json` for actual tests, Ruff, build and clean-source results.
The full local suite passes **242 tests** (one existing joblib core-count warning);
A separate clean checkout at f6ec52e, without copied research caches, also passes
241 tests and the stdlib integrity/history checks using the same dependencies.
Ruff, formatting, dependency consistency and wheel build pass. Existing CI runs
the new evidence guard via pytest; all old CI checks remain. Initial legacy
history-count failures prompted moving only these new artifacts/configuration
into the existing m185 naming convention; no historical guard was weakened.
No model architecture or project dependency was added. The optional existing
profile-audit tool was installed into an isolated temporary directory. Scientific
geometry/genotype/similarity JSON and all276 profile records replay exactly.

```bash
python scripts/verify_sod1.py
python scripts/verify_m184.py
python scripts/verify_readability.py --historical
ruff check .
ruff format --check .
pytest
python scripts/audit_sod1.py --output /tmp/sod1-new-geometry
PYTHONPATH=/tmp/proteinmod-sod1-deps python scripts/profile_sod1.py --output /tmp/sod1-new-profiles.json
```

For the optional profile tool, install the measured version into an isolated
directory: `python -m pip install --target /tmp/proteinmod-sod1-deps pyhmmer==0.12.3`.

For fresh sources, use only the exact primary URLs and pinned hashes in
`source_inventory.json`. Official sequence-only sources are safe; raw API metadata
must be processed through `scripts/sod1_sources.py`, never whole-JSON decoding.
The verified genotype acquisition is
`python scripts/sod1_sources.py --genotypes --output /tmp/sod1-new-source`.
It requires a fresh output directory, validates the header and enforces the preregistered
item/count; compare with the committed projection hash before substituting live data.
`python scripts/sod1_supplement.py --output /tmp/sod1-new-headers.json` reproduces the
header-only acquisition. HTTP/parser failures require fixing the metadata-only
path, never falling back to a full phenotype-table read.

The coordinate SHA256 is
`378722d38fea43812a185c331bb35e2fa2fd456d4e74d93bc7ab945698088145`.
The canonical genotype-projection JSON SHA256 is
`1248c930c9c82888f331d544b9c95e9717ba00793ebc5c471689bf763097aeb9`;
it is a normalized API projection, **not a captured raw-CSV wire digest**.
PF00080.26 SHA256 is
`0da04d63e48c5d52ae1e4b219b234d138a0a4528c4129d09f9c72eef51e02cbb`.
Full source URLs, author/API revisions, quarantine scope and hashes are preserved.
No raw coordinate file, phenotype table or complete outcome-derived metadata is
committed. Tests/integrity checks need no research cache; real geometry replay
needs the pinned mmCIF, and HMM replay needs pyhmmer and the explicit profiles.
No full M0–M15 retraining or new predictive performance claim is made.
