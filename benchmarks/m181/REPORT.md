# M18.1 blinded homology adjudication: external confirmation remains BLOCKED

Starting point: PR38 `3eb51fbb25325d6a9a20bccf48211cc834aa5113`, verified12/12 CI.
Prospective eligibility amendment: `ce0e581`, committed before new homology evidence.
M17 H1–H5, >=3/4 directional denominator, thresholds, models and all M0–M18 evidence
are unchanged. No candidate test-label cells for CspA, CI2 or TEM1 were inspected.
No fits, predictions, external performance, paired outcome differences or calibration
claims are reported. This package resolves the interpretation of the four flags with
bounded evidence; it does not silently turn the old M18 BLOCKED record into a PASS.

## The four exact flags

Original scoring: match2/mismatch−1/gap-open−3/gap-extend−0.5. All four global optima
are ambiguous and all four original review flags remain true. Both-sided coverage
and original block coordinates are preserved in `alignment-diagnostics.json`.

| Pair | Identical/aligned | Original identity | Both-coverage minimum | Identity/full shorter | Local aligned |
|---|---:|---:|---:|---:|---:|
| CspA–BBC1 |20/48|.416667|.738462|.312500|3|
| CI2–BBC1 |21/49|.428571|.765625|.333333|5|
| CI2–1LP1 |19/46|.413043|.730159|.345455|4|
| CI2–1A32 |21/47|.446809|.746032|.333333|5|

These global alignments leave17/16,14/15,17/9 and16/16 residues unpaired respectively.
Identity conditional on aligned residue pairs exceeds40%, while identity over a full
sequence does not. This explains sensitivity to short gapped alignments; the full-length
identity is a **diagnostic**, not a replacement admission threshold. A local3–5-residue
match is not evidence of a conserved domain. Ambiguous optima are retained, not hidden
by selecting a favorable alignment.

## Reciprocal profiles and structural evidence

Ran25 searches (five exact assay constructs × five official Pfam HMMs) locally with
PyHMMER0.12.1, one CPU, prefilters/bias filter disabled, permissive reporting and the
**published sequence AND domain gathering thresholds** for decisions. Versioned HMM
accessions, raw text profiles, GA cutoffs, domain spans, scores and checksums are stored.
No missing hit is reported as a measured zero. The own-family positive controls all pass:

| Construct | Profile | Sequence bits | GA sequence/domain |
|---|---|---:|---:|
| CspA |PF00313.29 CSD|113.2883|27/27|
| CI2 |PF00280.25 potato_inhibit|93.6350|24.5/24.5|
| BBC1 |PF00018.35 SH3_1|37.9637|22.9/22.9|
| 1LP1 affibody |PF02216.23 SpA_B|51.8523|27/27|
| 1A32 S15 |PF00312.28 Ribosomal_S15|107.7993|31/31|

All20 cross-family searches fail GA. This is reciprocal evaluation against a **five-family
panel**, not exhaustive HHsearch, a full-proteome search or proof that remote homology is
impossible. Tiny-panel E-values are not interpreted as full-database E-values. Distinct
clans support the same interpretation: CSD/OB CL0021, SH3 CL0010, CI2 CL0367; other
family/clan metadata are retained rather than invented when absent.

Primary PDBe/SIFTS CATH/SCOP and InterPro mappings establish:

| Exact construct | Structure/reference mapping | CATH / SCOP support |
|---|---|---|
| CspA |1MJC entity1 A, entity residues4–68, exact65aa|CATH2.40.50.140, OB fold; SCOP b.40.4.5 |
| CI2 |3CI2 entity1 A, residues4–66, exact63aa|CATH3.30.10.10, α/β inhibitor sandwich; SCOP d.40.1.1 |
| BBC1 |1TG0 entity1 A, residues5–68, exact64aa|CATH2.30.30.40, SH3 topology; InterPro IPR036028 |
| 1LP1 |entity1 A, residues4–58, exact55aa|No CATH/SCOP mapping returned; own SpA profile plus deposited helices4–18,23–37,40–55 |
| 1A32 |entity1 A, residues23–85, exact63aa|CATH1.10.287.10, mainly α; SCOP a.16.1.2 |

All mapped CATH spans cover the corresponding complete assay constructs. The 1LP1
historical construct is the **in-vitro-selected affibody** in entity1, not the protein-Z
partner entity2. The deposited sequence is checked, not assigned from the PDB title.
Full protein A P38507 has multiple domain types; that full architecture is not attributed
to the55aa construct. 1LP1's missing curated fold assignment is explicit. Its exact-sequence
SpA profile hit and deposited three-helix arrangement support a different architecture
from CI2; they do not manufacture a CATH superfamily assignment.

**Pair adjudication:** CspA–BBC1 and CI2–BBC1 have distinct mapped topologies and own-only
profile support; CI2–S15 additionally differ in structural class. CI2–affibody has own-only
profile support and α-helical deposited architecture, with missing CATH/SCOP evidence.
For each pair, the combined evidence favors an incidental gapped alignment rather than
meaningful family relatedness. None is classified as proven homologous, and none establishes
zero remote ancestry. The original numeric review flags remain true. This outcome-blind
eligibility amendment supplies an interpretation, not a threshold relaxation or a retroactive
change to M18. Further profile–profile/phylogenetic work could alter the bounded interpretation.

## Additional independent primary family: TEM1

[Firnberg et al.2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4032126/),
DOI10.1093/molbev/msu081, describes new TEM1 codon-library experiments and explicitly
identifies **Data S2** as protein missense fitness, combining synonymous codon measurements.
The primary PMC-linked workbook is `supp_msu081_Data_S1-S4.xlsx`, SHA256
`bcdd031301113ede74652d4e2eba3f03cf48eb46222d92ccab5c8787950aa5d3`.
The Europe PMC supplement bundle URL/checksum are in the committed manifest.

Column-level chain: primary experiment → author's S2 sheet `S2 Missense mutation fitnesses`
→ column **R, Fitness**; **S, Estimated error in fitness** is separate. A/B/C are Ambler
position/author WT AA/mutant AA. No ProteinGym-transformed column or predicted thermodynamic
score is substituted. S1 U is codon fitness, S3 averages are substitution summaries, and S4
is effective amino-acid diversity; those are not interchangeable labels.

The measured endpoint is relative ampicillin-resistance capability of TEM1 in E.coli via
bandpass selection across sublibraries and sequencing. It is neither growth-rate fitness
under an arbitrary single concentration nor folding ΔΔG. Higher reported fitness is better;
the table uses a WT-relative scale. No log transform, pooled cross-assay RMSE or kcal/mol
interpretation is imposed. Censoring/missing target cells have deliberately not been surveyed:
future common exclusions must be frozen before label access, with no favorable replacements.

Genotype-only A/B/C projection gives5,740 rows,287 author-WT sites including termination,
and5,434 canonical non-WT substitutions. Concatenating the **author WT column**, not a
variant consensus, exactly matches reviewed UniProt P62593's286aa sequence after removing
the terminal stop. Ambler numbering skips239 and253; positional labels must not be used
as contiguous sequence offsets. All mutant symbols are canonical and WT-site assignments
are consistent. Against all18 historical reference sequences, TEM1 has zero unchanged
review flags and no exact alias; no TEM1/Firnberg target lineage appears in historical
benchmark/configuration records. This supports a new measured family PF13354, not encoder
pretraining independence.

Structure caution: deposited1BTL has263 mature residues, misses canonical1–23, and differs
at canonical82 (assay V, structure I) and182 (assay A, structure V). Do **not** silently
relabel it as a perfect WT structure or whitelist those mismatches. No graph has been
admitted. Another verified matching structure or a prospectively justified shared-site
cohort is required. The primary article license is **CC-BY-NC3.0**, not MIT/CC0; no workbook
or labels are redistributed. Commercial/distribution permissions remain a separate gate.

## Admission matrix and stop decision

| Candidate/gate | Evidence verdict | Admission |
|---|---|---|
| CspA raw abundancePCA |M18 primary lineage; distinct-profile/fold evidence|Original flags retained; bounded adjudication supplied |
| CI2 raw abundancePCA |M18 primary lineage; distinct-profile/fold evidence|Original flags retained; bounded adjudication supplied |
| TEM1 S2 R |PASS new primary label lineage and reference/genotype audit|BLOCKED multimodal mapping/permission review |
| FYN SH3 |Known historical family|REJECT novel-family claim |
| Ubiquitin Mavor2016 |Methods metadata found; historical M2 ubiquitin reference already present|Not qualified; also inline example-output exposure recorded |
| Exact FLIP2 Hydro transform |No new source-to-target transform established|BLOCKED as before |
| Original M17 confirmation |At most three potential distinct test families, fewer fully admitted|BLOCKED: original >=3/4 denominator cannot be applied |
| Final folds/predictions |Prerequisite admission failed|NOT ISSUED / NOT RUN |

Three candidates cannot satisfy the fixed four-family confirmation design, even under
the most favorable interpretation of every short-alignment flag. This is the decisive
remaining cohort-size blocker, compounded by TEM1 structure mismatches and permissions.
We did not redefine the denominator, run a three-family exploratory outcome study or tune
on labels. H1–H5 external mechanistic verdicts, paired family effects, low-label curves and
calibration remain **not evaluated**. All original development negative findings stand.

The [exposure ledger](exposure-ledger.json) records a failed alternative-source search:
Mavor2016 methods embedded illustrative barcode scores and an example matrix, which appeared
during methods inspection. They were not used for selection/training; that source is excluded
from a fully blinded confirmation claim. The actual CspA/CI2/TEM1 candidate target columns
remain untouched. Claiming that no numeric outcome text appeared anywhere would be false.

## Reproducibility, licensing and release gates

Pfam HMMs are [CC0 downloadable data](https://interpro-documentation.readthedocs.io/en/latest/license.html);
attribute Pfam/InterPro. PDBe/SIFTS mappings and UniProt evidence retain source links and
checksums; source-code MIT does not replace their terms. Sources acquired2026-10-18,
profile accession revisions/date fields retained. The initial profile discovery included
three irrelevant accessions (GntR/GATA/profilin); fetch logs retain that failed lookup,
and **none** was used as the S15/SpA family or in the25-search experiment.

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
python scripts/audit_m181.py
```

The evidence verifier is stdlib-only and uses no phenotype source, model cache or network.
To rerun the profile audit, install `pyhmmer==0.12.1` in a disposable environment, then run
`python scripts/profile_m181.py`; this uses the committed exact HMMs and reference sequences.
Runtime/device/platform are recorded; no model/device comparison is performed.
For the primary genotype-only replay, download the Europe PMC supplement ZIP at the URL
in `evidence/TEM1-supplement-manifest.json`, verify its SHA, extract only the named workbook
to `data/cache/m181/source/TEM1-Data-S1-S4.xlsx`, verify the XLSX hash above, and run
`python scripts/qualify_m181_tem.py`. No workbook, target cells, pickles or weights are committed.
Metadata acquisition is documented by URL/hash manifests, not advertised as independently
reproduced by a second operator. Raw targets remain quarantined. M0–M18 training and MPS
runs are not repeated; historical source/artifact hashes and role/metric audits are checked.

| Release gate | Status |
|---|---|
| Verified PR38 baseline and prospective amendment |PASS |
| Four-pair bounded homology investigation |PASS, limits above |
| Additional primary column lineage/genotype evidence |PASS TEM1 |
| Four admitted independent external families |BLOCKED |
| Exact TEM1 structure and commercial redistribution |PARTIAL/BLOCKED |
| Historical artifact/source invariance |PASS subject to verifier |
| Offline checks/Ruff/CI |See verification.json and exact-head PR checks |
| Independent cold-source evaluation, calibration, external claims |NOT RUN |
| Release tag |NOT QUALIFIED; none created |
