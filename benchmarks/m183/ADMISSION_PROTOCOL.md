# M18.3 preregistration — outcome-blind masked extraction v1

Starting point: PR #42, d9579d6a51eaf200558d6a932e6072cff4cdab32. Issue #43
explicitly authorizes a new extraction scope; it does not authorize changes to M2.
This contract is committed before new geometry or license qualification results.

## Scope and admission

`masked-reference-v1` retains the complete assay reference and its numbering.
It requires a unique original M2 alignment and 100% identity, canonical residue
identity and observed C-alpha coverage **at every explicitly declared eligible
site**. An empty, repeated, unsorted, noninteger or out-of-range mask is invalid.
The original full-reference audit is retained verbatim, including rejection.
No old M2 accepted flag, thresholds or historical source will change. New masked
outputs must not be passed off as accepted full-reference M2 artifacts.

Preregistered eligible sites: CspA 1–65; CI2 1–63; TEM-1 24–286 of the full
286-residue reference (signal residues 1–23 remain missing); GFP 1–237 excluding
63,79,230–237. Coordinate files and exact reference hashes are those pinned in
M18–M18.2. Eligible sites cannot be expanded using phenotype values.

Use deposited heavy-atom geometry for ASA/RSA; calculate contacts on the eligible
induced graph using the original 8 Å / peptide-bond rules. Retain every excluded
reference row as explicitly missing/incompatible. For GFP quantify excluded-site
contacts and the change in eligible-site ASA when the two mismatched residues are
removed from the deposited model. Removal is a sensitivity calculation, **not**
a reconstruction of WT geometry. Search for closer full-span templates without
outcomes. Local masking cannot demonstrate absence of propagated conformational
background effects. A template-background limitation that prevents defensible
WT interpretation remains a gate; no numerical ASA acceptance threshold will be
invented after inspection.

Independently recalculate node/edge distances, atom identity, connectivity,
missing descriptors and M14 chemical feature validity. Exercise frozen M17 input
shapes and mutation-order invariance on the same 96 genotype-selected candidates
per family already recorded by M18/M18.2. All models must use their exact common
intersection. No label-dependent subsampling, exclusions or mask changes.

## Licensing

Seek exact-item linkage between GEO GSE266299 raw scores and the authors' CC-BY
4.0 Zenodo deposit, or authoritative GEO-specific terms. Record item identity,
research, commercial and redistribution permissions separately. An article or
repository license alone does not license an unlinked raw table. Metadata,
archive directory and byte equality may be inspected; no outcome values.

## Confirmatory gate

Retain M17 H1–H5, four families, >=3/4 directional agreement, original similarity
thresholds and prior outcome-blind homology adjudications. All four provenance,
permission, mapping, physical-background and common-cohort gates must pass before
issuing final folds/checkpoints/metrics manifest. Only then execute the original
frozen model protocol once; hash predictions before opening untouched external
outcomes. Otherwise publish BLOCKED, zero outcome cells accessed, no final
manifest and no confirmatory model fits. Preserve every historical artifact.
