# M19: ridge regularization geometry — negative development finding

The preregistered expanded search did not improve the strongest M17 comparator.
This is reused development evidence, not independent confirmation. No SOD1 source,
score, aggregate distribution, prediction or admission gate was accessed/changed.

## Frozen design and lineage

User-approved protocol committed at7664fd2 before fitting. Parent56be251b938b9cb8e9ef454893931eecaa2538c4
is PR49; this PR is stacked, without merging unrelated branches. Exact role/config/
row hashes are in protocol.json. All preexisting tracked files are hash-checked
against historical_hashes.json. Original M0–M18.5 evidence remains byte-identical.

Data: eight constructs, four Pfam families, 768 selected measured variants from
Tsuboyama2023 cDNA-display proteolysis through ProteinGym1.3. See
../../benchmarks/m12/DATA_PROVENANCE.md and ../../configs/m12.json for primary
publication, source-member hashes, licenses and references. ProteinGym curation MIT,
structure coordinates wwPDB CC0; those do not replace original-study attribution.
Training target is train-assay midrank percentile, not common thermodynamic units.
Frozen ESM2 t12/35M revision6fbf070e65b0b7291e7bbcd451118c216cff79d8 supplies961
mutation-site features. No encoder updates, new pooling or architecture changes.

Exactly80 H1 cases: four families x four fractions x five seeds. Full fraction has
256 training variants from four proteins/two families; validation32 variants from
another family; test32 original test variants from the remaining two proteins.
Low-label counts28/64/128; original test/validation identities unchanged. Seeds at
full fraction duplicate deterministic ridge fits and are not independent evidence.

Historical scaled/unscaled grids [.1,10,100]. Expanded scaled/unscaled geometries
each receive nine penalties10^-6..10^2. Training-only centered feature energy E
normalizes X/sqrt(E); equivalent original-coordinate alpha=lambda*E. Coordinate
standardization still changes penalty geometry. Validation Spearman selects all
settings; ascending order and strict greater-than break penalty ties, unscaled wins
geometry ties. Test scores never choose a model. Primary is joint validation-selected
expanded minus historical unscaled at full fraction. Threshold+.02 and>=3/4 positive
families, plus passing numerical controls. Exact definitions in PROTOCOL.md.

## Results

Equal-family Spearman, averaging seeds within family first:

| Fraction | Historical scaled | Historical unscaled | Expanded scaled | Expanded unscaled | Validation-selected expanded | Shuffled training |
|---|---:|---:|---:|---:|---:|---:|
|10%|.103824|.168015|.116912|.204559|.170515|-.069191|
|25%|.168971|.238971|.120882|.274191|.276985|.046985|
|50%|.177941|.258824|.150147|.283603|.211544|-.005662|
|100%|.204412|.344853|.140441|.290441|.147059|-.025368|

Primary family differences:

| Held-out family | Selected expanded rho | Difference vs historical unscaled |
|---|---:|---:|
|SH3/PF00018|-.105882|-.355882|
|Homeodomain/PF00046|.291176|-.041176|
|Chromo/PF00385|.051471|-.248529|
|WW/PF00397|.351471|-.145588|

Primary difference=-.197794,0/4 positive families: **FAIL for improvement**.
The relative change in positive baseline rho is approximately-57.36%; correlation
ratios are descriptive, not a57% change in biological predictive accuracy.
Expanded unscaled alone loses.054412 at full labels. The25% improvement is secondary,
not a replacement primary endpoint. All selections, penalties, validation candidates,
per-protein correlations and rank MAE appear in runs.json; predictions.csv retains
all15360 rows, including shared selected-arm and negative-control predictions.
family-effects.json records every fraction/arm/family contrast.

## Interpretation and failure analysis

A wider validation search is not guaranteed to improve transfer. One validation
family and32 outcomes can select a penalty/geometry that transfers poorly to the
remaining family. Full-label geometry selection amplifies that failure. This does
not prove feature-energy normalization is harmful: normalization with its matching
alpha is a reparameterization, and the grid/search selection is the intervention.
It also does not prove that regularization explains the entire historical scaling
effect. No test-informed replacement grid or architecture was tried.

The primary loss is negative in every family; leave-one-family-out means and family
SD in summary.json quantify descriptive sensitivity. Four development families do
not support a reliable population bootstrap CI or generalization significance claim.
Rank MAE is assay-local evaluation; there is no pooled native RMSE, calibrated
predictive uncertainty, noise ceiling or unseen-study/unseen-family confirmation.

The scalar control has failed cases at the original implementation1e-7 tolerance.
Even float64-rescaled inputs can change iterative LSQR stopping after tiny rounding
perturbations, especially on weakly regularized/ill-conditioned systems. This is a
numerical limitation, not permission to relax tolerance. All errors are retained in
summary.json; overall success is false irrespective of the negative predictive result.
A direct-solver investigation would require a separately registered study.

## Reproduction and evidence

Implementation at6648aa9; protocol at7664fd2. A full repeat after a report-only
NumPy-count serialization correction is the archived execution; the preceding
complete-fit attempt is retained locally. EXECUTION_NOTES.md records earlier stops.
No favorable run was selected. Binary states stay ignored under runs/m19; their
identities are recorded in runs.json. Summary records actual wall time, package
versions, platform/device, replay and numerical errors, and input cache provenance.
No peak memory or cold-source acquisition timing was measured.

From a clean checkout with the original verified M12 source/graph/residue caches:

```sh
python -m pip install -r requirements-m3.txt -e '.[dev,esm,m3,structure]'
# Prepare the original pinned M12 caches using docs/M12.md if absent.
# runs/m19 must not exist. The runner refuses to overwrite it.
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/run_m19.py
pytest -q
ruff check .
ruff format --check .
python scripts/verify_readability.py --historical
```

Fresh inference/encoder extraction is not independently reproduced here; cached
representations are verified by original manifests. Exact saved-state replay and
320 deterministic arm refits are checked during execution. All400 state files
(including80 shuffled controls) are replayed; controls are not independently refit
inside the runner. The complete-fit attempts are separately compared tensorwise.
The selected arm shares one of the two expanded states, rather than a new fit.
Offline test_m19_evidence.py reconstructs every protein correlation and primary
criterion from committed predictions, without model/data downloads.

Claim-to-evidence: primary question/acceptance->PROTOCOL.md; split identities->
protocol.json and original M17 split manifest; source/model lineage->M12 config and
summary evidence; all selection decisions->runs.json; predictions->predictions.csv;
paired uncertainty->family-effects.json/summary.json; numerical controls->summary;
software/history checks->verification.json. No new improvement application bullet
is justified. The transferable ML evidence is disciplined falsification, model
selection failure analysis and numerical/provenance auditing; protein prediction
is not neural decoding.

## Qualification

| Gate | Status |
|---|---|
|Protocol before fitting, fixed roles/search, train-only handling|PASS|
|Historical numerical reproduction and immutable artifacts|See verification.json|
|Saved-state replay, deterministic arm refits, test-label isolation|See summary.json|
|Scalar-rescaling numerical gate|FAIL; preserved|
|Primary scientific improvement|FAIL; negative in4/4 families|
|Independent validation/external admission|BLOCKED; unchanged|
|Cold-source reproduction / memory profiling|NOT RUN|
|Release/tag|NOT AUTHORIZED|
