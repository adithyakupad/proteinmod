# M17 frozen protocol, before model testing

Baseline 0908f90727bf36c0c5de7bdf7f591656efb27022. Existing M12 four-family
scores motivated this study and are DEVELOPMENT evidence. No independent
confirmation is claimed from reanalysis or newly sampled Tsuboyama measurements.
All historical artifacts remain immutable. Source license is now MIT, committed
by the owner on the verified M16.1 branch. No merge/release is authorized.

## H1–H5 and falsification

H1 sample complexity: structural ridge's advantage over flexible/frozen-feature
probes decreases as labels increase. Within each of four fixed G3 family roles,
nest 10/25/50/100% of 64 original training variants per training protein (ceil),
five seeds. Fixed original-test 16/protein; separate original-validation 16/protein
in the next family; no test or calibration labels enter tuning. Report paired
per-family rho differences and train/validation gap. Descriptive support requires
mean structural-minus-ESM rho to drop >=0.02 from 10% to 100%, with >=3/4 family
slopes in that direction; otherwise contradicted/unresolved. This is not a p-value.

H2 family shift: test EXACT same original-test genotypes across G1/G2/G3.
For each test protein: next family held out for validation; each protocol trains
four proteins x32 variants. G1 includes the test protein's disjoint training rows;
G2 includes its family partner but no test-protein rows; G3 excludes the family.
G3 train four proteins in the other two families. G1/G2 replace one of those
proteins with the permitted test/partner protein. Same 128 labels and four proteins;
composition and familiarity necessarily change, so this is not isolated causality.
Separately compare narrow vs diverse training with TWO proteins x64 variants:
narrow both proteins of one remaining family; diverse one protein from each of
two remaining families. Enumerate both narrow choices and all four diverse choices;
fixed validation/test, seed-independent full labels. Report family paired contrasts,
no pooled RMSE. Support diversity benefit only if mean gain >=0.02 and >=3/4 families
positive. Otherwise report contradictory/unresolved, not significance.

H3 capacity/inductive bias: at 25/100%, four G3 folds, five seeds, train original
GNN and rich edge GNN, identical-architecture zero-message, degree-preserving
rewired-edge and chemistry-shuffle controls. Same nodes, mutation inputs, labels,
validation and 12-epoch/4-patience CPU budget. Rich graph controls have identical
nominal parameter counts; zero-message weights may be inactive (report effective
capacity limitation). Original GNN has fewer parameters and edge features and is
an architectural reference, not a pure capacity match. Also fit train-scaled
structural nonlinear MLP vs structural ridge with validation-only selection.
Claim topology evidence only if rich edge minus zero-message and rewired mean rho
both >=0.02 with >=3/4 families positive; never infer causality from attention.

H4 graph information: audit exact mapped C-alpha contacts against Euclidean cutoff
plus backbone edges, connected components, atom completeness, construct differences,
missing descriptors, distances from mutation sites to missing reference positions.
Unknown/missing physical atoms are not absent interactions. Degree-preserving null
uses undirected double-edge swaps, rejects self/duplicate edges, retains bidirectionality;
backbone/feature assignment under null is deliberately nonphysical. Report what is
preserved and broken. Chemistry is a proximity candidate, not a validated bond.

H5 representation mismatch: freeze ESM/ProteinMPNN, compare mutation-local and global
pooling, scaled/unscaled ridge, structural and biochemical descriptors, fixed nonlinear
probes, and WT-only/no-site-signal control. Validation selects alpha only; MLP uses
train-only StandardScaler, fixed hidden16, alpha grid [0.1,10,100], max_iter200,
validation rho; labels are train-assay midrank percentiles. No supervised encoder
reuse or finetuning. Report dimensionality, selected alpha, train/validation gap,
constant/undefined rho and paired effects. A >=0.02 paired improvement across >=3/4
families is descriptive support for the specific intervention, not proof of a mechanism.

## Statistical/uncertainty contract

Primary per-protein within-assay Spearman (constant => undefined, not discarded);
macro within each held-out family, paired differences against structural ridge,
then equal-family average. Five seeds quantify fit/subset variability, NOT five
independent families. Four families are insufficient for reliable family-bootstrap
population CI; report range/SD/directions, no confirmatory p-values or stable CIs.
Do not pool native RMSE across incompatible assays. Ensemble spread/error and
selective prediction use within-assay rank errors for descriptive diagnostics only.
Native 80/90% calibration uses only original M13 native-compatible evidence and
its separate calibration roles: clearly marked reanalysis, with unbounded G3
family-block intervals retained. Do not invent kcal/mol for proxy rank assays.

## Independent data gate, before outcomes

Qualify primary experiment identity, license, WT/construct mapping, source overlap,
family assignment, censoring and endpoint. FLIP2 Hydrophobic Core metadata/README
points to Escobedo et al. 2024, Zenodo11175470, first-hand lab measurements CC-BY4;
this is NOT Tsuboyama. Full 1.3GB primary archive will not be downloaded. Obtain
bounded source/sequence provenance, verify reference/structure and exact label-blind
cohort, commit hash-pinned manifest BEFORE accessing test outcomes. Sample up to96
per construct by SHA256(salt+protein+sequence), no target filtering except missing/
nonfinite labels excluded with counts. Training64/validation16/test16 if enough.
All methods share exclusions; frozen-development transfer and external within/cohort
fits are separate outputs. Three constructs include SH3 overlapping development;
exclude that from unseen-family claims. Require external family evidence for others.
Proxy stability is rank-only, not native ΔΔG; no physical calibration extrapolation.
Frozen pretraining overlap is unknown. Exact folds and eligibility decisions must
be committed in a second-stage manifest before any external fits/outcome inspection.
If references, lineage, endpoint or license cannot be qualified, mark BLOCKED and
run no independent confirmatory test. No false success by substituting same-study data.

Nutschel2020 is independent but T50/thermal inactivation is not folding ΔΔG and ACS
redistribution terms need review: defer as a confirmatory native cohort. Nisthal2019
GB1 is primary independent but same previously studied GB1 protein and one family:
not a clean unseen-family confirmation. Tsuboyama-derived repackagings are rejected
for independent measurement generation. ProteinGym is curation, not a new experiment.

All deviations, failures, exclusions, runtimes, versions and artifact hashes are
reported. No test-driven redesign, threshold changes, release tag or silent reruns.
