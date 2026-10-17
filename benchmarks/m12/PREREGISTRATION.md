# M12 fixed experiment plan

Registered before model fitting or held-out metric inspection. Eight ProteinGym v1.3
Tsuboyama 2023 experimentally measured domains; four externally annotated Pfam 34.0
families, two proteins each: SH3, chromodomain, homeodomain, WW. Sources and exact
construct sequences/checksums are in configs/m12.json. Selection was based on domain
annotations, measured assay availability, short constructs and experimental structures,
not prediction performance. This tests stability mutation-effect transfer only.

Label-blind SHA256 ordering selects 96 measured genotypes per protein; 64/16/16
within-protein partitions. Retain single/multiple variants as present; no label-based
stratification. Missing/nonfinite labels are excluded/rejected explicitly. No new labels
or fabricated stability/function pairs. Curated DMS_score is kept assay-local and
higher-is-better; prediction units are training-assay rank, not physical delta-delta-G.

G1: original fixed genotype roles within all eight proteins. G2: eight protein folds;
test one protein, validate the same-index protein from the next family, train the other
six (including a homolog of both validation and test proteins). G3: four family folds;
test both members of one Pfam family, validate both members of the next, train the
other two families. G2/G3 use all 96 selected genotypes of each assigned protein.
No protein crosses roles in G2/G3 and no family crosses roles in G3.

Full assay constructs have global/local sequence alignment audits, including aligned
length, identity, coverage, ambiguous optima and local blocks. The constructs isolate
the annotated domains; other full-length UniProt domains do not define the assay
family. Flag unresolved cross-family local/domain matches (>=25 aligned residues,
>=50% identity, >=60% shorter coverage) or global clusters (40/60/80% identity with
>=70% coverage on both sequences); stop rather than silently accept them. G2 is
within-family supervised transfer; G3 is domain-family holdout, not unseen fold.

Fit transferable sequence-composition/mutation-property ridge, frozen ESM-2 site probe,
structural ridge, fresh structural GNN, weighted fusion, bidirectional cross-attention.
Ridge alpha grid [0.1, 10, 100], validation-only selection. Neural models use identical
fixed 16-wide, two-layer budget, 12 epochs maximum, patience 4, seeds 0/1/2; select
epoch by validation protein-macro Spearman (equal family weights in these balanced
folds). No historical supervised checkpoint is reused. Frozen ESM revision is fixed;
pretraining exposure to these families is unknown and cannot be excluded.

Fit target transforms only on training labels: per-assay training midranks. Score
per-protein native-label Spearman; equally average proteins within family, then
families. Undefined correlations score zero for selection and are separately visible.
Never pool raw RMSE across assays. Sequence-all and structure-matched cohorts are
reported separately; every model in a cohort receives identical roles and test IDs.
No structural substitutions for missing residues; missingness masks are retained.

PITX2 uses author chain P. SAV1 has a repeated cloning tag causing an ambiguous
unconstrained alignment: the full reference uniquely equals polymer residues 7–49,
which are explicitly declared and checked exactly; no alignment ambiguity is guessed.
Experimental NMR B columns are marked unknown confidence, never pLDDT or predictive
uncertainty. Geometry is reference construct context, not mutant rearrangement.

Primary comparison: family-macro Spearman on matched cohorts, with per-family seed
mean/dispersion and G1→G2→G3 gaps using their stated (different) variant counts.
Four families are the effective replication units. Show family-level dispersion;
no confidence interval or mutation-level significance claim from four families.
Report costs, exclusions, failures and negative results; no test-driven grid changes.
No M13, cross-protein supervised pretraining, fine-tuning or new architecture.
