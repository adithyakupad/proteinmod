# M4 cross-protein pilot

Three assays, 600 label-blind sampled variants each. G1 retains original variant splits; G2 assigns entire proteins to train, validation and test.
All selections use validation macro Spearman. Targets are midrank percentiles fitted only within each source-training assay; held-out proteins have no label calibration.
Only rank correlations are aggregated. Raw assay values, especially ordinal NucB categories, are not commensurate.

## Macro Spearman (equal protein weight)

| Protocol | Cohort | Model | Macro rho | Seed SD | Defined proteins |
|---|---|---|---:|---:|---:|
| G1 | sequence_all | esm_sites | 0.48625 | — | 3/3 |
| G1 | sequence_all | mutation_count_ridge | 0.15645 | — | 3/3 |
| G1 | sequence_all | sequence_ridge | 0.34671 | — | 3/3 |
| G1 | sequence_all | within_one_hot | 0.61402 | — | 3/3 |
| G1 | structure_matched | esm_sites | 0.49318 | — | 3/3 |
| G1 | structure_matched | mutation_count_ridge | 0.16629 | — | 3/3 |
| G1 | structure_matched | sequence_ridge | 0.39385 | — | 3/3 |
| G1 | structure_matched | structural_gnn | 0.47043 | 0.02269 | 3/3 |
| G1 | structure_matched | structural_ridge | 0.46116 | — | 3/3 |
| G1 | structure_matched | within_one_hot | 0.62539 | — | 3/3 |
| G2 | sequence_all | esm_sites | 0.09322 | — | 3/3 |
| G2 | sequence_all | mutation_count_ridge | 0.19329 | — | 3/3 |
| G2 | sequence_all | sequence_ridge | 0.13140 | — | 3/3 |
| G2 | structure_matched | esm_sites | 0.12577 | — | 3/3 |
| G2 | structure_matched | mutation_count_ridge | 0.19653 | — | 3/3 |
| G2 | structure_matched | sequence_ridge | 0.14278 | — | 3/3 |
| G2 | structure_matched | structural_gnn | 0.13004 | 0.00100 | 3/3 |
| G2 | structure_matched | structural_ridge | 0.16690 | — | 3/3 |

## Per-protein results

| Protocol | Cohort | Model | Protein | Test n | Rho mean | Seed SD |
|---|---|---|---|---:|---:|---:|
| G1 | sequence_all | esm_sites | amylase | 90 | 0.36125 | — |
| G1 | sequence_all | esm_sites | gb1 | 90 | 0.75012 | — |
| G1 | sequence_all | esm_sites | nucb | 90 | 0.34738 | — |
| G1 | sequence_all | mutation_count_ridge | amylase | 90 | 0.27297 | — |
| G1 | sequence_all | mutation_count_ridge | gb1 | 90 | 0.20465 | — |
| G1 | sequence_all | mutation_count_ridge | nucb | 90 | -0.00827 | — |
| G1 | sequence_all | sequence_ridge | amylase | 90 | 0.40741 | — |
| G1 | sequence_all | sequence_ridge | gb1 | 90 | 0.49358 | — |
| G1 | sequence_all | sequence_ridge | nucb | 90 | 0.13915 | — |
| G1 | sequence_all | within_one_hot | amylase | 90 | 0.58350 | — |
| G1 | sequence_all | within_one_hot | gb1 | 90 | 0.76411 | — |
| G1 | sequence_all | within_one_hot | nucb | 90 | 0.49444 | — |
| G1 | structure_matched | esm_sites | amylase | 90 | 0.33177 | — |
| G1 | structure_matched | esm_sites | gb1 | 90 | 0.75379 | — |
| G1 | structure_matched | esm_sites | nucb | 82 | 0.39397 | — |
| G1 | structure_matched | mutation_count_ridge | amylase | 90 | 0.27297 | — |
| G1 | structure_matched | mutation_count_ridge | gb1 | 90 | 0.20465 | — |
| G1 | structure_matched | mutation_count_ridge | nucb | 82 | 0.02124 | — |
| G1 | structure_matched | sequence_ridge | amylase | 90 | 0.43748 | — |
| G1 | structure_matched | sequence_ridge | gb1 | 90 | 0.53372 | — |
| G1 | structure_matched | sequence_ridge | nucb | 82 | 0.21034 | — |
| G1 | structure_matched | structural_gnn | amylase | 90 | 0.58552 | 0.03868 |
| G1 | structure_matched | structural_gnn | gb1 | 90 | 0.64658 | 0.00839 |
| G1 | structure_matched | structural_gnn | nucb | 82 | 0.17918 | 0.03777 |
| G1 | structure_matched | structural_ridge | amylase | 90 | 0.56811 | — |
| G1 | structure_matched | structural_ridge | gb1 | 90 | 0.54595 | — |
| G1 | structure_matched | structural_ridge | nucb | 82 | 0.26943 | — |
| G1 | structure_matched | within_one_hot | amylase | 90 | 0.57998 | — |
| G1 | structure_matched | within_one_hot | gb1 | 90 | 0.76411 | — |
| G1 | structure_matched | within_one_hot | nucb | 82 | 0.53209 | — |
| G2 | sequence_all | esm_sites | amylase | 600 | 0.12469 | — |
| G2 | sequence_all | esm_sites | gb1 | 600 | 0.14860 | — |
| G2 | sequence_all | esm_sites | nucb | 600 | 0.00637 | — |
| G2 | sequence_all | mutation_count_ridge | amylase | 600 | 0.36428 | — |
| G2 | sequence_all | mutation_count_ridge | gb1 | 600 | 0.19098 | — |
| G2 | sequence_all | mutation_count_ridge | nucb | 600 | 0.02462 | — |
| G2 | sequence_all | sequence_ridge | amylase | 600 | 0.19679 | — |
| G2 | sequence_all | sequence_ridge | gb1 | 600 | 0.15792 | — |
| G2 | sequence_all | sequence_ridge | nucb | 600 | 0.03950 | — |
| G2 | structure_matched | esm_sites | amylase | 578 | 0.15582 | — |
| G2 | structure_matched | esm_sites | gb1 | 600 | 0.20638 | — |
| G2 | structure_matched | esm_sites | nucb | 553 | 0.01512 | — |
| G2 | structure_matched | mutation_count_ridge | amylase | 578 | 0.36279 | — |
| G2 | structure_matched | mutation_count_ridge | gb1 | 600 | 0.19098 | — |
| G2 | structure_matched | mutation_count_ridge | nucb | 553 | 0.03581 | — |
| G2 | structure_matched | sequence_ridge | amylase | 578 | 0.20435 | — |
| G2 | structure_matched | sequence_ridge | gb1 | 600 | 0.17322 | — |
| G2 | structure_matched | sequence_ridge | nucb | 553 | 0.05077 | — |
| G2 | structure_matched | structural_gnn | amylase | 578 | 0.24154 | 0.06049 |
| G2 | structure_matched | structural_gnn | gb1 | 600 | 0.18120 | 0.01792 |
| G2 | structure_matched | structural_gnn | nucb | 553 | -0.03263 | 0.08142 |
| G2 | structure_matched | structural_ridge | amylase | 578 | 0.34578 | — |
| G2 | structure_matched | structural_ridge | gb1 | 600 | 0.18755 | — |
| G2 | structure_matched | structural_ridge | nucb | 553 | -0.03265 | — |

## Same-test-ID comparison

Primary G2 evaluates every selected variant of a held-out protein, whereas G1 evaluates only its original held-out variant partition. The following additional descriptive comparison restricts G2 predictions to exactly the G1 test IDs; it does not retrain/select models.

| Cohort | Model | G1 macro rho | G2 macro rho on same test IDs |
|---|---|---:|---:|
| sequence_all | esm_sites | 0.48625 | 0.11574 |
| sequence_all | mutation_count_ridge | 0.15645 | 0.15645 |
| sequence_all | sequence_ridge | 0.34671 | 0.15185 |
| structure_matched | esm_sites | 0.49318 | 0.11823 |
| structure_matched | mutation_count_ridge | 0.16629 | 0.16629 |
| structure_matched | sequence_ridge | 0.39385 | 0.16492 |
| structure_matched | structural_gnn | 0.47043 | 0.14971 |
| structure_matched | structural_ridge | 0.46116 | 0.14922 |

G1 and G2 still use different training/validation protein sets and budgets, so this is a protocol comparison, not a causal ablation. With one assay per protein, protein shift and assay-endpoint shift are confounded.

## G2 roles and leakage controls

- Test amylase: train ['nucb'], validation ['gb1'], test ['amylase'].
- Test gb1: train ['amylase'], validation ['nucb'], test ['gb1'].
- Test nucb: train ['gb1'], validation ['amylase'], test ['nucb'].

No supervised rows from validation/test proteins enter model fitting or feature scaling. Validation-protein labels only select hyperparameters/epochs. Unit tests poison test labels and require unchanged fits and predictions.
Three proteins permit one source-training protein per G2 fold; results therefore depend on which source and validation protein are paired with each test protein. This small pilot is not a population-level estimate.
Within-protein one-hot ridge is G1-only: unaligned position-specific columns cannot defensibly transfer across unrelated lengths. G2 uses a shared mutation-composition/property sequence ridge and mutation-count ridge, plus the unchanged frozen ESM site representation.
The neural model shares one scalar head and all weights; there are no assay IDs in model inputs or task-specific heads. Source-assay rank normalization is target handling, not a stability/function multitask objective.

## Reference and structural coverage

- amylase: 415 compatible nodes; 22 sampled structural exclusions; 0 source rows excluded for changes outside the declared domain crop. PDB differences: ['S105F', 'P126Q', 'S142L', 'D163E', 'V214A', 'V254S', 'N356Q'].
- gb1: 55 compatible nodes; 0 sampled structural exclusions; 0 source rows excluded for changes outside the declared domain crop. PDB differences: ['Q2T'].
- nucb: 109 compatible nodes; 47 sampled structural exclusions; 35 source rows excluded for changes outside the declared domain crop. PDB differences: [].

GB1 is cropped to the variable 56-residue binding domain, removing its fixed assay fusion partner. NucB removes a fixed signal-peptide prefix; 35 source variants that change this prefix are explicitly excluded. This conditions the benchmark on the defined domain and does not model secretion/fusion effects.
Sequence-only cohorts include structural-invalid variants. Structure-matched cohorts refit every compared model on exactly the same eligible train/validation/test rows. Missing graph residues are excluded, not imputed. Descriptor-missing masks remain explicit. Experimental B-factors are not pLDDT.
Geometry describes reference constructs, not mutant rearrangements. The full M2 alignment/confidence audits are nested in audit.json.

## Sequence similarity (reference domains)

| Pair | Aligned identity | Coverage left/right | Identity / shorter length |
|---|---:|---|---:|
| gb1 / amylase | 0.7037 | 0.9643 / 0.1271 | 0.6786 |
| nucb / amylase | 0.5463 | 0.9818 / 0.2541 | 0.5364 |
| nucb / gb1 | 0.4182 | 0.5000 / 0.9821 | 0.4107 |

Global alignment has gap/length biases, especially for short domains. Family annotations are descriptive; no profile-based evolutionary family separation has been verified. **No unseen-family generalization is claimed.** Frozen ESM pretraining overlap is unknown; this is supervised protein-held-out evaluation, not pretraining exclusion.

## Failure analysis

Compare G1 with G2 on the same cohort rather than selecting a model after seeing held-out test scores. Negative correlations and undefined constant-prediction correlations are retained. Undefined validation correlations count as zero during selection; reports disclose undefined test correlations and their denominators.
NucB is a tied four-level ordinal activity label, GB1 measures binding in a fusion assay, and Alpha Amylase measures activity. Rank alignment removes scale differences but does not make these biological endpoints identical. Source/assay shift, sparse GB1 mutated sites, mutation-burden differences and structure exclusions can dominate transfer.
Two GNN seeds measure initialization/training variation, not uncertainty over future proteins. Small macro seed SD can hide opposing per-protein variation; inspect the per-protein table. Three protein folds are too few for a broad generalization claim.

## Compute and reproducibility

Actual local run: 59.922 seconds; peak process RSS 640221184 bytes; Python 3.12.15; deterministic CPU GNN. Frozen ESM preparation uses the existing pinned checkpoint (MPS where available).
Historical M0-M3 artifacts verified unchanged: 103.
Per-fit runtime, selections, training/validation correlations and epoch traces are in runs.json. Model weights and dense embeddings stay in ignored caches. Dataset hashes, source revisions, structure identities, seeds, exact role IDs and exclusions are in config/manifest/audit files.
See docs/m4_protocol.md for exact commands and citations. Cross-attention, fusion, LoRA, generative design and multitask heads are excluded.
