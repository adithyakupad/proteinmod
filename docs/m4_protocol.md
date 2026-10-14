# M4: a small cross-protein generalization benchmark

## Purpose and dependency verification

M4 establishes protein-held-out evaluation before M5. It does not implement cross-attention, LoRA, fusion, generative design or task-specific stability/function heads. The objective is rank prediction across different experimental endpoints, not predicting all assays in a common physical unit.

PR #6 was verified against its successful Python 3.11–3.14 CI and M2 preservation audit, marked ready and merged as `09d306b`. PR #7 was retargeted to main, updated with a normal history-preserving merge (no content conflicts), rerun locally (87 tests plus Ruff), and merged only after all head `cf8aced` checks passed. Its merge is `5068e74`. M4 starts from that main on `codex/m4-cross-protein`.

Historical directories `benchmarks/amylase`, `amylase_m1`, `structures` and `amylase_m3` are hashed before and after experiments. M0–M3 configuration and numerical evidence are never overwritten. The existing M3 GNN accepts an optional graph tuple so the **same weights** can operate on different residue counts; the original call with one argument is unchanged and tested for exact equivalence.

## Pinned, bounded dataset collection

Only three assays from [FLIP2 Zenodo v3](https://doi.org/10.5281/zenodo.18433203) are used. No ProteinGym archive is downloaded. Research considered ProteinGym's assay/WT metadata and rank-based evaluation; FLIP2 provides small individual files and dataset-specific licenses, avoiding a broad archive download.

| Protein | Source file | Experimental endpoint | Reference/domain | Structure |
|---|---|---|---|---|
| Alpha Amylase | `amylase/by_mutation.csv.gz` | Continuous functional activity | Existing verified M0 reference, 425 aa | M2 1UA7 chain A |
| GB1 | `gb1/sampled.csv.gz` | IgG binding | First 56 aa of the designated WT construct; fixed fusion remainder excluded | 1PGA chain A |
| NucB | `nucb/random.csv.gz` | Four ordered activity categories 0–3 | Primary-source full WT positions 33–142, 110 aa mature enzyme | 5OMT chain A |

GB1: Wu et al. (2016), [eLife 5:e16965](https://doi.org/10.7554/eLife.16965), CC-BY-4.0 as documented in the [FLIP2 dataset README](https://zenodo.org/records/18433203/files/gb1/README.md). The pinned dataset's designated WT first row supplies the exact recombinant construct (WT-normalized binding 1.0); domain sequence is explicitly pinned, never inferred by maximizing target values. The native 1PGA construct has T2 versus assay Q2. This node is excluded, even though the four mutagenized GB1 sites are elsewhere. GB1 is an assay fusion: domain-only inputs omit its fixed fusion partner, not claim a native isolated-domain experiment.

NucB: Thomas et al. (2024), [Engineering highly active and diverse nuclease enzymes](https://doi.org/10.1101/2024.03.21.585615), CC-BY-4.0 data as documented by [FLIP2](https://zenodo.org/records/18433203/files/nucb/README.md) and the [primary repository](https://github.com/google-deepmind/nuclease_design). The full WT and meaning of the four activity levels are explicit in [constants.py at 7022848](https://github.com/google-deepmind/nuclease_design/blob/702284847c4558567d8288169a86dc147cfaab32/nuclease_design/constants.py). Ordinal categories are not continuous kinetic measurements. The fixed signal-peptide prefix is removed, and 35 raw variants changing outside the crop are excluded and listed. The remaining domain exactly matches 5OMT's declared sequence; one residue lacks coordinates. [5OMT](https://doi.org/10.2210/pdb5OMT/pdb) and [1PGA](https://doi.org/10.2210/pdb1PGA/pdb) coordinates are wwPDB CC0; cite their depositions/publications.

Amylase attribution/license remains as recorded in M0/M1 (original assay MIT; Zenodo collection CC-BY-4.0). Family labels (GH13 amylase, Protein G Ig-binding domain, NucB nuclease) are descriptive metadata, not certified evolutionary separation.

`configs/m4.json` binds raw file SHA256/MD5, WT sequence hashes, crop coordinates, structure hashes/IDs, licensing, model revision and all experiment settings. The new adapter explicitly interprets GB1's blank validation flag as false; M0's strict loader is unchanged. All targets, canonical sequence lengths, duplicates and original partition membership are validated by the existing loader before cropping. Crop changes outside the declared domain are explicit exclusions, not silently ignored mutations.

From each original partition select the smallest SHA256(protein ID, cropped sequence) values, capped at **420 train / 90 validation / 90 test per protein**, independent of labels and structural coverage. Total 1,800 variants, with single and multiple mutants as present in each source. This is a bounded pilot, not a leaderboard/full-dataset claim.

## Protocols and scale handling

**G1:** Shared models fit the original training variants from all three known proteins; validation/test retain their official membership. An additional per-protein one-hot ridge is a G1-only M0-style comparator. Training row counts and the supervised budgets are disclosed; this per-protein comparator does not share weights.

**G2:** Three cyclic protein folds, each with one distinct training, validation and test protein. All 600 selected rows of a protein take its assigned role, regardless of its G1 variant partition. Protocols are independent experiments: no model or fitted scaler/label transform is reused between them.

| Held-out test | Supervised training | Validation-only selection |
|---|---|---|
| Amylase | NucB | GB1 |
| GB1 | Amylase | NucB |
| NucB | GB1 | Amylase |

No supervised example from the G2 test or validation protein enters fitting. The validation protein selects hyperparameters/epochs but never updates weights. This three-protein design is deliberately small; source and validation identities are confounded with each held-out test. It does not estimate performance after training on a large, diverse protein population.

For each training assay, transform **only its training labels** to empirical midrank percentiles `(average_rank - 0.5) / n_train`. The one shared scalar regression objective therefore concerns relative experimental performance, not raw values pooled across unlike assays. Validation and test labels are never used to fit a target transform or calibrate predictions; their raw ordering suffices for Spearman correlation. No protein/assay IDs are fed to a model, and no task-specific heads or multitask losses are introduced.

Select ridge alpha from `[0.1, 10, 100]` by validation macro per-protein Spearman; ties choose smaller alpha. All input scaling fits the current training rows only. GNN settings are fixed before experiments: the M3 width-16/two-layer architecture, AdamW LR 0.003, weight decay 0.01, batch size 64, at most 30 epochs, patience 8, seeds 0/1. Each homogeneous-protein batch uses that graph while sharing every learned parameter. Validation macro Spearman selects the epoch; ties prefer the earliest. There is no validation refit or test-driven architecture search.

Report per-protein Spearman and equal-protein-weight macro scores; never pool raw measurements or average raw-unit RMSE. Undefined correlations are disclosed; constant-prediction correlations count as zero for fixed-denominator macro scores/selection (not silently dropped). Neural seed SD measures two initializations, not future-protein uncertainty. Keep negative outcomes.

## Models and matched cohorts

- Mutation-count/fraction ridge and transferable sequence ridge. Sequence features: full-domain amino-acid composition, mean WT/substituted identities at mutation sites, mean residue-property changes, mutation count/fraction and mean relative site position. Position-specific M0 columns have no defensible shared homology across these lengths, so native one-hot ridge is reported only in G1.
- Frozen ESM-2 mutation-site representation, unchanged 35M checkpoint/revision and pooling from M1. Amylase reuses its original verified complete cache; the two new **cropped domains** have explicitly versioned extraction caches. Only probes are trained.
- M3 handcrafted mutation-centered structural ridge, including RSA/contact descriptors, neighborhood composition, residue properties and mutation-pair distances.
- Shared compact structural GNN with mutation-aware inputs, distance/type edges, invariant mutation readout and global context. Same graph construction/feature definitions as M2/M3; no sequence/structure feature fusion with ESM.

`sequence_all` includes all 1,800 sampled rows. `structure_matched` includes the 1,731 rows whose **every** mutated site has an identity-compatible C-alpha mapping: 578 amylase, 600 GB1, 553 NucB. Every sequence and structure model in that cohort is fitted/evaluated on identical role IDs. It is invalid to compare a sequence-all score directly to a structure-matched score. Structural-invalid variants have an explicit boolean mask and reason; zero placeholder vectors are never fitted or scored. Full graph confidence/alignment audits are retained. Reference geometry is not mutant rearrangement.

## Leakage and similarity audits

Assert disjoint variant IDs/full sequences across roles and disjoint supervised proteins in G2; reject aliased protein IDs with identical references. Keep exact selected/role IDs and source row identifiers. Tests alter held-out labels and require identical target transforms, ridge selection, neural epoch selection and predictions. Tests cover distinct node counts with shared model weights and unchanged M3 forward behavior.

Compute all pairwise global reference-domain alignments with explicit scoring, aligned identity and coverage of both proteins. Gap-rich/short-domain alignment identities are not family certificates. No HMM/profile family separation or pretraining exclusion is established, so **no unseen-family generalization is claimed**. G1 remains interpolation among variants with shared substitutions/design history; official random partitions are not mutation-distance-separated. Frozen ESM may have encountered homologs during unsupervised pretraining.

## Reproduction

Python 3.12 from repository root; existing M1 caches can be reused:

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,esm,structure,m3]' -r requirements-m3.txt
.venv/bin/proteinmod download-m4 --config configs/m4.json
# On a fresh machine, build the existing amylase frozen cache first:
.venv/bin/proteinmod extract --config configs/amylase_m1.toml
.venv/bin/proteinmod prepare-m4 --config configs/m4.json --local-only
.venv/bin/proteinmod run-m4 --config configs/m4.json
.venv/bin/python scripts/report_m4.py --output runs/m4
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
```

Only `download-m4` acquires the small raw assays/structures. Preparation uses local raw coordinates and a cached frozen model with `--local-only`; training/evaluation is offline. CPU GNN execution is deterministic on Apple Silicon; MPS is used for frozen extraction where available. M3's real deterministic-MPS aggregation failure remains documented; M4 does not silently relax determinism.

Configuration, source hashes, dependency versions, seeds, raw dataset hashes, structure identities, model revision, selected/role IDs, exclusions, per-protein predictions, metrics, plots, runtime and memory are preserved. Expensive dense embeddings and weights remain ignored. Recompute report metrics from predictions and verify their checksums; output defaults to ignored `runs/m4`, not historical directories.
