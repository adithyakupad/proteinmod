# ProteinMod

ProteinMod implements reproducible supervised activity prediction on **FLIP2 Alpha Amylase,
by-mutation**, a single-protein assay. It does **not** demonstrate cross-protein
or family generalization. M0 contains elementary baselines; M1 adds frozen ESM-2 features.

## Install and reproduce

Python **3.11–3.14** is supported; the measured run used Python 3.12.15 on Apple
silicon. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
proteinmod download --config configs/amylase.toml
proteinmod validate --config configs/amylase.toml
proteinmod run --config configs/amylase.toml
ruff check .
pytest
```

For M0, only `download` accesses the network. M1 adds explicit model extraction below. `validate`, `run`, and tests work offline
once dependencies and data are present. CI uses synthetic fixtures, never downloads
the benchmark. For an existing exact official file:

```sh
proteinmod run --config configs/amylase.toml --data /absolute/path/by_mutation.csv.gz --output runs/local
```

The configured checksum is enforced for local files too. Paths in TOML are relative
to the working directory. `data/cache/`, virtual environments and `runs/` are ignored.
The download stores a source/checksum manifest beside the cached file. To reproduce
the recorded dependency versions, install `requirements-benchmark.txt` before the
editable package; these pins describe the Python 3.12 measured environment.

## Provenance and scientific controls

- Official [FLIP2 catalog](https://flip.protein.properties/), versioned
  [Zenodo record 18433203, v3](https://zenodo.org/records/18433203), and
  [Didi et al. (2026)](https://proceedings.mlr.press/v306/didi26a.html).
- Zenodo's amylase README attributes the assay to Van der Flier et al., *Enzyme
  structure correlates with variant effect predictability*, CSBJ 23 (2024),
  3489–3497, and links the [original MIT-licensed data](https://github.com/florisvdf/mutation-predictability/tree/main/data/AMY_BACSU).
- The config records the published compressed-file MD5. Every run records SHA256,
  complete config, reference provenance, package versions, Git revision/dirty state
  and elapsed time. The original assay `reference.fasta` is bound to Git blob
  `920b6c14b659029dd1f5497329dab98c0c8755e2`; its 425-residue sequence is embedded in
  the config. The blob check uses the exact original FASTA header `> 0` and newline.
  A custom reference requires source attribution plus a Git blob or sequence SHA256.
  A checksum binds bytes; the source must independently establish biological WT identity.
- Canonical uppercase residues, equal lengths, finite numeric targets, missing/duplicate
  IDs, duplicate sequences and assignments are checked. Official CSVs have no IDs:
  stable 1-based source row IDs are generated and explicitly identified in QC.
  Invalid inputs fail instead of silently being dropped or deduplicated.
- The official CSV has `set=train, validation=True` for 601 rows. These become the
  effective validation partition and are **excluded from fitting**. Test/validation
  overlap is rejected. No random split, resampling or test-label model selection occurs.
- QC reports lengths, effective counts, validation/raw-set overlap, sequence leakage,
  per-partition mutation burdens and pairwise unique-substitution overlap. The actual
  official distribution has **410 shared substitutions between effective train and
  test**, despite no shared full sequences. We preserve the official file and report
  this observation; do not interpret its name as strict substitution disjointness.
- Ridge uses fixed full-sequence 20-letter position features or a single mutation-count
  feature. No scaling is used. All candidates fit only training rows, choose minimum
  validation RMSE (tie: smallest alpha), and remain fit on training rows only. Seed 0,
  deterministic LSQR solver, tolerance 1e-8, at most 10,000 iterations. Test labels
  are used only for the final evaluation. Constant predictions have undefined
  correlations, serialized as JSON null rather than invented zeros.
- Removing `[reference]` permits mean and one-hot models while explicitly skipping
  mutation-count ridge and mutation annotations. An invalid supplied reference fails
  clearly. WT is never guessed from variant frequency.

## Actual M0 result

Measured on the official downloaded file on October 14, 2026: **3,706** variants,
length **425**; **2,403 train / 601 validation / 702 test**. No duplicate sequences
or IDs. Source SHA256:
`2e76acc9da3ef04a8ccecd8a3f631b84ee2e4e1f0cdea406f18153387b986d04`.

| Baseline | Test Spearman rho | Test Pearson r | MAE | RMSE |
|---|---:|---:|---:|---:|
| Training mean | undefined | undefined | 0.048968 | 0.053985 |
| Mutation-count ridge | 0.229543 | 0.237123 | 0.042635 | 0.053048 |
| Full-sequence one-hot ridge | 0.648837 | 0.683810 | 0.033651 | 0.042845 |

Selected alpha: **1** for one-hot, **10** for mutation count, chosen only on validation.
These are ProteinMod M0 scores, not a claim to replicate the paper's entire protocol.
Checked-in evidence lives in `benchmarks/amylase/`: `metrics.json`, `qc.json`,
`manifest.json`, `test_predictions.csv` and `report.md`. New runs generate the same
artifact set under `runs/amylase/`. Scores use the distributed target units directly.

## How to interpret the baseline

A held-out mutation split evaluates unseen variants/substitutions on the same protein;
positions can still be familiar. A held-out position split tests previously unperturbed
positions. A held-out protein split tests transfer to different protein backbones.
The overlap statistics above qualify what is actually held out in this distribution.

Given WT `ACDE` and variant `CCDF`, compare aligned residues: position 1 changes
A to C and position 4 changes E to F, giving the sorted set `A1C;E4F` and burden 2.
Length mismatches cannot be represented as substitutions and are rejected. Full-sequence
one-hot ridge learns regularized additive contributions of residue identity at each
position. It is a meaningful inexpensive control for learned representations, although
it cannot explicitly model epistasis or establish transfer to other proteins.

## M1: frozen ESM-2 (implemented)

Install the optional experiment dependencies and run from the repository root:

```sh
source .venv/bin/activate
python -m pip install -e '.[dev,esm]'
# Optional: install the exact measured Python 3.12 environment first:
# python -m pip install -r requirements-m1.txt
proteinmod download --config configs/amylase_m1.toml
HF_HUB_DISABLE_XET=1 proteinmod extract --config configs/amylase_m1.toml
MPLCONFIGDIR=/tmp/proteinmod-mpl XDG_CACHE_HOME=/tmp/proteinmod-cache OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 proteinmod run-m1 --config configs/amylase_m1.toml
ruff check .
pytest
```

`extract` explicitly downloads the pinned checkpoint/tokenizer if absent; it is the
only M1 command requiring network access. Reuse existing files without network:

```sh
proteinmod extract --config configs/amylase_m1.toml --local-only
proteinmod run-m1 --config configs/amylase_m1.toml --output runs/amylase_m1_repeat
```

`run-m1` uses verified cached features and refuses to overwrite M0's historical
artifact directory. M0 config, code paths, historical scores and predictions are
preserved. All three M0 baselines reproduced **exactly (maximum prediction
difference 0)** in the matched M1 experiment. The same data checksum and 702 test
identities/targets are checked before accepting the comparison. Original 32 tests
continue to pass; CI runs offline mock encoders without Torch or model weights.

The [official ESM-2 35M checkpoint](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
is MIT-licensed. Model and tokenizer are pinned to revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`. Use the optional dependency pins in
`requirements-m1.txt` for the measured environment. The implementation uses frozen
eval mode, no gradient updates, float32 last-layer residue states, batches of 8,
explicit special/padding masks and residue-token checks; overlong sequences fail
instead of being truncated. Model/dataset downloads, regression and plotting are
separate steps. Caches bind dataset SHA, ordered sequences, WT, revision, settings
and extractor source hash; content checksums and shapes are verified on reuse.
Changing an extraction setting creates a new cache. A corrupt cache fails; remove
that cache directory and re-extract. Interrupted extraction is rerun from the start.

Set `embedding.device` in TOML to `cpu`, `mps`, `cuda` or `auto` (CUDA, then MPS,
then CPU). Unsupported devices fail clearly. Reduce batch size for limited memory;
there is no hidden fallback or truncation. This local run used **MPS**, Python
3.12.15, Torch 2.14.1, Transformers 4.57.6 and float32 inference. WT plus all 3,706
variants extracted in **224 seconds**, including initial checkpoint loading;
the local model cache is about **134 MiB** and compressed feature cache **26.3 MiB**.
Inference additionally needs batch-dependent activation memory; CPU is supported
but was not timed. Checkpoints and embeddings remain under ignored `data/cache/`.

Actual full-data results on October 14, 2026:

| Representation | Test Spearman rho | Test Pearson r | MAE | RMSE |
|---|---:|---:|---:|---:|
| M0 one-hot ridge | 0.648837 | 0.683810 | 0.033651 | 0.042845 |
| ESM mutant mean (480) | 0.498379 | 0.515328 | 0.040200 | 0.046462 |
| ESM mutant-minus-WT mean (480) | 0.498379 | 0.515328 | 0.040200 | 0.046462 |
| ESM mutation-site aggregate (961) | 0.634051 | 0.687888 | 0.031407 | 0.039956 |

Site features improve MAE/RMSE over one-hot, while one-hot retains higher Spearman.
Full-sequence mean pooling underperforms one-hot in this run. Do not interpret
pretraining as an unconditional improvement or these paired scores as significance tests.

Learning curves use **25, 121, 241, 601, 1,202 and 2,403** training labels with three
nested subset seeds (0/1/2). At 25 training labels, site features achieve Spearman
**0.301947 ± 0.020724** versus one-hot **0.210356 ± 0.025385**; RMSE is
**0.052909 ± 0.003142** versus **0.055495 ± 0.001535**. At 121 labels, site features
achieve Spearman **0.458044 ± 0.033698** versus **0.380898 ± 0.058901**.
These are mean ± sample SD across training subset seeds, **not test confidence
intervals**. Every model uses the same subsets and fixed validation/test partitions.
The 601 validation labels are still available at every training size: these are
low-training-label experiments, not 25-label total annotation budgets.

![M1 learning curves](benchmarks/amylase_m1/learning_curves.png)

All six models and four metrics at every training size are recorded in
`benchmarks/amylase_m1/learning_curve_runs.json` and `learning_curve_summary.csv`.
The same directory contains full-data predictions, QC, model/cache/run provenance,
PNG/SVG plots and a report. New runs go to `runs/amylase_m1/`. The protocol is
in `docs/m1_protocol.md`; alphas and representation choices were fixed before
inspecting M1 test outcomes. Ridge still uses training rows only and validation
RMSE selection without feature scaling or validation refit.

A mutant mean summarizes the whole sequence. Its WT delta subtracts a constant
vector for this single-WT assay; with ridge's intercept they are mathematically
equivalent, as the measured results confirm within numerical precision. They can
differ for multiple WTs, models without an intercept, nonlinear heads, or different
preprocessing. Site features average mutant states and mutant-minus-WT states only
at substituted sites and append site count, with zeros for WT. Symmetric pooling
handles multi-site mutants without depending on mutation ordering and can retain
local effects that global pooling dilutes. Pretraining may supply useful biochemical
structure with few labels; it can also miss assay-specific signal.

Limits: one protein, **410 shared train/test substitutions**, only three subset
seeds, and unknown pretraining-corpus overlap. Extracting held-out sequence features
is **label-free transductive evaluation**, not cross-protein generalization.
M1 adds no GNN, structural integration, cross-attention, multitask head or fine-tuning.

## Roadmap (not implemented)

Sequence PLM + structural GNN + bidirectional cross-attention; stability/function
heads; advanced multi-mutation modeling; progressive protein/family-held-out evaluation.
M0 already annotates multiple substitutions but implements only elementary supervised
baselines. Data validation, features, fitting and artifact output are separate modules
under `src/proteinmod` so later features can be added without changing split handling.
