# Local verification, October 14, 2026

- Editable install with optional experiment dependencies succeeded:
  `python -m pip install -e '.[dev,esm]' --no-build-isolation`.
- Downloaded the pinned official Hugging Face safetensors checkpoint and tokenizer
  through `HF_HUB_DISABLE_XET=1 proteinmod extract --config configs/amylase_m1.toml`.
  Actual WT + 3,706 variant extraction completed in 224.123 seconds on Apple MPS.
  Model/feature caches are ignored; checkpoint file checksums are retained in
  `checkpoint_files.json`, extraction provenance in `manifest.json`.
- `proteinmod extract --config configs/amylase_m1.toml --local-only` verified and
  reused the completed embedding cache without loading weights or network access.
- Ran all models and the 1/5/10/25/50/100% nested training curves locally with seeds
  0, 1, 2. All six representations/baselines use the exact same subsets and fixed
  validation/test rows. There are 108 model/seed/fraction evaluations; identical
  100% fits are reused across seeds. Every ridge alpha is selected on validation.
- Final command, from clean implementation commit recorded in `manifest.json`:
  `MPLCONFIGDIR=/tmp/proteinmod-mpl XDG_CACHE_HOME=/tmp/proteinmod-cache OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 proteinmod run-m1 --config configs/amylase_m1.toml`.
- M0 test identities, measured labels and compressed-data SHA matched. Historical
  baseline predictions reproduced with max absolute difference **0.0 for all three
  baselines**. Original `benchmarks/amylase`, original config, fit/data/benchmark
  code and existing 32 tests have no diff from main. Historical file hashes are
  recorded in `metrics.json`.
- `pytest`: **48 passed**, including all original 32 tests. New tests run with
  synthetic/mock features, require no Torch installation or weights, verify token
  alignment and special/pad exclusion, multi-site symmetry, cache identity/checksum,
  frozen/eval initialization, nesting, test-label isolation and M0 preservation.
- `ruff check .`: passed. `git diff --check`: passed.
- PNG plot was visually inspected; PNG and SVG plots are included.
- Local compute: Python 3.12.15, macOS arm64, float32 frozen encoder, float64 ridge
  features, batch 8, four CPU threads configured. Detailed package versions and
  source hashes are in the manifest and `requirements-m1.txt`.

M0 history is not overwritten. This evidence commit follows the clean code commit
used for the final experiment. No test-result-driven architecture tuning occurred.
Seed SD describes training-subset variability, not test uncertainty; the full
601-label validation set remains available even for 25 training labels.

A fresh core-only Python 3.12 environment was also created and installed with
`.[dev]`; Torch, Transformers and Matplotlib were confirmed absent. All **48 tests
passed** there. The plot test mocks optional plotter metadata as well as rendering;
this fixes an isolation issue revealed by the first CI run and changes no experiment
or prediction code. All recorded package source hashes remain current.
