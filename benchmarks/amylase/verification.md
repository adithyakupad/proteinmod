# Verification evidence

Executed locally on October 14, 2026, Python 3.12.15, macOS arm64.

- `python -m pip install -e '.[dev]'`: succeeded, editable wheel installed.
- `proteinmod download --config configs/amylase.toml`: succeeded against Zenodo;
  configured published MD5 matched. See `download_manifest.json` for SHA256.
- `proteinmod validate --config configs/amylase.toml`: succeeded; `qc.json` records
  all 3,706 rows, official assignments and biological checks.
- `proteinmod run --config configs/amylase.toml`: succeeded on downloaded official
  data. `manifest.json` records the clean implementation commit and source hashes;
  `metrics.json` records all validation candidates and actual test metrics;
  `test_predictions.csv` contains 702 measured/predicted test rows.
- `ruff check .`: passed.
- `pytest`: **32 passed**, offline synthetic fixtures, including checksum failure,
  multi-site mutations, invalid reference provenance, split overlap, duplicate
  leakage, numerical metrics, deterministic fit, test-label isolation and missing WT.
- `git diff --check`: passed.

CI is configured for Python 3.11–3.14; only Python 3.12 was executed locally.
This artifact commit follows the implementation commit recorded in the run manifest;
no source code changed between that benchmark and the saved evidence.
