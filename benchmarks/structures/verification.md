# Reproducible M2 verification

Executed locally on October 14, 2026, Python 3.12.15, macOS arm64, Biopython 1.88,
NumPy 2.5.4, Matplotlib 3.11.2. No model weights or accelerator were needed.

- Editable install `python -m pip install -e '.[dev,structure]' --no-build-isolation`
  succeeded. The minimal measured dependency pins are in `requirements-m2.txt`.
- All five explicit `proteinmod structure-download --config configs/<case>.toml`
  commands succeeded against official RCSB/AlphaFold endpoints. Configs pin SHA256;
  each case includes its actual `acquisition_manifest.json`. Raw coordinates stay
  in the ignored cache, not in Git.
- Final offline command from clean implementation commit `57d014d`:
  `MPLCONFIGDIR=/tmp/proteinmod-mpl XDG_CACHE_HOME=/tmp/proteinmod-cache python scripts/verify_structures.py`.
  Each case manifest records that exact clean code revision, source hashes, input
  checksum, configuration, library versions, timing and output checksums.
- 1UA7 PDB/mmCIF: 422/425 Cα coverage; 415 exact identity-compatible sites; seven
  reviewed differences; 2,195 edges; no coordinate/RSA/contact-record discrepancy.
- 1UBQ PDB/mmCIF: 76/76 coverage and identity-compatible sites; 326 edges; no
  coordinate/RSA/contact-record discrepancy.
- AF-P00691-F1 v6 mmCIF: 425/425 Cα coverage of the assay span, 421 identity-compatible
  sites, four differences, 2,235 edges; pLDDT mean 97.273835, min 78.81, max 99.0 in
  that span. Experimental B-factors are never interpreted as pLDDT.
- Unreviewed assay/1UA7 mapping was deliberately rejected with seven differences;
  see `unreviewed_amylase_rejected/audit.json`. Even after reviewed approval,
  mutation sites at mismatches or absent coordinates retain null context features.
- All 19 historical M0/M1 artifact hashes stayed identical; recorded individually
  in `verification.json`. Original M0/M1 configs, feature/fit/benchmark/experiment
  modules and all existing tests have no diff from main (the CLI only adds commands).
- `pytest`: **78 passed**, including all 48 original M0/M1 tests. Also **78 passed**
  in a fresh lightweight environment containing Biopython but no Torch, Transformers
  or Matplotlib. Tests use synthetic PDB/mmCIF and mocks, never real-file downloads.
- `ruff check .`, `ruff format --check .`, `git diff --check`: passed.
- Real-protein PNG figures were visually inspected; PNG/SVG are retained per case.

DSSP was unavailable locally: secondary structure fields are null and explicitly
marked unavailable. The optional executable integration is implemented but no
actual DSSP assignments are claimed. No pockets, GNNs, cross-attention, fusion,
fine-tuning or new fitness/stability models were implemented. These checks validate
mapping and reference geometry, not biological effects or prediction improvements.

The first five FLIP2 source rows are included as traceable variant contexts without
using their activity labels; synthetic diagnostics are labeled as such. See each
case report and the full protocol in `docs/m2_protocol.md` for representation limits,
solvent environment, modified-residue normalization and citation/license details.
