# ProteinMod M0 — FLIP2 Alpha Amylase

Single-protein activity assay; this is not cross-protein generalization.

Dataset SHA256: `2e76acc9da3ef04a8ccecd8a3f631b84ee2e4e1f0cdea406f18153387b986d04`

Effective partitions: {'train': 2403, 'validation': 601, 'test': 702}. Lengths: {425: 3706}.
Validation flag / raw set overlap: {'train': 601}.
Validation-flagged training rows are excluded from fitting. No sequence overlap.
Unique substitution overlaps: {'train/validation': 867, 'train/test': 410, 'validation/test': 272}.
The official by-mutation file is preserved; shared substitutions are reported, so this run does not claim strictly disjoint substitutions.
Verified WT: True; mutation burden: {'train': {0: 1, 1: 355, 2: 753, 3: 698, 4: 382, 5: 167, 6: 37, 7: 8, 8: 2}, 'validation': {1: 98, 2: 167, 3: 170, 4: 102, 5: 40, 6: 18, 7: 6}, 'test': {1: 35, 2: 127, 3: 197, 4: 190, 5: 103, 6: 38, 7: 12}}.

Ridge selected by validation RMSE, fit on training rows only. Fixed alphabet; no scaling.
Constant-prediction correlations are undefined (null), not zero.

| Model | Spearman rho | Pearson r | MAE | RMSE |
|---|---:|---:|---:|---:|
| training_mean | undefined | undefined | 0.048968 | 0.053985 |
| one_hot_ridge | 0.648837 | 0.683810 | 0.033651 | 0.042845 |
| mutation_count_ridge | 0.229543 | 0.237123 | 0.042635 | 0.053048 |

Selected alphas: {'one_hot_ridge': 1.0, 'mutation_count_ridge': 10.0}.
Skipped: {}.

Source: https://zenodo.org/records/18433203 (v3); https://flip.protein.properties/
Attribution: FLIP2, Didi et al. (2026); Enzyme structure correlates with variant effect predictability, Van der Flier et al. (2024), CSBJ 23:3489–3497; assay data MIT

See metrics.json, qc.json, manifest.json and test_predictions.csv for full evidence.
