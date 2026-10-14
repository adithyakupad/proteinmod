# M1: frozen ESM-2 on FLIP2 Alpha Amylase

Exact M0 dataset and split: {'train': 2403, 'validation': 601, 'test': 702}; lengths: {425: 3706}.
Single protein, 410 shared train/test substitutions; no cross-protein generalization.
Test-sequence extraction is label-free transductive evaluation, not test-label training.

Model revision: `6fbf070e65b0b7291e7bbcd451118c216cff79d8`.
Extraction device: mps.
M0 reproduction passed: True; maximum prediction differences: {'training_mean': 0.0, 'one_hot_ridge': 0.0, 'mutation_count_ridge': 0.0}.

| Model | Spearman rho | Pearson r | MAE | RMSE |
|---|---:|---:|---:|---:|
| training_mean | undefined | undefined | 0.048968 | 0.053985 |
| one_hot_ridge | 0.648837 | 0.683810 | 0.033651 | 0.042845 |
| mutation_count_ridge | 0.229543 | 0.237123 | 0.042635 | 0.053048 |
| esm_mean | 0.498379 | 0.515328 | 0.040200 | 0.046462 |
| esm_delta | 0.498379 | 0.515328 | 0.040200 | 0.046462 |
| esm_sites | 0.634051 | 0.687888 | 0.031407 | 0.039956 |

## Low-label results

Each cell is mean ± sample SD across 3 deterministic nested-subset seeds.
Validation/test never change. This measures training subset variation, not test uncertainty.
100% subsets coincide, so seed variance there is zero. Alphas selected by validation RMSE.

| Model | Fraction | Train n | Spearman rho | RMSE |
|---|---:|---:|---:|---:|
| esm_delta | 1% | 25 | 0.203437 ± 0.075848 | 0.053598 ± 0.001236 |
| esm_delta | 5% | 121 | 0.289625 ± 0.050063 | 0.051026 ± 0.000796 |
| esm_delta | 10% | 241 | 0.306934 ± 0.026315 | 0.050833 ± 0.001031 |
| esm_delta | 25% | 601 | 0.391279 ± 0.038829 | 0.048998 ± 0.000456 |
| esm_delta | 50% | 1202 | 0.445852 ± 0.025085 | 0.047647 ± 0.000410 |
| esm_delta | 100% | 2403 | 0.498379 ± 0.000000 | 0.046462 ± 0.000000 |
| esm_mean | 1% | 25 | 0.203437 ± 0.075848 | 0.053598 ± 0.001236 |
| esm_mean | 5% | 121 | 0.289625 ± 0.050063 | 0.051026 ± 0.000796 |
| esm_mean | 10% | 241 | 0.306934 ± 0.026315 | 0.050833 ± 0.001031 |
| esm_mean | 25% | 601 | 0.391279 ± 0.038829 | 0.048998 ± 0.000456 |
| esm_mean | 50% | 1202 | 0.445852 ± 0.025085 | 0.047647 ± 0.000410 |
| esm_mean | 100% | 2403 | 0.498379 ± 0.000000 | 0.046462 ± 0.000000 |
| esm_sites | 1% | 25 | 0.301947 ± 0.020724 | 0.052909 ± 0.003142 |
| esm_sites | 5% | 121 | 0.458044 ± 0.033698 | 0.046209 ± 0.001139 |
| esm_sites | 10% | 241 | 0.482087 ± 0.053343 | 0.045311 ± 0.001811 |
| esm_sites | 25% | 601 | 0.559550 ± 0.035076 | 0.042472 ± 0.000536 |
| esm_sites | 50% | 1202 | 0.608569 ± 0.017974 | 0.040424 ± 0.000621 |
| esm_sites | 100% | 2403 | 0.634051 ± 0.000000 | 0.039956 ± 0.000000 |
| mutation_count_ridge | 1% | 25 | 0.229543 ± 0.000000 | 0.054469 ± 0.002230 |
| mutation_count_ridge | 5% | 121 | 0.229543 ± 0.000000 | 0.052453 ± 0.000140 |
| mutation_count_ridge | 10% | 241 | 0.229543 ± 0.000000 | 0.052897 ± 0.000720 |
| mutation_count_ridge | 25% | 601 | 0.229543 ± 0.000000 | 0.052697 ± 0.000107 |
| mutation_count_ridge | 50% | 1202 | 0.229543 ± 0.000000 | 0.052757 ± 0.000132 |
| mutation_count_ridge | 100% | 2403 | 0.229543 ± 0.000000 | 0.053048 ± 0.000000 |
| one_hot_ridge | 1% | 25 | 0.210356 ± 0.025385 | 0.055495 ± 0.001535 |
| one_hot_ridge | 5% | 121 | 0.380898 ± 0.058901 | 0.051489 ± 0.001091 |
| one_hot_ridge | 10% | 241 | 0.447422 ± 0.091816 | 0.049897 ± 0.004322 |
| one_hot_ridge | 25% | 601 | 0.507369 ± 0.075976 | 0.049036 ± 0.003508 |
| one_hot_ridge | 50% | 1202 | 0.602259 ± 0.024208 | 0.044995 ± 0.000369 |
| one_hot_ridge | 100% | 2403 | 0.648837 ± 0.000000 | 0.042845 ± 0.000000 |
| training_mean | 1% | 25 | undefined | 0.055090 ± 0.002552 |
| training_mean | 5% | 121 | undefined | 0.053800 ± 0.000154 |
| training_mean | 10% | 241 | undefined | 0.053928 ± 0.000505 |
| training_mean | 25% | 601 | undefined | 0.053976 ± 0.000034 |
| training_mean | 50% | 1202 | undefined | 0.053974 ± 0.000098 |
| training_mean | 100% | 2403 | undefined | 0.053985 ± 0.000000 |

![Learning curves](learning_curves.png)

## Representation interpretation

Mean pooling averages all residue hidden states. Delta subtracts a constant WT mean; with a fitted intercept and identical ridge penalty these are mathematically equivalent up to numerical precision on this single-WT assay. Site features concatenate the mean mutant state at changed sites, mean mutant-minus-WT state at those sites, and site count. This symmetric aggregation handles multiple mutations; WT uses zeros. It can retain local mutation signal lost by whole-sequence averaging, but does not explicitly learn epistasis.

Pretraining may reduce the number of labeled examples needed by supplying biochemical structure. It can also discard assay-specific effects; no improvement is assumed. These results use one architecture and a predeclared grid, without test-driven redesign.

No GNNs, structural integration, cross-attention or fine-tuning. Frozen float32 encoder; ridge only is fitted. Fixed alphabet for one-hot, no feature scaling for any representation. See selections in learning_curve_runs.json and the source/config/environment manifest.
