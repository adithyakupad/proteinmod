# M8 actual stability-informed function results

G2 protein-held-out, equal protein macro, three fixed seeds. Δ compares the appropriate independent baseline; ± is paired sample SD across seeds, not a confidence interval.

| Stability source | Function model | Spearman | Δ vs baseline ± SD | Shuffled Δ | Dummy Δ |
|---|---|---:|---:|---:|---:|
| esm_sites_ridge | stability_only | 0.064873 | -0.037719 ± 0.010475 | -0.130100 | -0.114563 |
| esm_sites_ridge | feature_ridge | 0.196131 | +0.005118 ± 0.000000 | +0.000176 | -0.001631 |
| esm_sites_ridge | residual | 0.122329 | +0.019738 ± 0.021127 | -0.005654 | -0.006052 |
| esm_sites_ridge | neural_head | 0.099704 | +0.006893 ± 0.015839 | -0.024578 | +0.018639 |
| structural_ridge | stability_only | 0.210730 | +0.108139 ± 0.010475 | -0.090220 | -0.114563 |
| structural_ridge | feature_ridge | 0.192179 | +0.001166 ± 0.000000 | -0.000050 | -0.001631 |
| structural_ridge | residual | 0.166501 | +0.063909 ± 0.054027 | -0.000452 | -0.006052 |
| structural_ridge | neural_head | 0.118238 | +0.025427 ± 0.015004 | +0.002731 | +0.018639 |

Independent G2 references:

- feature_ridge: 0.191013 ± 0.000000.
- m7_function: 0.102592 ± 0.010475.
- neural_head: 0.092811 ± 0.052160.

Actual run: 35.01 s CPU, peak RSS 1473.7 MiB; 324 function model/seed/fold records and 161,433 held-out predictions. Cached ESM/representations exclude prior extraction/training costs.

Eight independently fitted M6 ridge source models (two sources × four folds). Function fits: 4 independent feature ridges, 12 independent neural heads, 216 conditioned/control ridges, 72 conditioned/control neural heads = 304; source fits bring total to 312. Twelve fixed M7 baselines are replayed, not retrained when caches exist.

## Interpretation limits

All function stability values are predicted and OOD; no paired experimental stability measurements are invented. Sources were trained on permitted M6 training proteins and selected on stability validation only. Function model selection uses function validation only. All candidates and negative controls retained. Shuffling preserves protein/role marginals; dummy matches training mean/SD, not all distributional properties.

Only three function proteins and three short stability constructs; source single-mutant training does not validate multi-mutant thermodynamics. Reference geometry and ESM pretraining limits remain. No calibrated uncertainty or mechanistic stability–function claim; no M9. See docs/m8_protocol.md and interpretation.md for design and failure analysis.
