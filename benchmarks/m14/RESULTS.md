# M14 actual local results

Family cohort: macro per-protein Spearman (two proteins per family, equal family weight).
Native cohort: macro per-protein RMSE in kcal/mol, positive destabilizing.
Mean ± population SD across three fixed seeds; ridges deterministic (one fit).

| Model | G1 rho | G2 rho | G3 rho | S1 RMSE | S2 RMSE |
|---|---:|---:|---:|---:|---:|
| structural_ridge | 0.5533 | 0.3996 | 0.4298 | 0.7728 | 1.3503 |
| original_gnn | 0.4170 ± 0.0142 | 0.4014 ± 0.0186 | 0.4020 ± 0.0104 | 0.6779 ± 0.0031 | 1.2521 ± 0.0322 |
| edge_full | 0.4440 ± 0.0127 | 0.3991 ± 0.0091 | 0.3799 ± 0.0065 | 0.6917 ± 0.0135 | 1.2237 ± 0.0534 |
| edge_distance | 0.4322 ± 0.0179 | 0.3898 ± 0.0026 | 0.3874 ± 0.0111 | 0.7181 ± 0.0366 | 1.2281 ± 0.0327 |
| edge_no_chemistry | 0.4284 ± 0.0069 | 0.3935 ± 0.0048 | 0.3798 ± 0.0142 | 0.7048 ± 0.0192 | 1.2223 ± 0.0476 |
| edge_shuffled_chemistry | 0.4110 ± 0.0219 | 0.4001 ± 0.0146 | 0.3792 ± 0.0084 | 0.7000 ± 0.0133 | 1.2175 ± 0.0365 |
| edge_shuffled_topology | 0.3991 ± 0.0214 | 0.3959 ± 0.0108 | 0.3746 ± 0.0094 | 0.7524 ± 0.0170 | 1.2233 ± 0.0346 |
| zero_message | 0.3990 ± 0.0180 | 0.3911 ± 0.0100 | 0.3874 ± 0.0163 | 0.7649 ± 0.0247 | 1.2157 ± 0.0605 |
| proteinmpnn_ridge | 0.5662 | 0.4538 | 0.4147 | 0.6373 | 1.2166 |

All models share exact test IDs within each protocol. G1 has 128 primary and
270 native test variants; G2/G3 have 768 primary variants; S2 has 1,800 native.
G1-to-G2/G3 gaps include the change from a within-protein test subset to all variants.
No cross-assay RMSE on rank predictions, no cross-family-pretraining claim.
See per_family.csv, per_protein.csv, failure_slices.csv, overfitting.csv and costs.json.
