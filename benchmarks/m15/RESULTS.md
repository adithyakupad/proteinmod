# M15 measured local results

Protein-macro Spearman; equal family weights (two proteins per family).
Neural mean ± population seed SD; deterministic ridges use one fit. No pooled raw RMSE.

## Original training-label fraction 0.25

| Model | G1 | G2 | G3 |
|---|---:|---:|---:|
| frozen_ridge | 0.4941 | 0.4070 | 0.3307 |
| frozen_head | 0.3906 ± 0.0301 | 0.2870 ± 0.0357 | 0.2147 ± 0.0473 |
| matched_head | 0.3907 ± 0.0343 | 0.2821 ± 0.0299 | 0.2165 ± 0.0490 |
| lora | 0.3951 ± 0.0363 | 0.2874 ± 0.0363 | 0.2142 ± 0.0472 |
| structural_ridge | 0.3735 | 0.3306 | 0.3266 |
| original_gnn | 0.1469 ± 0.1246 | 0.1905 ± 0.1059 | 0.2130 ± 0.1453 |
| edge_full | 0.2597 ± 0.0497 | 0.2483 ± 0.0817 | 0.1637 ± 0.1955 |

## Original training-label fraction 1

| Model | G1 | G2 | G3 |
|---|---:|---:|---:|
| frozen_ridge | 0.7040 | 0.5401 | 0.4008 |
| frozen_head | 0.6626 ± 0.0458 | 0.4461 ± 0.0143 | 0.3401 ± 0.0207 |
| matched_head | 0.6746 ± 0.0321 | 0.4496 ± 0.0123 | 0.3317 ± 0.0156 |
| lora | 0.6297 ± 0.0443 | 0.4460 ± 0.0126 | 0.3426 ± 0.0197 |
| structural_ridge | 0.5533 | 0.3996 | 0.4298 |
| original_gnn | 0.4170 ± 0.0142 | 0.4014 ± 0.0186 | 0.4020 ± 0.0104 |
| edge_full | 0.4440 ± 0.0127 | 0.3991 ± 0.0091 | 0.3799 ± 0.0065 |

All models use identical test IDs within each protocol/fraction. G1 has 128 test
variants; G2/G3 each cover all 768. Low-label selection affects original training
rows only. G1/G2/G3 gaps mix novelty, cohort composition and training diversity.
Four supervised held-out families do not establish unseen-pretraining-family transfer.
