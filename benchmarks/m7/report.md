# M7 actual local multitask results

Primary protein-held-out results; equal protein macro; mean ± sample SD across three fixed seeds.

| Task | Model | Macro Spearman | Macro RMSE kcal/mol |
|---|---|---:|---:|
| function | independent | 0.102592 ± 0.010475 | — |
| stability | independent | 0.479711 ± 0.007915 | 1.247250 ± 0.097217 |
| function | frozen | 0.103883 ± 0.101976 | — |
| stability | frozen | 0.405503 ± 0.046137 | 1.241647 ± 0.074266 |
| function | joint | 0.098700 ± 0.108709 | — |
| stability | joint | 0.488040 ± 0.028129 | 1.250014 ± 0.102484 |
| function | selected_joint | 0.078303 ± 0.083460 | — |
| stability | selected_joint | 0.476582 ± 0.037394 | 1.218225 ± 0.128092 |

Actual runtime: 506.79 s CPU, peak process RSS 1361.3 MiB. ESM extraction reused from immutable caches, excluded from training runtime.
73134 held-out predictions (includes grid candidates and selected aliases); 72 actual fits, plus 12 selected-result aliases.

## Negative transfer

function, joint: degradation in 4/9 protein-seed cases versus matched independent training.
function, selected_joint: degradation in 6/9 protein-seed cases versus matched independent training.
stability, joint: degradation in 7/9 protein-seed cases versus matched independent training.
stability, selected_joint: degradation in 6/9 protein-seed cases versus matched independent training.

Parameter controls: encoder 12,528; each task head 1,633. Effective independent pathway 14,161 per task, two independent pathways 28,322; joint 15,794; frozen control trains only 3,266. Unused independent heads exist solely for seed-matched initialization and are excluded from effective deployed counts.

Equal-joint and the independent pair use identical per-task example exposures and encoder forward-pass budgets. Grid selection costs three joint fits per fold/seed; it is not compute-matched to one independent pair.

First trainable-step task gradient cosine range -0.2103 to 0.2450. These local diagnostics do not establish causal interference across training.

Configuration/protocol was committed before the full experiments. See docs/m7_protocol.md for exact roles, validation-only selection and native-unit handling.

## Limits

Only three proteins per task; no paired outcomes. Auxiliary fold pairing is prespecified, not exhaustive. Function protein and assay shifts remain confounded. No verified unseen-family generalization. Source stability compact table lacks raw replicate/censoring metadata; 1TUD has 60/600 targets at its observed upper bound. Geometry is reference geometry. Neither head conditions on the other task prediction; no mechanistic stability–function claim or M8 implementation.
