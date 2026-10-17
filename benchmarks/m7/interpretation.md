# M7 scientific interpretation and failure analysis

## Outcome

The pilot does **not** establish a consistent benefit from stability–function multitask learning. All three seeds and prespecified loss weights were retained. No test outcome changed the cohort, architecture, weights or training schedule.

On held-out proteins, equal-weight joint versus matched independent training changes function macro Spearman by **−0.003892 ± 0.098843**, and stability macro RMSE by **+0.002763 ± 0.026875 kcal/mol** (paired seed mean ± sample SD; negative RMSE change would be better). Validation-selected weighting changes function by **−0.024289 ± 0.074490** and stability RMSE by **−0.029025 ± 0.088115 kcal/mol**. The mean shifts are smaller than seed variation; these are descriptive three-seed spreads, not confidence intervals across proteins. Stability's selected mean improvement is driven by seed0 (−0.130737), while seeds1/2 worsen (+0.024111/+0.019551). Reporting only the mean would conceal this instability.

The fixed equal model degrades stability RMSE in **7/9** protein-seed comparisons and function Spearman in **4/9**. Selected weighting degrades each task in **6/9**. There is no reliable Pareto improvement. The selected model is the result of a prespecified combined validation criterion, not the best test-performing weight.

## Where transfer fails

Mean held-out function Spearman (independent → equal joint → validation-selected):

| Protein | Independent | Equal joint | Selected joint |
|---|---:|---:|---:|
| Amylase | .173507 | .078199 | .046445 |
| GB1 | .081093 | .173867 | .120473 |
| NucB | .053175 | .044033 | .067991 |

Mean held-out stability RMSE kcal/mol:

| Protein | Independent | Equal joint | Selected joint |
|---|---:|---:|---:|
| 1LP1 | 1.145249 | 1.115890 | 1.029239 |
| 1A32 | .536781 | .551531 | .550782 |
| 1TUD | 2.059721 | 2.082619 | 2.074655 |

1LP1 accounts for the aggregate selected stability improvement, while 1A32 and 1TUD worsen. Stability ranking also changes independently of calibration: independent macro Spearman .479711, equal .488040, selected .476582. Selection by RMSE cannot be presented as improved ranking.

On known proteins, independent function Spearman is **.592716 ± .044870**, equal joint **.598420 ± .051405**; independent stability RMSE is **.562782 ± .009699**, equal joint **.587234 ± .011073**. This small function difference accompanies worse stability, even in the easier within-protein setting. The frozen random shared representation is much weaker within proteins (.363336 function Spearman; .863065 stability RMSE), supporting the value of learning the pathway in that setting, rather than establishing joint-training benefit.

Two of nine selected held-out models choose epoch1 in the head-only phase (amylase/1LP1 seeds1/2). These choices do not demonstrate improvement from shared encoder updates. The remaining selected epochs are 3,10,6,15,3,7,4; `fit_diagnostics.csv` and validation curves retain all traces and train metrics. The whole 20-epoch budget is executed even when validation selects an early checkpoint. Poor validation transfer can select underfit early models; test labels were not used to fix this.

The first trainable-step shared task gradient cosines span **−.2103 to +.2450**. Both losses reach the common projection, GNN and attention parameters, as tested explicitly. Opposing gradients are compatible with interference but are not a causal explanation for all negative transfer. Task endpoint differences, disjoint proteins, assay shift, small validation cohorts and calibration shift remain competing explanations. Units are normalized using train-only scales; batch counts are balanced, but equal normalized loss does not imply equal gradient magnitudes. Effective examples, repeats, norms and clipping diagnostics are recorded.

## Design validity and limits

M4/M5's 1,731 mapped function variants and M6's 1,800 stability variants retain every train/validation/test ID. The 69 structurally invalid function variants remain excluded from every matched model, with IDs archived. This experiment does not estimate performance on those exclusions or missing-modality cases. There are no paired endpoints; missing-task labels are never filled or inferred. Each task uses its own head and target calibration. No head consumes the other head's predictions. Geometry describes WT reference coordinates, not measured mutant rearrangements.

The three prespecified auxiliary fold pairs preserve G2/S2 protocols without testing all nine possible pairings. Function protein and assay shifts are confounded. Three proteins per task and three seeds cannot establish broad or unseen-family generalization. Near-homolog reference screening and exact sequence/protein role checks prevent the specified leakage routes but do not certify remote-family separation. ESM pretraining exposure to test proteins is unknown.

M6 source limitations remain: experimental proteolysis-inferred folding ΔΔG, kcal/mol, positive destabilization; source unfolding sign explicitly inverted; compact figure table lacks raw replicate/error/censoring flags. 1TUD has 60/600 values at its observed upper bound, so apparent absolute errors may involve saturation/quality limits. No rows were removed after seeing this. Author assay constructs, alignment, pLDDT semantics and raw-source checksums remain those of M6. No author label-derived structural/PCA features enter M7.

The historical fixed M6 encoder was function-supervised on amylase and validated on NucB. Reusing it for every G2 fold would leak test/validation function labels. New comparisons therefore start fresh non-PLM weights with matched seeds; frozen ESM tokens remain identical. M5/M6 references were independently replayed exactly (3,986 and 4,140 predictions); their different transfer/training protocols are contextual references, not matched multitask controls. Historical results were not rewritten.

## Reproducibility and cost

The experiment implementation and complete protocol were committed at **62b9a48** before running the 72 fits. The archived run manifest records that revision, source/config hashes, dataset/cache manifests, versions and CPU. The run took **506.79 s**, peak RSS **1,361.3 MiB**, using four CPU threads on arm64. ESM extraction is reused and excluded; times are descriptive local measurements, not isolated hardware benchmarks. A separate repeat of the first held-out equal-joint seed0 fit reproduces **578 function + 600 stability predictions exactly**, plus selected epoch, entire training trace and persisted tensors/calibrators. Cross-device bitwise identity and a second full 72-fit run were not tested.

Each independent effective pathway has 14,161 parameters; two total 28,322. Joint has 15,794 (12,528 encoder + two 1,633 heads). Independent and equal-joint task exposures and encoder forward-pass budgets match; the selected grid costs three joint fits per fold/seed. Large residue arrays and checkpoints stay ignored. All **156 M0–M6 benchmark files** remain byte-identical.

No M8 stability-conditioned function prediction, assay-aware conditioning, PLM tuning, uncertainty model, generative design or epistasis extension was implemented.
