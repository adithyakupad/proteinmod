# M7 prespecified multitask protocol

Issue #13; stacked on draft PR #12 head 1753cbb. M0–M6 artifacts and configurations are immutable. No M8 stability-as-input, assay conditioning, PLM tuning, uncertainty or epistasis extension.

## Data and endpoints

Reuse exact M5 structure-matched function cohorts (1,731 variants) and M6 experimental stability cohorts (1,800 variants), cached ESM-2 revision 6fbf070e65b0b7291e7bbcd451118c216cff79d8 and validated M2 reference graphs. Read M4/M6 protocols for provenance, licensing, construct alignment, exclusions and confidence semantics. Function assay targets are distinct: fit empirical midrank percentiles within each training assay only. Function comparisons use native-measurement Spearman per protein (ties retained), equal-protein macro; prediction is a train-rank score, not a native functional unit. Stability retains canonical folding mutant-minus-WT ΔΔG kcal/mol, positive destabilization. Train-only mean/SD normalizes each task's squared error; stability inversion restores native units. No validation/test normalization and no invented paired outcomes.

G1/S1 diagnostic uses original within-protein roles. G2/S2 primary uses original whole-protein roles. Prespecified fold pairing is amylase/1LP1, gb1/1A32, nucb/1TUD, preserving all six existing test proteins and every role ID. This economical three-pair design does not exhaust all nine combinations; auxiliary-protein composition is a limitation. Cross-task sequence/protein role collisions and near-homologs (>=30% aligned identity, >=80% coverage both ways) are rejected. Similarity is audited; unseen-family generalization is not established.

## Models and progression

Cached frozen mutant-residue ESM-2 tokens feed the unchanged M5 projected bidirectional cross-attention and compact two-layer structural GNN. Reuse M6's validated residue-to-reference mapping/readout, including explicit modality availability and mapped-mutation coverage. M7 deliberately bypasses only its no-gradient decorator, with tests for both tasks' gradients through projection, GNN and both attention directions. No ESM parameters are loaded into the training model. Inputs describe reference geometry, not mutant rearrangements. Two separate MLP heads receive the same representation; neither receives the other head's prediction or labels.

All newly compared pathways use identical fresh non-PLM initialization for a given seed. The fixed M6 pretrained encoder is unsuitable here: its amylase training and NucB validation would leak into some G2 tests. Historical M5/M6 references are replayed separately; their scores are contextual, not fair matched initialization/compute controls.

Fixed seeds 0,1,2. Width16, two attention heads, two GNN layers. Two head-only warm-up epochs, then 18 epochs with adapters/GNN/attention trainable; frozen-shared control keeps representation frozen all 20 epochs. Separate independent task models use identical initialization, progression, batch size, maximum optimizer steps and per-task exposures as joint models. Both independent models are materialized with two heads for exact initialization; unused heads receive no updates. Joint shared encoder has fewer deployed parameters than two independent encoders.

Each optimizer step draws one protein-homogeneous batch per active task. Both tasks receive the same number of batches; shorter batch lists cycle deterministically. Effective counts and steps are recorded, including repeats. Loss is the weighted mean of task-specific observed-example MSE means. Missing labels are indexed out BEFORE arithmetic; an all-missing batch raises. Finite observed losses and gradient norms are enforced, clipping5, AdamW LR.003 WD.01. Optional unavailable modalities are masked before projection/readout; real benchmark cohort has both modalities, so missing-modality performance is not calibrated.

Independent function selects minimum `(1-validation macro Spearman)/2`; independent stability selects minimum validation macro RMSE divided by stability training SD. Joint/frozen select the equal mean of those two validation criteria. All epochs run, avoiding variable training budgets from early stopping. Earliest best epoch wins. Fixed equal weighting is reported separately. Small prespecified stability:function loss-weight grid [.25,1,4] selects joint weight by the same combined validation criterion; ties choose smaller weight. Each seed/fold is selected independently without test inspection. Test labels are used only for final evaluation and failure analysis.

## Required evidence

Per-task/per-protein metrics and predictions on identical IDs, three-seed variation, paired joint-minus-independent changes, negative transfer, train/validation traces, first trainable-step gradient cosine/norms, parameters, optimizer steps, effective exposures, runtime, device, dependencies and fingerprints. Stability primary macro RMSE kcal/mol plus macro Spearman; function primary macro Spearman. Do not pool unlike function units. Report small cohorts, function assay shift confounding, M6 compact-table quality limitations and possible 1TUD upper-bound saturation. Shared improvement alone cannot establish a mechanistic stability–function relationship in unpaired proteins.

## Reproduce

Use the pinned local environment in requirements-m3.txt and the M4/M5/M6 preparation commands in their protocols. Existing raw/structure/residue caches are read and checksum-validated; no new dataset download is required for M7.

```
python -m proteinmod.multitask_experiments --config configs/m7.json
python scripts/report_m7.py --input runs/m7 --output runs/m7
ruff check .
ruff format --check .
pytest -q
```

Checkpoint cache: data/cache/m7/checkpoints/<config fingerprint>; no weights or dense embeddings committed. Run manifest records source/config/data/encoder hashes, versions, seed, CPU runtime and process peak memory. Historical benchmark checksums are checked before/after. Saved model contains schema, task order, calibrators and configuration; load only trusted checkpoints with weights_only=True.
