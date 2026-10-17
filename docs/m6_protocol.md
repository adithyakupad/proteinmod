# M6: independent experimental stability prediction

## Prediction and data contract

Predict **folding ΔΔG in kcal/mol**, positive destabilization, from a reference sequence/structure and substitutions. No function target, stability-informed function prediction or multitask optimization enters M6. Existing M0–M5 source behavior, model parameters and benchmark artifacts remain unchanged.

Primary source: [Tsuboyama et al., Nature 2023](https://doi.org/10.1038/s41586-023-06328-6), [Zenodo record 7992926, version v2_230420](https://zenodo.org/records/7992926), CC-BY-4.0. Use the authors' **experimental `ddg_<AA>` columns** in `Data_tables_for_figs/dG_site_feature_Fig3.csv`, not its PCA components, centered stabilities, inferred functional annotations or label-derived means. The compact figure archive is 24.1 MB and the author AlphaFold archive 14.0 MB; no full 1 GB measurement archive is required. `configs/m6.json` pins both archives, the CSV member and each selected structure by checksum. These data are experimentally inferred thermodynamic quantities from cDNA-display proteolysis and a two-state model, not direct calorimetry, melting temperatures, enrichment scores or new in-silico labels.

Source ΔG describes **unfolding** stability; source ΔΔG is mutant minus WT unfolding ΔG, positive stabilization. Conversion is explicit: `canonical_ddg = -source_ddg`. Equivalently, canonical folding ΔΔG is mutant minus WT folding free energy, positive destabilization. Retain both source value/sign/unit and converted target on every row. kJ/mol conversion divides by 4.184; no other quantity is silently converted to kcal/mol. Censored/nonfinite/missing targets are masked/rejected, never replaced by zero.

The paper specifies inference at **298 K and pH 7.4**, while its extended-data discussion states measurements were in **PBS at room temperature, approximately 22°C**. Both are retained separately. Row-specific temperature, raw replicate uncertainty and label confidence intervals are unavailable in the compact figure table. They are explicitly unavailable, not inferred from structural confidence or assumed to represent independent replicate measurements. The author analysis cohort is already selected; the pilot does not reproduce its raw-data quality-control process and cannot audit all saturation/aggregation artifacts from this table alone.

Select **1LP1, 1A32, 1TUD**, three natural assay constructs of 55, 63 and 60 residues. Author-provided AlphaFold coordinates define the pinned full construct sequence; every measured WT amino acid and one-based position must match it, and every WT ΔΔG anchor must equal zero. No numbering offsets are guessed. All mutations to Cys are excluded before fitting as a conservative, label-independent proteolysis/disulfide-artifact restriction. Retain at most **600 variants per protein**, selected by sequence-identity hashes without inspecting target magnitudes. Existing source fields and exclusions are preserved in the audit.

These compact tables contain single substitutions. The sequence/graph/model interfaces support permutation-invariant mutation sets and missing mapping, with synthetic multi-substitution tests; this pilot makes no experimental multi-mutant performance claim. Repeated measurements may only aggregate within identical directed sequence pairs, assay IDs and conditions; distinct assays/conditions remain separate tasks. This runner deliberately refuses to pool nonidentical assays/conditions into one regression.

## Frozen transfer and independent head

Reuse the exact **M5 bidirectional encoder** from the fixed G2 GB1-held-out, seed-0 checkpoint. That encoder's function supervision used **amylase training** and **NucB validation**, not the new stability proteins. The selection of this checkpoint is fixed by protocol, not stability test performance. ESM-2 remains the original frozen 35M checkpoint/revision. All sequence projections, GNN weights, Q/K/V projections and normalization parameters remain **frozen**. Remove the function head from the copied encoder; train only a new continuous `stability_head`. This is controlled frozen transfer, not simultaneous tasks or new shared-encoder training.

The encoder tensor fingerprint is pinned independently of checkpoint serialization; `weights_only=True` loads trusted checksum-validated tensors. No historical checkpoint is overwritten during normal M6 preparation, training or inference. Reproduction can rebuild the fixed M5 checkpoint from its original function protocol if the ignored cache is absent.

M6 reuses M2 graph construction, reference alignment, SASA, residue properties, spatial contacts/backbone edges and missing-descriptor masks. Author models explicitly use **AlphaFold pLDDT**, not experimental B factors. Geometry remains reference geometry, not measured mutant rearrangement. Available sequence and structural streams are masked independently before attention and readout; mutations without a graph node retain sequence queries with a structural mutation-coverage indicator. Both unavailable modalities are rejected. Missing-modality outputs are engineering-tested, not experimentally calibrated by this fully observed pilot.

Readout is the unchanged M5 mutation/global representation (dimension 97), plus two modality-presence flags and one mapped-mutation fraction (100 features). The separate head is `100 → 16 → 1`, ReLU. Optimize only this head with AdamW LR .003, decay .01, clipping norm 5, batch 64, at most 100 epochs, patience 15, seeds 0/1, deterministic CPU/four threads. Standardize targets using **training mean/std only**, invert at prediction time and persist the calibration with weights. Earliest minimum validation macro per-protein RMSE selects the epoch. No rank normalization of stability labels.

## Prespecified evaluation and baselines

**S1 within-protein diagnostic:** hash-selected variants split 70/15/15 (420/90/90 per protein). **S2 protein-held-out:** three rotating test proteins; the next protein is validation and the remaining protein supplies all training examples. Entire held-out protein labels are absent from supervised training/scaling/selection. S1/S2 primary populations differ; the report also restricts S2 to S1 test IDs for a population-controlled comparison.

Reject exact sequence overlap, shared forward/reverse sequence pairs, shared PDB/model constructs and close-reference protein leakage across S2 roles. The explicit near-homology screen blocks ≥30% global-aligned identity with ≥80% coverage of **both** sequences. Save identity and both coverage fractions for all pairs, including function-pretraining proteins. This screen is not a sensitive evolutionary-family certificate; no cross-family generalization claim is warranted. An initially considered pair of related constructs (1LP1/2B88) was excluded during label-blind reference inspection, before model evaluation.

Baselines: training mean; mutation-count ridge (constant for this single-mutant pilot); transferable physicochemical sequence ridge; amino-acid-substitution one-hot ridge; frozen ESM mutation-site ridge (mutant and mutant-minus-WT); recomputed structural-feature ridge with explicit availability/coverage; frozen-hybrid-feature ridge. Additionally fit positional one-hot ridge independently within each known protein for S1 only. Choose ridge alpha from [.1,10,100] using validation macro RMSE, training-only scaling. No unrelated proteins' residue positions are treated as aligned in S2.

Report per-protein Spearman, Pearson, MAE and RMSE in native units, macro metrics for this same source/condition task, and pooled metrics separately. Undefined constant-prediction correlations remain undefined per protein and count as zero in macro correlation summaries. Two-seed SD is descriptive, not a confidence interval. No hyperparameter/architecture redesign based on test results, no clipping predictions to observed test ranges, no test-protein calibration.

## Reproduce locally

Start at repository root with Python 3.12 and the measured `requirements-m3.txt` environment:

```sh
python -m pip install -r requirements-m3.txt
python -m pip install -e '.[dev,m3,esm]'
python -m proteinmod.stability_experiments download --config configs/m6.json
python -m proteinmod.stability_experiments prepare --config configs/m6.json
python -m proteinmod.stability_experiments run --config configs/m6.json
python scripts/report_m6.py --input runs/m6 --output runs/m6
ruff check .
ruff format --check .
pytest
```

Preparation uses the same already-cached pinned ESM checkpoint as M1/M4/M5. On a fresh machine, follow `docs/m4_protocol.md` to acquire the three function datasets and frozen checkpoint, run M5 residue preparation, and then rebuild **only the fixed M5 transfer fit**:

```sh
python -m proteinmod.hybrid_experiments prepare --config configs/m5.json
python -m proteinmod.stability_experiments rebuild-transfer --config configs/m6.json
```

The rebuild checks encoder tensor equality against the pinned M5 fingerprint and fails on mismatch. It consumes function labels only. Normal M6 fitting is offline on cached frozen features. Raw archives, structures, embeddings, extracted feature arrays and fitted weights stay under ignored `data/cache/m6` (the existing transfer checkpoint stays in M5 cache). Retain text metrics/predictions, figures and provenance under a new `benchmarks/m6`; protect all M0–M5 trees. No LoRA, fine-tuning, uncertainty decomposition, assay-conditioned model, explicit epistasis, multitask training or stability-informed function prediction.
