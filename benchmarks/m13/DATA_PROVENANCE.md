# M13 provenance, units and eligibility

M13 acquires no new assay collection or language model. It reuses checksum-pinned M12 and M6 configs; every M0–M12 benchmark/configuration byte is preserved. M12 remains the eight-domain, four-Pfam-family cohort described in [M12 provenance](../m12/DATA_PROVENANCE.md). Its complete original DMS targets/test partitions remain in m12_rows.json and predictions.csv.

## Primary physical-label source

[Tsuboyama et al., Nature 620, 434–444 (2023)](https://doi.org/10.1038/s41586-023-06328-6), [Zenodo v2_230420](https://zenodo.org/records/7992926), CC-BY-4.0. Source experimental method: cDNA-display proteolysis with two-state thermodynamic inference. These are experimental estimates, not simulated labels.

Archive: https://zenodo.org/records/7992926/files/Data_tables_for_figs.zip

Archive SHA256: `69f6d5a68ba961759879ab60b3f168b8fad1bce0b0b810ac490fe85001515890`.

Member: `Data_tables_for_figs/dG_site_feature_Fig3.csv`; SHA256 `79ff1e8676c50e9cdfefd1976a0bdcbbefdd42cc4c7f354fd5d146165f5a04aa`.

Only pdb_name, pos, wt_aa and ddg_<AA> labels are used by the native-label adapter. Author K50, PCA, mean-zero labels, fitted residual features, burial/contact/SASA and other outcome-derived columns do not become model inputs. Features are independently computed from sequences and checksum-verified M2 graphs.

Every measured WT site must match the independent assay reference; its source ddg_WT anchor must be zero. Canonical folding ΔΔG = ΔG_fold(mut) − ΔG_fold(WT), kcal/mol, positive destabilizing. Source ddg is positive stabilizing and is negated exactly. Native values, original source values, source row and amino-acid column are retained in native_labels.json. No kcal/kJ ambiguity or DMS-rank reinterpretation is allowed.

The primary Figure 3 release and ProteinGym processed M12 release can give slightly different values for the same experimental genotype. The crosswalk is a secondary physical prediction task, not evidence of independent replicated measurement noise. It does not replace M12 labels, and it does not supply fabricated labels for doubles or unrelated functional assays.

## Crosswalk eligibility

| Protein | Native labels / 96 |
|---|---:|
| BBC1 | 51 |
| CBX4 | 43 |
| MYO3 | 23 |
| NKX31 | 54 |
| PIN1 | 39 |
| PITX2 | 48 |
| SAV1 | 39 |
| SR43C | 50 |

**347 eligible / 421 excluded**. Exclusion counts: M6_prespecified_Cys_exclusion: 14, missing_censored_primary_single_label: 77, multiple_mutation_not_in_primary_single_table: 330.

Calibrators use only withheld calibration labels; native affine predictors use only training labels. The 347 eligible observations are all singles under the existing M6 Cys/missing/censoring policy, fixed before fitting. Original single/double ordinal diagnostics retain all 768 variants. Native calibration/coverage is conditional on this availability cohort, not all possible variants.

Compatible primary assay metadata: pH 7.4, PBS, measurement temperature approximately 22 °C, inference temperature 298 K (as recorded by M6). Values from other assay units/conditions are not pooled. Measurement replicate variance and calibrated experimental-noise intervals are unavailable in this consumed table; no measured aleatoric decomposition is claimed.

## Structures, frozen representations and OOD

All M12 test mutation sites have compatible coordinates, so sequence/structural/multimodal test IDs are identical. Seven structures are NMR; BBC1 is crystallographic (see M12 ERRATA). Confidence fields were masked as unknown in that fixed experiment. The separate M6 control uses its existing author AlphaFold structures and pLDDT semantics. pLDDT/B-factors, graph completeness and reference geometry are not prediction-error probabilities. No mutant rearrangements or new structures are inferred.

Frozen ESM-2 is facebook/esm2_t12_35M_UR50D at revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`. Cache manifests bind ordered sequence IDs, WT references, extraction code, model revision, devices and tensor hashes. Unknown language-model pretraining exposure prevents claiming these families were never seen by ESM. G3 refers to supervised family exclusion.

OOD feature distance uses only training-fitted standardization and training neighbors. Global/local sequence/domain audits remain in M12; nearest training reference identity is saved per test example. Fixed-bin OOD summaries, mutation-burden results, eligibility masks and structural coverage accompany the interval/risk evidence. Complete coverage here gives no evidence for robustness to absent structures.

## Execution and corrections

Preregistration commit: `32e2c6c`; no held-out-driven grid/seed/epoch search. Each complete run performs 1,764 individual fits (ridge alpha candidates/bootstrap members and neural seed models), yielding 180 uncertainty evaluations and 32,388 held-out records. The verifier replays 1,560 retained point models; unselected ridge candidates are documented by validation scores.

An initial report import occurred before the reporting module had been written; all fits had completed successfully and raw outputs were retained. The report was regenerated without fitting. A native shuffle-control correction then preserved native eligibility within protein and role, so ineligible doubles could not alter the marginal SD distribution in native single-mutant controls. The complete fixed experiment was rerun. control_audit.json preserves the preliminary control comparison and verifies every point/member/non-shuffle-bound record was unchanged. This is a control-correctness fix, not test-guided model tuning.

The M6 control loader was simplified to omit unused historical hybrid features/transfer checkpoints. native_control_feature_check.json verifies its three used arrays are identical to the original M6 arrays; full saved-model replay confirms predictions are unchanged. Historical evidence is not overwritten.

Configuration, calibration roles, seeds, software versions, device, wall time, CPU RSS and source-code patches are recorded. Large raw archives, frozen arrays and weights remain ignored caches; committed evidence contains measured/predicted values, bounds, metrics and small figures. See [reproduction](../../docs/M13.md).
