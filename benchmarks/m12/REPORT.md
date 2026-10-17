# M12: family-held-out domain mutation-effect benchmark

Actual locally executed predictions. Eight experimentally measured Tsuboyama 2023 domains, four externally annotated Pfam families (two proteins each), 96 label-blind sampled variants per protein. This is **stability mutation-effect transfer across small domain families**, not biochemical function transfer or unseen-fold generalization. Assays retain native processed DMS scores; models predict training-assay ranks. No pooled raw RMSE or thermodynamic calibration.

## Primary matched-cohort results

Each number averages proteins equally within each family, then four families equally, then seeds. Family SD describes the four effective replication units; it is not a confidence interval. Ridge is deterministic; neural runs use three fixed seeds.

| Model | G1 ρ | G2 ρ | G3 ρ | G3 family SD | G3 seed SD |
|---|---:|---:|---:|---:|---:|
| bidirectional | 0.6225 | 0.4932 | 0.3503 | 0.1280 | 0.0249 |
| esm_sites_ridge | 0.7040 | 0.5401 | 0.4008 | 0.0560 | 0.0000 |
| sequence_ridge | 0.3368 | 0.3057 | 0.3201 | 0.1274 | 0.0000 |
| structural_ridge | 0.5533 | 0.3996 | 0.4298 | 0.0723 | 0.0000 |
| structure_only | 0.4387 | 0.3983 | 0.3970 | 0.1238 | 0.0143 |
| weighted_fusion | 0.6375 | 0.5281 | 0.3914 | 0.1279 | 0.0437 |

G1 evaluates 16 variants/protein; G2/G3 evaluate all 96 variants of their held-out proteins. G2/G3 retain all original within-protein partition tags but assign complete proteins/families to roles. Sequence-all and structure-matched happen to coincide here: all 768 variants have usable mutation sites. Their sequence baseline predictions must agree exactly.

## Generalization gaps on identical test variants

Secondary diagnostic restricts all protocols to the same original G1 test IDs (16 per protein), using already-saved predictions without refitting or selection. Positive gaps mean worse held-out generalization.

| Model | Common G1 ρ | Common G2 ρ | Common G3 ρ | G1−G2 | G2−G3 |
|---|---:|---:|---:|---:|---:|
| bidirectional | 0.6225 | 0.4049 | 0.2315 | 0.2177 | 0.1734 |
| esm_sites_ridge | 0.7040 | 0.4140 | 0.2217 | 0.2901 | 0.1923 |
| sequence_ridge | 0.3368 | 0.2426 | 0.2147 | 0.0941 | 0.0279 |
| structural_ridge | 0.5533 | 0.3754 | 0.2816 | 0.1779 | 0.0938 |
| structure_only | 0.4387 | 0.3635 | 0.3011 | 0.0752 | 0.0624 |
| weighted_fusion | 0.6375 | 0.4490 | 0.2922 | 0.1885 | 0.1569 |

## Per-family G3 results

| Model | PF00018 | PF00385 | PF00046 | PF00397 |
|---|---:|---:|---:|---:|
| bidirectional | 0.1692 ± 0.0501 | 0.4478 ± 0.0317 | 0.2930 ± 0.0367 | 0.4914 ± 0.0456 |
| esm_sites_ridge | 0.4442 ± 0.0000 | 0.4545 ± 0.0000 | 0.3130 ± 0.0000 | 0.3915 ± 0.0000 |
| sequence_ridge | 0.1963 ± 0.0000 | 0.5318 ± 0.0000 | 0.2544 ± 0.0000 | 0.2979 ± 0.0000 |
| structural_ridge | 0.3591 ± 0.0000 | 0.5460 ± 0.0000 | 0.4331 ± 0.0000 | 0.3808 ± 0.0000 |
| structure_only | 0.2432 ± 0.0364 | 0.5864 ± 0.0520 | 0.3552 ± 0.0298 | 0.4033 ± 0.1025 |
| weighted_fusion | 0.2261 ± 0.0886 | 0.4958 ± 0.0463 | 0.3092 ± 0.0796 | 0.5345 ± 0.0160 |

## Audits and compute

182 actual fits, 23296 held-out predictions; summed fit time 107.2 s. Exact versions, peak process RSS and full wall time are in environment.json. Feature extraction/device times and tensor hashes are in audit.json; large frozen tokens and model checkpoints remain ignored local caches.

Preserved 279 M0–M11 benchmark/configuration file hashes. No original partition or historical result was overwritten.

All G2 roles are protein-disjoint with test/validation families represented in training. All G3 roles are family-disjoint. Global/local identity and aligned-block audits, ambiguity flags, and 40/60/80% coverage-filtered sequence clusters are retained. No unresolved cross-family cluster/partial-domain match passed the declared screening rules. External family definitions are Pfam 34.0 domain features on the deposited polymer entities, not whole-protein annotation names. Full-length annotations may describe additional domains absent from these assay constructs.

Graphs are reference geometry only. Mapping/construct insertions, connectivity, missing descriptors and removal masks are recorded per protein. PITX2 selects chain P; SAV1 uses the declared exact reference polymer span 7–49 to resolve a cloning-tag alignment ambiguity. Seven structures are NMR and BBC1/1TG0 is crystallographic. All confidence descriptors were conservatively masked in this fixed experiment; B columns are not interpreted as pLDDT or model uncertainty. See ERRATA.md for the preregistered BBC1 display-name typo. Mutation masks and substituted features follow the M3/M5 tested interfaces.

Target midranks and feature scaling use training only. Ridge alpha and neural epochs use validation only. Balanced validation protein counts give equal family weights. Every G3 encoder is randomly initialized; no M0–M11 supervised representation is transferred. ESM-2 is frozen at an immutable revision; unsupervised pretraining exposure and structure-database exposure cannot be excluded.

## Interpretation and failure analysis

Transfer is model- and family-dependent; compare the per-family table before interpreting the macro average. Negative correlations are retained. The four family outcomes are descriptive evidence, not a population-wide family-transfer estimate. Seed variation measures optimization sensitivity, not additional independent families.

G1 has fewer training variants per protein than G2/G3 have per training protein; G3 additionally trains on only two families. Thus gaps mix reduced family diversity, training-set size and task difficulty. The common-test diagnostic removes test-genotype composition differences, not these training differences. With one assay per construct, protein/family and construct-specific assay effects remain inseparable. Shared study methodology reduces but cannot eliminate that confound. The cap, short domains and 12-epoch budget favor speed and are not a performance ceiling.

Validation traces and training correlations in runs.json expose early stopping and overfitting; no larger search was launched after inspecting tests. Constant/undefined correlations are explicitly null per protein and zero only for macro selection/aggregation. No CI is reported from four families and no mutation-level significance claim is made. Additional independent families, assay endpoints and paired homolog measurements are needed before broad functional or unseen-fold claims.

## Reproduction

See DATA_PROVENANCE.md and docs/M12.md for pinned acquisition, offline verification, tests and exact commands. The preregistration/config were committed before fitting. No M13 work was added.

![Family-balanced results](family_results.png)
