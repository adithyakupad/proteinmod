# ProteinMod: prospective validation design for benign human proteins

**Status: metadata feasibility assessment and draft statistical-analysis plan; NOT an admitted cohort, committed preregistration, or authorization to access outcomes.**

Prepared 2026-10-20, America/New_York. Repository inspected at archive/m203-interrupted, commit 808441a80488cb4a633bc4329a096929df5cd521. Local main is 402bd8815dcd8857d416f47b1b41fdb922cf471d. No repository file or branch was changed. This document is stored outside the repository. PRs #55/#57/#59 were inspected during the preceding repository review and were open, draft and unmerged at that inspection; they were not altered or re-polled here.

## 1. Scientific question and historical interpretation

Question: do already frozen sequence and structural predictors retain assay-local ranking ability on human proteins absent from supervised development, and does the historical unscaled-versus-scaled ESM ridge contrast transfer?

Scope excludes pathogen proteins, antibiotic-resistance proteins, HIV Env, new variant generation, protein optimization and clinical prediction. Human disease association does not imply a pathogen protein. Only published abundance-assay metadata are considered. No new models, label acquisitions or evaluations are authorized by this plan.

The historical review covered milestone protocols, integration summaries and admission reports, rather than rerunning experiments or opening prediction tables:

- M0–M3 establish the original single-protein baseline, frozen representation work, reference mapping and compact structural model. Single-protein evidence cannot establish unseen-family transfer.
- M4–M11 address protein-held-out comparisons, fusion, multitask/metadata inputs and explicit interactions. Their assay, protein and sampling shifts constrain causal interpretation. Changing capacity together with input information is not a clean information ablation.
- M12–M15 introduce family-held-out development, uncertainty, richer edge information and adaptation. Reusing these cohorts does not create new independent evidence. Pretraining overlap remains different from supervised overlap.
- M16 integration/reproduction qualifications distinguish cached reproduction from full cold-source reproduction and retain release holds. Historical statements about source licensing must be read at their original dates, not treated as current legal status.
- M17's full-label unscaled-minus-scaled ESM ridge macro Spearman difference is +0.140441 on four development families. This is a development observation, not independent confirmation. Original preregistration uses a 0.02 effect margin and at least 3/4 directional families for applicable descriptive intervention claims.
- M18–M18.5 preserve independent-admission blockers, explicit mapping masks, item-specific rights and the SOD1 aggregate-metadata exposure. Later qualification does not retroactively convert failed earlier gates into passes.
- M19's expanded regularization failed against historical unscaled ridge, and its original scalar-equivalence numerical tolerance failed. Neither result is repaired or selected away here.
- M20 and M20.1/M20.2 retain no-go external confirmation: CspA/CI2's processed-score export/censoring lineage is unresolved. That route is stopped. M20.3 is an interrupted archive, not a completed scientific result; no archived source investigation is resumed.

Local anchors: docs/m1_protocol.md through docs/m11_protocol.md; docs/M12.md–M17.md; benchmarks/m17/PREREGISTRATION.md; benchmarks/m18*/REPORT.md; benchmarks/m19/REPORT.md; benchmarks/m201/REPORT.md; benchmarks/m202/REPORT.md. M20's report was read from codex/m20-blinded-scientific-closeout:benchmarks/m20/REPORT.md without checking out or merging that branch.

## 2. Bounded candidate inventory and feasibility matrix

PASS means documented at the stated scope; PARTIAL means a specific verification remains; FAIL means a required condition is not met. Article licenses, software licenses and data-item rights are separate. No row is fully admitted. No file checksum is inferred from a DOI or from a publisher's metadata checksum.

| Candidate / intended endpoint | Published source and exact-item lead | Rights evidence | Methods / selection provenance | WT / structure / evolutionary independence | Feasibility |
| --- | --- | --- | --- | --- | --- |
| Human GCK; abundance only | Gersing et al., Genome Biology (2024), DOI 10.1186/s13059-024-03238-2; Figshare 26707496, Additional file 2; MaveDB experiment 00000096-b | Deposit landing page explicitly displays CC BY + CC0. PASS for observing a stated grant; PARTIAL for exact file, version and grant scope. MIT plotting code is not a data license. | Primary paper links supplementary data, sequencing record and abundance deposit. PARTIAL: exact score column, replicate aggregation, export producer, read-support filters and censoring must be traced without reading labels. Do not substitute the separate activity dataset. | Exact construct/isoform and structure mapping not verified. Hexokinase annotation is a metadata lead, not a homology certificate. | PARTIAL; strongest directly inspected supplementary-rights lead. |
| Human ASPA; abundance only | Grønbæk-Thygesen et al., Nature Communications (2024), DOI 10.1038/s41467-024-48481-0; source repository archived under DOI 10.5281/zenodo.8382504; MaveDB experiment 00000657-a | Article states CC BY 4.0 with third-party exceptions. Exact archived data-member license not verified; Zenodo landing page was inaccessible. Do not inherit article rights automatically. | Publication links original experiment and source/export repository. PARTIAL: select abundance, exclude toxicity and predicted stability columns; establish filtering and complete export lineage. | PDB 2O53 is a published structure lead, not a verified mapping. Construct, oligomeric state, cofactors and family overlap remain unresolved. | PARTIAL; no source payload opened. |
| Human PRKN/Parkin; abundance only | Clausen et al., Nature Communications (2024), DOI 10.1038/s41467-024-45829-4; source archive DOI 10.5281/zenodo.8009574; sequencing record GSE254618 | Article CC BY 4.0; data archive grant not independently verified because landing-page access failed. | Primary publication identifies cellular abundance assay and source repository. PARTIAL: pin exact output, score direction, filtering, construct and export path. | Multidomain architecture is a specific concern: its ubiquitin-like domain requires review against historically used ubiquitin. Whole-protein novelty cannot waive domain overlap. Exact structure not selected. | PARTIAL / high risk of family-independence failure; reject if inherited gates fail. |
| Human SOD1; abundance only | Existing M18.5 record; exact MaveDB score set urn:mavedb:00001217-a-3 | Inherited PASS: specific item CC0 recorded in M18.5; not freshly revalidated here. | Inherited named-column lineage and reference records exist. Initial aggregate decoding is an immutable deviation; owner acceptance permits consideration, not pristine confirmation. | Inherited mature-reference 2C9V mapping and PF00080 checks; strict full-reference failure retained. Independent provenance review still required for the new study. | PARTIAL: conditional, deviation-disclosed prospective candidate only. |

The bounded slate contains four candidates, not four qualified evolutionary families or four independent laboratories. Shared investigators/platforms among several candidates imply correlated measurement biases. Stop after adjudicating this slate; failed candidates do not trigger an unlimited replacement search. A smaller study requires a new registered scope and cannot be presented as satisfying a four-family rule.

Rights comparison: TPMT's original publication describes conditional academic/nonprofit/commercial terms; it is not silently promoted to an unrestricted substitute. PTEN Figshare 16815903 is another published deposit with a displayed CC BY + CC0 notice, but is not an admitted reserve or an instruction to access its score table. Additional Tsuboyama measurements cannot establish a new measurement-generation study because historical development already used that source. None of these alternatives is adopted here.

## 3. Metadata-only admission contract

Before any later execution proposal, an independent custodian must provide a bounded metadata packet for each candidate:

1. Immutable publication and deposit version, exact file/member name, file size and publisher-declared digest if available; data/software/article license text, third-party exceptions, attribution and redistribution scope. Label-file SHA256 is deferred until an authorized custodian can hash opaque bytes without exposing outcomes. No files are acquired in this task.
2. Named assay endpoint, experimental context, score definition and favorable direction derived from Methods, not observed distributions. Abundance is an experimental proxy, not folding free energy, enzymatic activity, clinical pathogenicity or protein fitness in general.
3. Full processing graph from experimental counts to released endpoint: normalization, replicate combination, barcode aggregation, pseudocounts, missingness, coverage filters, censoring and any outcome-dependent selection. A README or consumer notebook alone does not prove export provenance. Unresolved mandatory transformations mean FAIL.
4. WT accession/isoform and actual experimental construct, fusion tags, numbering and structure chain/assembly/state. Require identity-compatible mapping at eligible sites under inherited mapping rules. Retain unresolved residues and masks; do not change cutoffs to rescue a candidate. No variant identities are generated or selected here.
5. Full-reference and domain-level independence against every historically supervised protein, development/validation set and prior external cohort. Use the existing numerical similarity thresholds plus family/profile/fold review; short alignments and ambiguous optima are disclosed. Count domains, orthologs and repeated constructs conservatively. Pretraining exposure cannot be ruled out.
6. A written disposition for prior exposure and all ambiguities. Require four admitted independent families before a four-family study can proceed. The present slate does not meet this gate.

Any later metadata tool must whitelist fields before semantic decoding and exclude distributions, clinical classifications, score-based counts, selection flags and example rows. Do not call whole score-set JSON endpoints merely to obtain licenses. Prefer license-only/deposit metadata, separately published Methods and source-code excerpts containing transformations; never execute author code. If a provider cannot separate permitted metadata, record a blocker.

## 4. Prospective statistical-analysis plan — design only

### Estimand and model freeze

Primary estimand: the mean across four admitted families of the paired difference in within-assay Spearman correlation, historical unscaled ESM ridge minus historical scaled ESM ridge. This is a transportability test of fixed supervised development models, not retraining on external families.

Before any future outcomes, register exact checkpoints, development fold identities, feature definitions, encoder revisions, software/environment hashes, scalers, alpha choices and training roles. Use all four original full-label M17 G3 development folds as separate fixed models: compute each fold's external correlation, then average folds within family. This is a new explicitly declared aggregation, not an ensemble fit and not an exact replication of M17's original test sampling. Repeated deterministic seeds do not add independent observations. Do not select the best fold, average externally rank-normalized predictions, or tune any model on external data. If archived checkpoints cannot be identified and verified, STOP; this task does not authorize reconstruction or fitting.

Secondary descriptive comparisons use only available, originally declared frozen structural ridge, structural neural and representation/control arms with matched historical roles. No M19 expanded-grid winner, new architecture or new search replaces the historical comparator.

### Cohort and exclusions

Propose one abundance endpoint and one qualifying construct per independent family. Use all eligible published single-amino-acid substitutions where available, restricted to canonical residues with unambiguous experimental numbering; exclude WT, synonymous, nonsense, indels and compound alleles by genotype definition. No new variants are generated. Deduplicate aliases through a source-defined deterministic rule fixed before outcomes; do not choose a replicate by its score or error bar.

A later authorized custodian would supply genotype-only identities; the exact cohort and hashes must be registered before outcome access. Sequence-all and common structure-mapped cohorts are separate estimands. Compare all model arms on identical IDs within the latter. Structural coverage exclusions depend only on reference/genotype mapping, not score magnitudes, clinical categories, measured uncertainty or retained/passing outcome flags.

Missing/nonfinite released labels would be excluded identically for every arm, with denominator and reason reported and no replacements. This administrative rule cannot undo upstream score-dependent censoring. If a family's evaluable labels are constant, Spearman is undefined; retain the family in the planned denominator and declare the primary endpoint non-estimable rather than dropping it or inventing zero. No label-dependent minimum range rule is allowed. Minimum genotype/position coverage must be fixed before outcome access; adequacy is unresolved here, so this is not an execution-ready registration.

### Aggregation, criteria and multiplicity

Within each fixed model fold, calculate Spearman with average ranks for ties separately for each construct/assay. Average constructs equally within family if a later protocol explicitly permits multiple constructs, and average model folds within family. The primary contrast is paired by family and fold. Equal-family macro aggregation prevents large datasets from dominating. Never pool native-unit RMSE, normalize labels across families or call abundance units kcal/mol.

Preserve the historical descriptive effect criterion: macro contrast at least +0.02 rho and positive differences in at least 3/4 admitted families. Describe satisfying it as consistency with the specified transfer prediction, not a significant population claim. A mirrored negative margin/direction criterion may describe evidence against that transfer prediction; intermediate, mixed or non-estimable findings are inconclusive. These new decision labels do not rewrite the original M17 hypotheses.

One primary contrast; all other comparisons are secondary descriptive analyses. Report every declared arm and family, including negative and undefined results. No best-subgroup headline, post-hoc endpoint switch or best-seed selection. No confirmatory hypothesis p-values are planned with four convenience-sampled families. Under an idealized independent sign-null, even four concordant signs yield a one-sided probability of 1/16; shared study effects weaken that interpretation further.

### Uncertainty and controls

Report four family effects, range, SD and leave-one-family-out sensitivity. Four families do not justify a reliable family-population bootstrap interval. For within-family sensitivity only, propose 2,000 paired bootstrap resamples of residue positions, retaining all substitutions at a sampled position and the same draws for every model arm. Prespecify seed 21. Report percentile intervals as conditional sensitivity intervals, not experimental replicate uncertainty, since spatial/site correlations and released-score estimation remain imperfectly represented. Recompute all ranks in each resample and report undefined draws; do not use bootstrap significance to change candidate selection.

Assay replicate uncertainty is only propagatable if raw-replicate rights, processing and independence assumptions pass a separate gate; no score-SE weighting or sigma-as-physical-noise assumption. Report finite-panel descriptive inference when no defensible population interval is available.

Design-stage controls: constant and WT-only/no-site-signal predictions should give undefined correlations, explicitly reported. Existing archived shuffled-training and graph-null checkpoints may be compared only if their identities, effective capacity and training budgets were already fixed; none is trained now. A later authorized evaluation could use a prespecified label-permutation diagnostic within each assay, with pairings and dependence caveats registered beforehand; it is not a new model or proof of assay quality. Predictions, selection and preprocessing must be invariant to held-out label changes before unblinding.

### What the study can identify

| Historical question | Identifiable under a fixed, external-only study? |
| --- | --- |
| H1 sample complexity | PARTIAL at most: only transport of contrasts between already archived nested-budget models, if all are available and registered; no external training/learning curve or external sample complexity. Not part of the primary full-label test. |
| H2 familiar vs unfamiliar family | NO. All test families are external; there is no matched familiar-family intervention. |
| H3 capacity/structural inductive bias | PARTIAL: fixed, matched archived control contrasts can test external ranking differences, not establish an isolated physical mechanism or retrain capacity controls. |
| H4 graph information | PARTIAL: reference mapping and missingness audits describe representation limitations. Reference geometry does not measure mutant rearrangement or prove causal interactions. |
| H5 representation/preprocessing mismatch | YES for the specifically frozen scaled/unscaled rank-transfer contrast, conditional on all admission gates; not a general mechanistic explanation or evidence for every H5 subclaim. |

No hypothesis is supported or contradicted by this metadata assessment itself.

## 5. Blinding limitations, stop rules and current disposition

This task did not download or parse mutation-effect tables, generate variants, train models or calculate external performance. However, web search outputs were not cleanly separable from result narratives: unsolicited ASPA summaries included distribution/aggregate findings and individual variant-effect descriptions; PRKN and other searches also exposed aggregate result summaries. Those descriptions are not reproduced, used for candidate selection, or converted into data here. Their receipt is a blinding limitation. Therefore this document does NOT claim pristine outcome blinding. Candidate choice rests on organism, endpoint, publication/deposit traceability and rights, not the incidental findings. A new independent reviewer must adjudicate the exposure and source definitions before any confirmatory registration; later labels cannot be called untouched simply because no table was downloaded. This is separate from SOD1's inherited incident.

The existing source literature is public, making absolute knowledge independence difficult even for a new reviewer. Human abundance assays are not automatically safety-neutral in every downstream application; the proposed use remains bounded retrospective validation, with no variant optimization or clinical decision-making.

Other limitations: shared laboratories and assay technology; assay/family/construct shifts confounded; uncertain pretrained homology; domain overlap (especially PRKN); unknown available genotype coverage; varying oligomer/cofactor states; inherited source-selection censoring; noisy, assay-specific proxy endpoints; no calibrated physical uncertainty; four convenience-sampled families and low inferential power.

**Current decision: PARTIAL feasibility / STOP before execution.** Four fully qualified families have not been established. The next permitted activity under this document is metadata-only adjudication of this finite slate, not acquisition of labels or extension of the interrupted investigation. Any future execution would require a separate explicit authorization and a complete prospective registration with exact items, mappings, cohorts, checkpoints, exclusions, exposure disposition and numeric adequacy requirements. Failure to qualify is a design constraint, not negative evidence about generalization.

No tests or CI were run: this is a prose-only design outside the repository, and running research tests could access historical outcomes. Repository preservation was checked using opaque file hashes. No merges, commits, pushes or releases were made.

## 6. Source register

Metadata/publication links, not label-download instructions:

- GCK paper and data-availability record: https://doi.org/10.1186/s13059-024-03238-2
- Exact GCK supplementary landing page (CC BY + CC0 displayed): https://springernature.figshare.com/articles/dataset/Additional_file_2_of_Characterizing_glucokinase_variant_mechanisms_using_a_multiplexed_abundance_assay/26707496
- Separate GCK activity software deposit, not the proposed abundance endpoint: https://zenodo.org/records/7636310 ; license name was not rendered in the inspected page, so no data-rights inference from it.
- ASPA primary publication: https://doi.org/10.1038/s41467-024-48481-0 ; linked archive https://doi.org/10.5281/zenodo.8382504 (landing page inaccessible in this session).
- PRKN primary publication: https://doi.org/10.1038/s41467-024-45829-4 ; linked archive https://doi.org/10.5281/zenodo.8009574 (landing page inaccessible in this session).
- SOD1 rights, mapping and incident: existing benchmarks/m185/REPORT.md and source_inventory.json; no live score-set endpoint opened.
- TPMT original data-availability restriction: https://doi.org/10.1038/s41588-018-0122-z (search excerpt only; not a fresh exact-item rights audit).
- PTEN deposit metadata, not an adopted candidate: https://springernature.figshare.com/articles/dataset/Additional_file_2_of_Integrating_thousands_of_PTEN_variant_activity_and_abundance_measurements_reveals_variant_subgroups_and_new_dominant_negatives_in_cancers/16815903 (search metadata displayed license; subsequent landing-page open failed).
- Tsuboyama provenance exclusion: https://zenodo.org/records/7992926 and historical M12/M19 records.

The live metadata checks establish publication/deposit leads, not complete export verification. No accession, license or structure claim marked PARTIAL should be read as a gate pass.
