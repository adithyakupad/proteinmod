# ProteinMod M0–M19 research-platform closeout

## Scope and decision

Issue52 consolidates the maintained research platform, not a new prediction model.
All eight requested PRs merged in dependency order without squash/force-push. Main
14fa4c37fbafda9fbc4bd7f5be482f632062c9d1 has the exact tree of verified PR51 head
4b0441b092aa141479d10855206dd34c4997434b. dependencies.json records every original
head and merge commit. Final-main CI is recorded separately from candidate-head CI.

Original source is MIT licensed. The release candidate is a **code-only wheel**,
with explicit third-party notices and optional external acquisition. A full Git
archive is not a uniformly MIT dataset/model distribution. No release/tag is made;
final approval remains required. Independent four-family confirmation is BLOCKED.

## Scientific synthesis

The defensible thesis is that representation preprocessing, regularization and
family shift materially affect model rankings. In bounded experiments, graph,
fusion, confidence-conditioning and LoRA complexity did not consistently improve
held-out prediction over strong ridge baselines. These are measured development
findings, not a universal claim that structural biology or neural models fail.

| Research path | Maintained evidence and limits |
|---|---|
| M0/M1 | Official FLIP2 Alpha Amylase split; one-hot ridge rho.648837, RMSE.042845 in distributed assay units. Frozen ESM mutation probes and low-label curves. Split shares410 substitutions train/test, so not substitution-disjoint |
| M2/M3 | Experimental/predicted structure parsing, reference mapping, RSA/contacts/missingness; mutation-invariant structural ridge and compact GNN; reference geometry is not measured mutant rearrangement |
| M4/M5 | Protein-held-out evaluation and weighted/unidirectional/bidirectional fusion; matched structural/sequence cohorts; no general family claim from single-protein results |
| M6/M7 | Native stability sign/unit contracts, separate function/stability heads, correctly masked unpaired-task losses; negative transfer retained |
| M8 | Stability-conditioned function experiments with shuffled controls; OOD native-stability covariates remain physically unsupported and safe export returns null |
| M9/M10 | Assay metadata and typed structural confidence/missingness, capacity/shuffle controls; mixed/negative results, no pLDDT/B-factor/uncertainty conflation |
| M11 | Measured combinatorial variants, additive and symmetric pair models; explicit3D controls do not establish physical interaction recovery |
| M12 | Eight measured short domains/four Pfam families, G1/G2/G3; source/assay-specific labels preserved; pretraining overlap unknown |
| M13 | Bootstrap/ensemble and leakage-safe calibration diagnostics; one G3 calibration family does not provide finite80/90% family-block guarantees; no calibrated OOD deployment claim |
| M14/M15 | Richer edge GNN and frozen ProteinMPNN controls; last-block rank2 query/value LoRA and matched heads; no consistent superiority over ridge |
| M16/M16.1 | Integration, selected checkpoint replay, historical cold M0/M1 CPU/MPS source regeneration, real structure/native-adapter checks; strict historical cross-device failures preserved |
| M17 | H1–H5 preregistered development mechanism study:1096 fits/34304 predictions; scaling effect, family heterogeneity and degree-preserving graph nulls |
| M18–M18.5 | Blinded provenance/structure/permission qualification; original rejections, masks, failed candidates and SOD1 aggregate-metadata deviation retained; zero independent predictive evaluations |
| M19 | Controlled expanded ridge search failed gain criterion and scalar-equivalence gate; no test-informed rescue sweep |

Detailed historical reports/configurations remain under benchmarks/ and docs/.
They are dated evidence: earlier source-license HOLD entries are not rewritten.

### Two family protocols, never one combined headline

Historical M12/M14/M15 G3 evaluates all96 selected variants per held-out protein,
768 across four family folds. Macro Spearman: structural ridge.4298, ProteinMPNN
ridge.4147, original GNN.4020, frozen ESM ridge.4008, edge GNN.3799,
bidirectional attention.3503, LoRA.3426. Sources: benchmarks/m12/summary.json,
benchmarks/m14/summary.json, benchmarks/m15/metrics.json and their reports.

M17/M19 H1 G3 instead evaluates original16 test variants/protein,32/family.
At full labels, four training proteins contribute64 rows each from two families;
a third family validates, fourth tests. M17 unscaled ESM rho.344853 versus
scaled.204412: paired mean+.140441, positive4/4 family contrasts. This intervention
was preregistered within development data, not externally confirmed.

M19 retains those IDs/fractions/seeds. Expanded unscaled.290441; validation-selected
expanded.147059 versus historical unscaled.344853, delta−.197794, positive0/4.
Family deltas: SH3−.355882, homeodomain−.041176, chromo−.248529, WW−.145588.
The >=.02/>=3-of-4 criterion fails. Scalar invariance max1.210020597608441e-7
exceeds unchanged1e-7 tolerance in one expanded case. Both failures are preserved.
The secondary25%-label gain is not substituted for the failed primary endpoint.

Validation-family selection can transfer poorly. The results do not establish that
normalization itself causes failure: energy normalization with alpha adjustment is
a reparameterization, while the search and selection procedure is the intervention.
Four families and repeated seeds cannot support strong population inference.

### External confirmation firewall

SOD1 individual variant outcomes remain blinded. Initial metadata JSON decoding
included a scoreDistribution object; values were not displayed/used but strict
predecode blinding failed. Independent review is required before any future
confirmation. The deviation is immutable at benchmarks/m185/BLINDING_DEVIATION.md.
SOD1's2632 structurally eligible variants and96 selected candidates are not2632 or96
independent confirmations. Its conditional153-residue structure mask does not erase
the original strict M2 missing-Met rejection. CspA/CI2 file-specific GEO permissions
remain unresolved; GFP exact-WT structural-background compatibility remains a
historical blocker if used instead of the proposed SOD1 replacement. TEM1/source
permissions retain their recorded restrictions. No final external manifest, label
unblinding, new architecture, training sweep or release tag occurred in closeout.

## Engineering closeout

README now separates current scoped qualification from immutable historical HOLD
records. THIRD_PARTY_NOTICES.md and wheel metadata declare MIT for original code,
with separate data/model obligations. The code-only wheel excludes benchmarks,
configs, external workbooks, weights, arrays and caches. Dependencies install
separately; their binary-license obligations are not transferred to ProteinMod.
Full-repository redistribution and cross-resource annotation terms remain PARTIAL.

Historical guards originally froze README/package metadata as well as research
artifacts. release_candidate/amendments.json records exact new hashes and prior
Git blobs for seven reviewed documentation/guard-wiring changes. A hard-coded
allowlist forbids exceptions for any benchmark, config or ML implementation.
verify_release_candidate.py checks all933 files from consolidated main:926 remain
byte-identical, seven are explicit amendments. The old source remains recoverable
and verified by SHA256 from Git objects. Tampering or unlisted changes fail tests.
No artifact manifest, model math, serialized state, split, admission threshold or
historical numerical result was changed. The M19 runner only delegates its final
history check; it was not rerun as a new training sweep.

## Verification and limits

Selected checkpoint replay, zero refitting of authentic-data models in closeout:
- M17: all1096 selected states /34304 held-out predictions, max error0.
- M15: one real G3 LoRA fold /192 predictions, max error0; original batch32.
- M19:480 records /15360 predictions, max error0; selected-arm aliases share states.
- Actual BBC1 single P56N and double D45S;N47R LoRA inference succeed; outputs are
  unbounded training-assay ranks, not kcal/mol or calibrated intervals.
- Safe M8 export invalidates all13848 archived unsupported physical predictions:
  prediction/unit null, thermodynamic_eligible false. Raw evidence unchanged.

replay-*.json, inference_examples.json, safe_ood.json and verification.json bind
these claims. Clean consolidated main tests run without research caches, using the
existing pinned dependency environment. A fresh temporary environment separately
installs the code wheel and research dependencies; final candidate clean-checkout
results, wheel manifest/hash and CI links appear in verification.json.

Historical M16.1 empty-cache acquisition covered3706 M0/M1 variants on CPU/MPS,
1UBQ PDB/mmCIF geometry and1800 native-stability rows; not rerun here. It retained
raw-layer11/12 engineering-screen failures and the older1e-6 cross-device failure,
while its functional portability criterion passed (max normalized drift6.41497e-5).
No CPU/MPS equality or complete cold M0–M19 replay is claimed. Large checkpoints
and frozen embeddings still require the exact pinned caches or explicit acquisition.
No deployment/security certification or independent legal opinion is claimed.

## Qualification and review

See QUALIFICATION.md for separate scoped code and scientific statuses. Recommended
first tag after owner review: **v0.1.0**, matching existing package metadata and
explicitly described as a research release. Do not create it automatically.
No substantive new improvement application bullet is supported; the defensible
ML evidence is controlled falsification, leakage/shift analysis, numerical auditing
and reproducible engineering. Protein mutation prediction is not neural decoding.
