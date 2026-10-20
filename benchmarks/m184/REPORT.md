# M18.4 — outcome-blind WT background and permissions qualification

**Admission remains BLOCKED. No external outcome values were decoded, no final
confirmation manifest was issued, and no confirmatory model was fit or run.**
This is a qualification investigation, not an independent validation result.

Parent PR46 f7df45bfc94cbbeadfeaf2538f4a85ca88d37c2f was verified at its exact
head with12 successful CI checks. The user's explicit PR46 instruction supersedes
the older PR44 base in issue45. Work began clean on a dedicated stacked branch.
The evidence plan was committed as35a744f8bece849da6224c0c40554a40a6a9b025
before this investigation. The original M17 H1–H5, four-family denominator,
>=3/4 directional criterion, budgets, seeds and outcome restrictions remain intact.

## GFP: exact sequence is necessary, not sufficient

The primary RCSB search was expanded from13 inspected records to **all732 returned
95%-identity polymer entities**. None contains the full exact237-aa assay segment.
This bounds the negative search; it is not proof that no compatible structure can
exist in another database or future deposition. We did not confuse search score
with identity, rename residues or edit an experimental construct into a fake WT.

The canonical AlphaFold P42212 model metadata (v6) describes a238-aa sequence
with initiator Met and F at the assay's L63 position after index correction.
It therefore cannot establish exact assay-WT geometry. Metadata are archived;
no canonical model was silently substituted.

The official ESMFold API was queried with the verified237-aa assay reference
only. The returned154104-byte PDB is pinned by SHA256
`39bc266a5726afe082704ae714a9b463a18eb8869ea6d5eae052a3b7335eb462`.
Unchanged full-reference M2 accepts **237/237** exact compatible CA sites, unique
mapping, no missing positions or residue differences. The full diagnostic graph
has237 nodes,1110 undirected edges, one component and no isolated nodes. All96
previously genotype-selected GFP candidates pass input construction and exact
mutation-order invariance. Confidence descriptor missingness is explicit. Existing static/dynamic/ridge/edge
widths 34/25/270/29 pass, including 2220 finite directed chemical-edge rows.

The original227 eligible positions and all candidate IDs were kept for paired
geometry comparisons. The newly predicted terminal/mismatch positions did not
expand the confirmatory cohort. A single proper-rotation Kabsch fit on the227
common CA atoms gives:

| Outcome-blind comparison | Actual result |
|---|---:|
| Common CA RMSD |8.947457 Å |
| Common backbone RMS displacement (907 available atoms) |8.880839 Å |
| Backbone at excluded mismatch positions63/79 (8 atoms) |7.824855 Å |
| Backbone at20 prior contacting positions (79 atoms) |8.449025 Å |
| Common identity side-chain atom displacement RMS |10.012503 Å |
| Common CA contacts at8 Å:1GFL / prediction / intersection |1145 /1055 /847 |
| Contact Jaccard |0.626016 |
| Mean absolute eligible RSA difference |0.150008 |
| Maximum absolute eligible RSA difference |0.807432 |

Side chains are compared only where residue identities agree, using matched
atom names. These are displacements under the global CA fit, not separate local
rotamer fits. At63/79 the identities differ, so we compare backbone rather than
claiming atomwise side-chain equivalence. Every selected variant's pairwise
mutation distances is archived in `GFP-geometry-audit.json`. The20-neighbor result
is descriptive; the historical33/96 variants touching those neighbors is unchanged.
RSA differences include the experimental deposition's other chains/ligands versus
a predicted monomer, so they do not isolate an L63F/Q79R causal effect.

The returned CA confidence field ranges **0.20–0.86**. Official ESMFold source at
2b369911bb5b4b0dda914521b9475cad1656b2ac multiplies categorical lDDT by100
before PDB output; the API does not expose its normalization or weight revision.
We retain **unknown** confidence semantics and missing pLDDT/B-factor descriptors,
without multiplying by100 or applying AlphaFold thresholds. An output hash pins
this response, not the served checkpoint or independent cold regeneration.
The PDB header declares CC-BY-4.0 for returned atlas data; that statement is not
a blanket license for model software/training data. pLDDT, experimental B-factors,
and predictive uncertainty remain separate quantities.

M2 allows checksum-bound local predictions with unknown confidence. There is no
invented blanket prohibition on predicted structures. However, M18.3 preregisters
specific deposited coordinates, masks and deposited-context ASA. Replacing that
source with this full237-node prediction, its confidence policy and chromophore
chemical state requires an explicit **prospective source/scope amendment** before
confirmation. Canonical sequence/CA completeness alone does not establish mature
GFP chromophore chemistry or accurate WT geometry. The large observed differences
are a limitation, not a post-hoc numerical rejection threshold. No such amendment
was issued while the mandatory permission gate remains unresolved. Original1GFL
background admission remains blocked, including its full-M2 rejection record.

## CspA/CI2: authority gap narrowed, not fabricated away

The complete authoritative GEO disclaimer is now available through indexed primary
text, closing the previous retrieval gap. GEO imposes no NCBI restriction and permits
public access; it explicitly reserves possible third-party rights. We do not claim
that GEO research reuse is legally prohibited. The preserved admission criterion
requires exact-item permission or licensed lineage, not just general accessibility.

The author concept DOI now resolves to **Zenodo16266645 (2025), CC-BY-4.0**.
We inspected both ZIP central directories (93 code /3682 data entries), the licensed
README and the processed-table header, without decoding outcome rows or notebook
outputs. The table is the same size/CRC as the prior licensed processed table and
has replicate/fitness/growthrate fields. No authoritative export from that table
to the GEO `fitness_raw`/`sigma_raw` columns was established in the inspected
README, file inventory and archived relevant source-only Fig2/FigS1 projections.
A different filename/header/hash does not disprove derived lineage. A shared paper
or matching protein IDs does not establish it either.

The remaining precise gap is an explicit statement covering the exact GEO item
and those raw columns, or a source/version/column transformation linking them to
the licensed deposit. The local permission-request draft names the item SHA256,
columns, intended research, predictions/model sharing, redistribution and commercial
scope separately. Its recipient is verified by the primary CRG directory. **It was
not sent.** We retain the mandatory gate and all labels remain blinded.

## Qualification matrix

| Gate | Status | Evidence / remaining limit |
|---|---|---|
| PR46 exact-head baseline |PASS |12/12 parent checks; exact revision archived |
| Readability and historical bytes |PASS |833 protected parent files exact; only parent CI workflow adds guard |
| Original M2/M17/M18 rules |PASS |No source/API/configuration/old artifact edits; frozen guards pass |
| CspA /CI2 /TEM1 prior geometry/cohorts |PASS |Four-family M18.3 replay,384 candidate IDs exact |
| GFP exact-WT sequence/strict M2 diagnostic |PASS |237/237,96 genotype interfaces; no outcomes |
| GFP structural-background admission |PARTIAL |No exact experimental hit; prediction source/confidence/chemical qualification incomplete |
| CspA/CI2 exact GEO permission/lineage |PARTIAL |New deposit/header/terms evidence; specific authority gap remains |
| Four-family independent confirmation |BLOCKED |No final source/fold/checkpoint/read manifest;0 confirmation runs |
| M0/M1 scientific output replay |PASS |Metrics/QC/predictions and M1 subset/learning-curve data exact |
| M17 selected-state replay |PASS |1096 fits /34304 historical development predictions, max error0 |
| M15 real LoRA checkpoint replay |PASS |PF00018 G3 seed0 full fraction,192 predictions, max error0 |
| GNN/attention/LoRA/control kernels |PASS |Frozen-source digest equality retained |
| Full local and clean-source tests / Ruff / formatting / build |PASS |225 tests in both checkouts; lint/format/package checks pass |
| Independent cold-source confirmation |PARTIAL |Network search reproducible in principle; exact API response/backend and old model caches still required |
| License audit |PARTIAL |Existing MIT source LICENSE preserved; upstream item gates remain separate |
| Release qualification |NOT QUALIFIED |No tag/release; external confirmation not admitted |

## Reproduction

Use the existing project virtual environment; no dependency was added. This new
work is CPU geometry/metadata only, with deterministic operations and no random
seed consumption. Package versions, replay timing and peak memory are recorded in
`verification.json` and `replay-*.json`. A separate clean local clone at436ebbb (no research caches copied) also passed
225 tests and the stdlib M184/frozen-history checks, using the same installed
dependency environment. This is source isolation, not independent data/model
regeneration. Exact current CI results belong to the
PR check suite at its verified head, avoiding a self-referential commit hash here.

```bash
python scripts/verify_m184.py
python scripts/verify_readability.py --historical
ruff check .
ruff format --check .
pytest
python -m pip check
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/proteinmod-wheel .
```

With the checksum-pinned coordinate/genotype caches:

```bash
python scripts/audit_m184_gfp.py --output /tmp/m184-new-gfp-audit.json
python scripts/audit_m183.py --output /tmp/m184-new-old-geometry
python scripts/replay_readability.py --output /tmp/m184-new-replay --m17 --m15-fold
python -m proteinmod.cli run --config configs/amylase.toml --output /tmp/m184-new-m0
python -m proteinmod.cli run-m1 --config configs/amylase_m1.toml --output /tmp/m184-new-m1
```

The M184 comparison regenerates only its diagnostic cache. The generated M2
manifest hash embeds time/runtime/revision; exclude that one field when comparing
geometry replay JSON. All other diagnostic values reproduced exactly. Old M183
family records must compare exactly. M0/M1 metrics/QC/prediction bytes and M1
subset/learning-curve data must match their historical benchmark directories;
runtime manifests/plot metadata are outside that byte comparison.

For a fresh network inventory (primary sequences only):

```bash
python scripts/search_m184_gfp.py --output /tmp/m184-new-primary-search
```

RCSB is live and can change. The committed732-entity snapshot is the frozen record,
not an assertion that a future query must return732. Current ESMFold acquisition
uses the official POST endpoint with the exact sequence in the diagnostic config;
verify the response SHA before using it. Its backend is unpinned: do not substitute
a different response silently. The coordinate and raw target files are not committed.
`cache_inventory.json` pins fetched metadata/source/coordinate bytes, not download
availability. The default integrity guard works without these caches. Full neural
replay still depends on original dataset/embedding/ProteinMPNN/ESM2/M17/M15 caches;
we do not claim independent cold-source scientific replication or full M0–M15
retraining. Performance numbers are tiny-kernel diagnostics, not training claims.

Historical scientific reports and failed admissions are immutable. No new model,
architecture, supervised budget, target filtering, stability label interpretation,
family threshold, source-code license choice, merge, tag or release was introduced.
