# ProteinMod

ProteinMod is a reproducible research benchmark for **protein mutation-effect
prediction**. It compares classical sequence models, frozen ESM-2, residue graphs,
cross-attention, independent/joint stability and function heads, assay metadata,
structure reliability, explicit interactions, uncertainty and bounded ESM-2 LoRA.

The main finding is that more complex models do **not** consistently improve
held-out-protein or held-out-family prediction. This is research software for
source-matched substitution experiments, not a universally validated predictor.

## Qualification and evidence

**Scoped research-code release candidate; not deployment-ready.** M0–M19 are
integrated on main at `14fa4c37fbafda9fbc4bd7f5be482f632062c9d1` after merging
PRs #38/#40/#42/#44/#46/#47/#49/#51 in order. Original source is [MIT](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) keep data/model terms separate.
The candidate is a code-only wheel, not a cleared dataset/checkpoint bundle.

**Independent four-family confirmation remains BLOCKED.** SOD1 individual labels
remain blinded, but an aggregate-metadata decoding deviation requires independent
review. CspA/CI2 GEO item permissions and GFP structural-background questions remain
explicit blockers; candidate counts are not independent validation results.

M17's unscaled ESM ridge improved mean family contrast by0.1404 over scaled ESM;
physical topology did not establish an advantage over degree-preserving rewiring.
M19's expanded ridge search failed: validation-selected rho0.1471 versus historical
unscaled0.3449, difference−0.1978, negative in4/4 families. Its scalar-equivalence
control also failed (1.21e-7 vs unchanged1e-7 tolerance). No result was replaced.
See the [M19 report](benchmarks/m19/REPORT.md), [closeout technical report](release_candidate/REPORT.md),
[qualification matrix](release_candidate/QUALIFICATION.md), and
[proposed release notes](release_candidate/RELEASE_NOTES.md).
Historical M16/M16.1 FAIL/HOLD records remain dated evidence, not current license claims.
No tag, release or publication is authorized before final review.

## Install and test

Python 3.11–3.14 supports the offline core. The full measured Apple Silicon
CPU environment uses Python 3.12 and the pinned versions below:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-m3.txt -e '.[dev,esm,m3,structure]'
ruff check .
ruff format --check .
pytest -q
python -m pip check
```

For the small core alone: `python -m pip install -e '.[dev]'`.
Tests use synthetic fixtures and do not download models or experimental datasets.
See [closeout verification](release_candidate/verification.json) for actual clean-checkout
installation, wheel contents, final-main CI, inference and replay results. Full
M0–M19 cold-source reproduction is not claimed.

## Start with the measured benchmark

```sh
proteinmod download --config configs/amylase.toml
proteinmod validate --config configs/amylase.toml
proteinmod run --config configs/amylase.toml --output runs/example-m0
```

This downloads one official FLIP2 Alpha Amylase file and preserves its original
2,403/601/702 train/validation/test partition. One-hot ridge test Spearman is
**0.648837**, RMSE **0.042845** in the distributed assay units. The official split
shares 410 substitutions across train/test; it is not substitution-disjoint.
[Full evidence](benchmarks/amylase/report.md).

For source acquisition and reproducible neural experiments, follow the milestone
protocols: [sequence/structure](docs/m3_protocol.md), [cross-protein](docs/m4_protocol.md),
[cross-attention](docs/m5_protocol.md), [native stability](docs/m6_protocol.md),
[multitask](docs/m7_protocol.md), [family-held-out](docs/M12.md),
[uncertainty](docs/M13.md), [edge-aware graphs](docs/M14.md), [LoRA](docs/M15.md).
Downloads are explicit; hashes bind datasets, references, structures and encoders.
Expensive weights/features are ignored caches, not committed binaries.

## Bounded inference example

After acquiring/preparing M12/M15 and running M15 as documented in [M15](docs/M15.md),
these use real saved **held-out-family** LoRA weights, rather than fabricating a model:

```sh
python scripts/infer_m16.py --protein BBC1 --mutations P56N
python scripts/infer_m16.py --protein BBC1 --mutations 'D45S;N47R'
```

The examples accept validated benchmark substitutions in either order, verify
checkpoint hashes, and report task, assay, units and interval eligibility. Unknown
variants/proteins are rejected. These predictions are training-assay rank outputs,
**not kcal/mol**; M13 intervals cannot be attached to a separately trained LoRA model.
For all other model paths, use the [checkpoint replay recipes](docs/M16.md#reproduction)
which verify masks, roles, metadata and native-unit conventions against held-out outputs.
There is no single trained checkpoint that jointly includes every research extension.

For safe handling of archived M8 stability outputs, use
`python -m proteinmod.stability_inference --output runs/safe-stability.json`.
Unsupported OOD predictions are explicitly null and thermodynamically ineligible;
raw benchmark files remain diagnostic evidence, not deployable physical predictions.

## Matched family benchmark highlights

Eight small domains, four independently annotated Pfam families, 768 measured
variants (438 singles/330 doubles). G1 holds out variants, G2 proteins, G3 families.
Each row below uses the same M12 cohort and original roles; G1 and G2/G3 have different
test populations. Correlations average proteins/families, then fixed seeds.

| Model | G1 Spearman | G2 | G3 |
|---|---:|---:|---:|
| Frozen ESM-2 ridge | 0.7040 | 0.5401 | 0.4008 |
| Structural ridge | 0.5533 | 0.3996 | 0.4298 |
| Original residue GNN | 0.4170 | 0.4014 | 0.4020 |
| Edge-aware GNN | 0.4440 | 0.3991 | 0.3799 |
| Bidirectional cross-attention | 0.6225 | 0.4932 | 0.3503 |
| Frozen ProteinMPNN ridge | 0.5662 | 0.4538 | 0.4147 |
| Last-block ESM-2 LoRA | 0.6297 | 0.4460 | 0.3426 |

These are stability-effect ranking assays, not general biochemical function scores.
[M12 per-family results](benchmarks/m12/REPORT.md),
[M14 controls](benchmarks/m14/REPORT.md), [M15 capacity/cost controls](benchmarks/m15/REPORT.md).
The original GNN and M12 `structure_only` wrapper are distinct comparators.

## M17/M19 controlled development cohort

These results use original16 test variants per protein (32/family), not M12
all96/protein. Four families, training-only transformations and validation-only
selection are retained. At full labels M17 unscaled/scaled ESM ridge rho is
0.3449/0.2044; M19 expanded unscaled/selected rho is0.2904/0.1471. These are
reused development outcomes, not new independent validation.

## Scientific limits

- Reference structures describe WT geometry; mutant rearrangements are not observed.
  Unmapped residues, mismatches, structural missingness and confidence types remain explicit.
- Native M6 folding ΔΔG is kcal/mol, positive destabilizing. DMS ranks, enrichment,
  abundance, activity and thermodynamic units are never treated as interchangeable.
- Four families are four effective replication units. Seed SD is not a population
  confidence interval; ESM/structural pretraining overlap is unknown.
- Joint training shows negative transfer. Assay conditioning, confidence gates,
  explicit 3D interactions and LoRA do not establish consistent generalization gains.
- M8 OOD stability predictions are diagnostic covariates, not validated thermodynamic
  inputs. pLDDT, B-factors, predictive uncertainty and OOD are distinct quantities.
- One G3 calibration family cannot support finite 80/90% family-block conformal
  guarantees. Aleatoric variance is not independently measured.
- ProteinMod source is licensed under [MIT](LICENSE) on this branch. Dataset/model
  terms remain separate; the license does not clear third-party redistribution.

The [historical README](HISTORICAL_README.md) preserves the original milestone narrative;
all historical benchmark files and configurations remain unchanged.

M18 external qualification is [blocked before label access](benchmarks/m18/REPORT.md): primary abundance-assay lineage is traced, but four historical sequence-similarity flags remain unresolved. No independent external performance is claimed.

M18.1 adds [blinded homology/profile adjudication and a primary TEM-1 lineage audit](benchmarks/m181/REPORT.md). The four original flags remain recorded; confirmation is blocked because fewer than four test families are qualified.

M18.2 traces a [fourth primary family and TEM-1 workbook permissions](benchmarks/m182/REPORT.md). Genotype-only common masks are recorded; strict structural admission remains blocked, so no confirmatory performance is claimed.

M18.3 implements [versioned masked extraction and a four-family blinded cohort audit](benchmarks/m183/REPORT.md): 384/384 candidate inputs pass, with original M2 rejections preserved. Confirmation remains blocked by GFP background compatibility and unverified GEO raw-item license linkage; no external outcomes were unblinded.

For exact single/multiple-mutation example outputs and fail-closed OOD export, see
[reproduction and inference](release_candidate/REPRODUCTION.md). All historical
negative results, M19 numerical failure and SOD1 deviation remain immutable.
