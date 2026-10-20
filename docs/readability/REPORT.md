# ProteinMod readability refactor — review report

Baseline: **8a07008ba427cb9cd82c84b5508a1326b562afa0**, clean
`codex/m18-3-masked-structural-admission` working tree. Refactor branch:
`codex/readability-refactor`, stacked on PR44. No merge, tag or release.

Read [the repository audit and before/after examples](AUDIT.md),
[the complete file inventory](inventory.json), [frozen baseline](BASELINE.json),
[bounded stage records](STAGES.md), [numerical evidence](equivalence.json), and
[the ten-function technical ownership guide](TECHNICAL_OWNERSHIP.md).

## Scope and result

The audit inventories all 158 original maintained Python files. 28 source modules /
46 functions receive targeted improvements. 130 original Python files, all
scientific artifacts/configurations/fixtures/reports and the three source-hashed
cache modules remain byte-identical. Existing APIs, public argument names,
checkpoint/module attributes, serialized fields, numerical operators, control
flow, model construction, RNG calls and scientific gates are preserved.

Internal names now describe residue correspondence, graph endpoints, hidden
states, edge messages, mutation pooling, sequence/graph attention queries,
training targets/scales/batches, calibration roles and task masks. Five LoRA
intermediate expressions make its projection and mutant/WT readout explicit.
No general framework, new architecture, learning objective or model-development
milestone is introduced. Historical scripts/tests are retained as stable oracles.

This is an engineering/comprehension refactor. It does not establish independent
authorship or add a new scientific performance claim.

## Verification

| Check | Result | Evidence / qualification |
|---|---|---|
| Frozen initial software baseline | PASS |215 tests,9. 36s; clean tree and821 tracked-file hashes captured before edits |
| Stage1 data/geometry | PASS |92 targeted tests; declared local-renaming AST comparison |
| Stage2 model kernels | PASS |34 targeted tests including LoRA gradients, masking and checkpoint restoration |
| Full refactored suite | PASS |Final result in verification.json; four new contract tests added to original215 |
| Ruff and dependency check | PASS |Check/format and `pip check`; no new dependency |
| Wheel packaging | PASS |Local `pip wheel --no-deps --no-build-isolation` produced proteinmod0. 1. 0 wheel |
| Protected current files | PASS |792 baseline files byte-identical,28 source files declared, CI provenance wiring explicitly changed |
| Restricted operation equivalence | PASS |Full module ASTs match after declared local renames and inlining five suffix intermediates |
| M0 historical replay | PASS |QC, metric/selection JSON and test-prediction CSV byte-identical |
| M1 historical replay | PASS |Metrics, full learning-curve records/subsets/predictions, PNG and report byte-identical |
| M1 generated metadata | EXPECTED DIFFERENCE |SVG timestamp/object IDs and new provenance manifest differ; SVG otherwise equal after explicit normalization |
| M17 historical selected states | PASS |All1,096 states /34,304 predictions replay with maximum error0. 0, both baseline and current |
| M15 real checkpoint | PASS |PF00018 G3, full training fraction, seed 0:192 predictions, max historical error0. 0 at original batch size32 |
| Kernels, gradients, training trace and RNG | PASS |Exact baseline/current hashes for GNN, attention, LoRA/WT gradients, selected state, three-epoch training, degree-null edges and final Torch RNG |
| M18. 3 masked geometry | PASS |All four family records /384 candidate IDs exact; outcomes not opened; BLOCKED status retained |
| Runtime/memory screen | PARTIAL |Three fresh-process pairs; tiny CPU GNN timing/RSS only, not production training or MPS |
| All M0–M15 retraining | NOT RUN |No claim of exhaustive retraining; old numerical records preserved and selected replays executed |
| Static typing/coverage | NOT RUN |No configured type checker/coverage requirement; pyright and pytest-cov unavailable locally |
| Independent cold-source reproduction | NOT RUN |Same operator/environment, existing checksum-pinned caches |
| External scientific admission / release | BLOCKED |Prior GFP physical-background and GEO item-license blockers unchanged; no unblinding or release |

Original historical source guards intentionally hash old source bytes. They
**should not pass directly against refactored source**. They were not edited.
The new verifier first checks current protected bytes and the restricted source
transformation, then runs the exact original guards in a temporary Git checkout
of the frozen baseline. CI now makes these two distinct claims explicitly.
There is no substituted/fabricated source hash in a current experiment manifest,
no rewrite of historical hashes and no relaxation of original M2/M18 admission.

## Numerical and performance findings

M0/M1 and M17 give direct historical replay evidence rather than test-only
assurance. The M15 comparison restores a real 35M-parameter ESM-2 LoRA checkpoint;
the gradient/RNG differential uses the existing tiny offline ESM test model.
Full historical M15 training and every M15 fold were not repeated.

Two single-item M15 CLI examples (one single mutation, one double) were exactly
equal between baseline and refactor. They differed from old batched inference by
7. 45e-8 and4. 17e-7 **in both revisions**. Replaying the original 32-item batching
removed that difference for all 192 fold variants. This is existing batch-shape
rounding sensitivity, not new evidence of cross-device portability. The report
does not conceal that initial mismatch or invent a relaxed success threshold.

Three fresh baseline/current process pairs, with alternating order, each measured
seven repeats of 300 tiny GNN forwards. Median-of-process medians: baseline
0. 023755s versus current 0. 024008s (**1. 06% slower**). Process peak-RSS medians were
417,726,464 versus 417,415,168 bytes (**0. 075% lower**). All outliers are retained
in equivalence.json; one current process was noticeably slower. An initial
unpaired RSS observation differed by 16%, which did not persist in the repeated
medians. These noisy, small-workload measurements support no speedup claim and
cannot rule out production-workload memory regressions. Full-suite wall times
also include different verification work and host contention; they are not model
performance benchmarks.

The style edits preserve exact tensor expression order after inlining. They do
not promise equivalence for custom side-effecting module hooks, deliberate local
variable introspection, arbitrary devices or different numerical-library versions.
Those cases were not exercised. Runtime metadata/source hashes naturally describe
the new revision, while scientific schemas and historical files stay unchanged.

## Reproduction

Run from repository root with the existing development environment. Archive the
baseline source without changing the current checkout:

```sh
mkdir -p /tmp/proteinmod-style-baseline-src
git archive 8a07008ba427cb9cd82c84b5508a1326b562afa0 | tar -x -C /tmp/proteinmod-style-baseline-src
python scripts/verify_readability.py --historical
ruff check .
ruff format --check .
pytest -q
python -m pip check
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/proteinmod-wheel .
```

`--historical --stdlib-only` reproduces the dependency-light CI guard; the full
command additionally runs frozen M16. 1 and M17 metric/role checks. Git history
must include the baseline revision; all CI checkouts now fetch history.

For separate baseline/current numerical execution, choose **new** output paths;
the replay script refuses to overwrite an existing output directory:

```sh
PYTHONPATH=/tmp/proteinmod-style-baseline-src/src python scripts/replay_readability.py --output /tmp/proteinmod-original --m17 --m15-fold
python scripts/replay_readability.py --output /tmp/proteinmod-refactored --m17 --m15-fold
python -m proteinmod.cli run --config configs/amylase.toml --output /tmp/proteinmod-m0
python -m proteinmod.cli run-m1 --config configs/amylase_m1.toml --output /tmp/proteinmod-m1
python scripts/audit_m183.py --output /tmp/proteinmod-m183-geometry
```

The lightweight kernel replay requires existing optional neural dependencies but
no downloaded PLM. `--m17` needs the original M12 structure/residue caches,
ProteinMPNN source/weights and all checksum-pinned M17 selected states. `--m15-fold`
needs the pinned ESM-2 base, prefix caches and original M15 checkpoint. M0/M1 need
the original Alpha Amylase dataset and M1 embeddings. The M18. 3 geometry replay
uses its four coordinate files, genotype projections and old CspA/CI2 graph cache;
it does not read external outcome tables. Historical reports explain acquisition
and licenses; a missing cache is a verification limitation, never permission to
re-extract with changed provenance or access blinded outcomes.

Compare `kernels.json`, `m17.json` and `m15.json` exactly across revisions. Compare
M0/M1 scientific outputs against their immutable benchmark directories; do not
expect new provenance timestamps or generated SVG IDs to match. Timings and RSS
are observational. No generated weight files or expensive binary caches are
committed here.

## Remaining risks and review decisions

- The source transformation is deliberately restricted. A broader parser/runner
  reorganization, public API rename or cache-identity migration was isolated from
  this task rather than accepted without stronger evidence.
- Independent dataset licensing/admission and M18 background uncertainty remain
  unresolved scientific blockers. This style work does not resolve or weaken them.
- Source-hash verification requires the explicit historical/current workflow above;
  invoking an old source guard directly on changed files fails by design.
- Review commits are separated into baseline, data/geometry, kernels and
  experiment/control stages, followed by documentation/verification evidence.
  No commit rewrites prior history. Exact final head and CI conclusions belong in
  the draft PR, avoiding a self-referential evidence commit.
