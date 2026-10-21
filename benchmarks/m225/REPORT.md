# M22.5 — prospective synthetic checkpoint-substitution bridge

**PASS: all 20 preregistered local bridge cases and mandatory controls passed.**
Every resumed model/optimizer array is bit-identical to uninterrupted M22.1 training,
with exact RNG state, counters, preprocessing, training-loss trace and endpoint metrics.
No numerical tolerance rescued a local difference. No biological data, fitting,
variants, tuning, optimization or historical research replay was used.

## Prospective registration and preserved design

Parent [PR #64](https://github.com/adithyakupad/proteinmod/pull/64) remains a separate
unmerged draft at 3482d76961115078e97a1489c0d80f3325da8402. The new branch and draft
are stacked on that exact head. The prose [preregistration](PREREGISTRATION.md) was
committed at d71fdbc, followed by its machine-readable identity inventory at c7ee08c,
before any new bridge implementation or numerical tests. Tested implementation was
frozen at 40f97a5 before the complete registered run. The opaque hash inventory took
longer to finish than the initial prose commit; both registration commits predate code.

The original config bytes and two scenarios, seeds 0–9, 16→64 tanh→1 float64 MLP,
Glorot initialization, Adam settings, training-only mean/population SD, original
PCG64 namespaces, permutation/chunk32 schedule and epochs 12/120 are unchanged.
The bridge calls original.fit directly for both uninterrupted and resumed training.
It does not copy or rewrite the optimization loop, use M22.3's grouped sampler,
change the scenario, choose a checkpoint by validation, or change data generation.

The M22.3-style boundary reuses its canonical serialization, array validation,
environment identity and explicit exception primitives. Its fixed 6→12 fixture and
checkpoint schema remain unchanged; this is a distinct M22.5 schema for the existing
M22.1 state, not a claim of format interchangeability with the M22.3 fixture.

## All-case decision

| Original scenario | Seeds retained | Prefix updates / terminal updates | Exact prefix / disk / resume / repeat / terminal | Endpoint differences |
| --- | --- | --- | --- | --- |
| ample_clean, train 512 | 0–9 (10/10) | 192 / 1,920 | PASS for every seed | All train/validation/test MSE and Spearman differences = 0 at both epochs |
| scarce_noisy, train 32 | 0–9 (10/10) | 12 / 120 | PASS for every seed | All train/validation/test MSE and Spearman differences = 0 at both epochs |

For every pair the independent 12 prefix matches the uninterrupted run's epoch-12
snapshot. The saved prefix is reloaded, checked, then continued for 108 epochs. A
separate uninterrupted rerun matches at both endpoints. The terminal envelope also
round-trips exactly and cannot be resumed beyond 120. The complete original 120-epoch
batch plan is bound to its identity, and expected RNG/counter boundaries are checked.

[results.json](results.json) retains every case, its state/file/input/schedule/source
hashes, exact comparison components, full endpoint metrics, controls and seed-level
summaries. For each scenario, epoch, split and metric, ten paired seed differences
have mean 0 and sample SD 0. The conditional fixture interval [0,0] is an engineering
summary, not an inference over platforms, tasks or scientific model performance.
No failed case was removed or replaced. [verification.json](verification.json) records
local checks and result identity. JSON checkpoints stay ignored in runs/m225/registered-run.

## Controls and integrity

| Gate | Result | Evidence |
| --- | --- | --- |
| Envelope schema/identity/dtype/shape | PASS | Unknown/missing fields, unsupported schema/version, array coercion/nonfinite/shape/dtype errors and negative second moments rejected |
| File corruption and bounded parsing | PASS | Trusted external file SHA256, payload checksum, truncation, duplicate keys, invalid JSON and size limit controls |
| Input/config/scenario/seed/source/environment | PASS | Changed training values/order, scenario/seed, configuration, preprocessing/source/environment identity and mutated contract rejected before fitting |
| Optimizer/RNG/schedule | PASS | Reset Adam counter/moments, incorrect update counts, changed RNG or schedule rejected |
| Held-out noninterference | PASS | Training-only contract/API; adversarial evaluation-array replacement changes no saved state or resumed trajectory; exercised for every case |
| Exact continuation | PASS | 20 full runs, independent prefixes/reruns, disk continuation and terminal round-trips; fresh-process continuation also tested |
| Software checks | PASS | 30 focused bridge tests; 77 combined synthetic tests; 24 optimized-Python rejection/control tests; Ruff check and format |
| Parent/history preservation | PASS | All 994 parent files unchanged; all 38 original local refs unchanged, including archive/m203-interrupted at 808441a80488cb4a633bc4329a096929df5cd521 |

Original standalone implementations, M0–M21 evidence and M22.1/M22.3/M22.4 reports,
results, configs, frozen manifests, tests and routing files are byte-identical to the
parent. Existing M22.4 and release-candidate integrity verifiers pass unchanged.
The new pytest plugin and pytest.ini add branch-specific synthetic CI collection
without editing the parent's frozen root router. It permits only enumerated new
M22.5 files/evidence; any parent modification aborts collection. No biological test
module is collected on this dedicated branch's CI route.

The original M22.1 functions retain their historical guards and format. M22.5 rejects
untrusted checkpoints before calling them and validates the terminal state afterward.
New runtime gates use explicit exceptions, not assertions. Optimized-Python rejection
checks and an independent subprocess verify that these guards remain active under -O.

## Numerical scope and limitations

Exact equality is required original-versus-bridge within each environment. The local
run also matches the historical synthetic endpoint metrics with maximum difference 0.
CI independently reruns all 20 exact local substitutions on Linux and separately uses
the unchanged M22.1 fresh-run tolerance rtol=1e-8, atol=1e-9. Remote results are on the
draft PR. Checkpoint continuation across changed environments is rejected; tolerance
never substitutes for an exact bridge comparison.

Only the fixed M22.1 architecture, registered training cohorts and completed epochs
12/120 are supported. No mid-epoch, distributed, GPU, dropout or scheduler continuation
is claimed. Initialization/data RNG streams are isolated and deterministic; the only
RNG needed for continuation is the captured original order stream. The initial model
identity is also recorded. SHA256 requires trusted custody: an attacker controlling
both a fabricated internally valid checkpoint and the external trusted digest can
substitute weights. Interrupted writes fail loading; atomic crash-proof writes and
cryptographic signatures are not claimed. This result establishes serialization
substitution for these fixtures, not neural convergence, protein-model quality, or
biological generalization. No follow-up experiment is executed by this milestone.

## Reproduction

With the project's development dependencies installed, use new output paths each time:

```sh
pytest -q tests/test_m221_synthetic.py tests/test_m223_checkpoint.py tests/test_m224_integration.py tests/test_m225_bridge.py
python -O -m pytest -q tests/test_m225_bridge.py -k 'rejection or changed_identity or corruption or optimizer_reset'
ruff check .
ruff format --check .
python scripts/verify_m225.py
python scripts/run_m225.py --output runs/m225/reproduction --compare benchmarks/m221/results.json
python scripts/verify_m225.py --result runs/m225/reproduction/results.json
```

On the original checkout, `--local-refs` additionally checks all historical refs.
CI checks parent bytes without requiring the local archive to be pushed. Original
source drafts remain unmerged; no tag or release is created.
