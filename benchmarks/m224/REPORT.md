# M22.4 — synthetic infrastructure integration

**PASS for the tested integration scope.** Both independent synthetic modules coexist
without changing their training code or original evidence. Local tests, exact replays,
checkpoint rejection controls, Ruff, wheel construction/content checks and historical
integrity checks pass. Remote CI results are recorded on the draft integration PR.
This is infrastructure validation; it makes no biological or scientific performance claim.

## Dependencies and overlap

| Source | Exact imported head | Relationship |
| --- | --- | --- |
| [PR #62](https://github.com/adithyakupad/proteinmod/pull/62) | 2e82ff477979fffa398ae778ba2bac1162e24205 | M22.1 registered horizon diagnostic; independent numeric module |
| [PR #63](https://github.com/adithyakupad/proteinmod/pull/63) | df58fe0dbeb0df772465533659a8c5c7b3a9fd78 | M22.3 strict checkpoint fixture; independent numeric module |

Both drafts target main 402bd8815dcd8857d416f47b1b41fdb922cf471d. Neither depends on
the other. The only common changed path is root conftest.py. The modules use distinct
names, runners, test files, output directories and evidence packets. Their model
architectures and checkpoint formats differ and have not been interchanged.

Integration branch: codex/m224-synthetic-integration, created directly from main.
Implementation freeze: 7cd15f9. No branch merge was performed. The imported source
commits remain preserved by their original refs and the [preservation map](preservation.json).

Both original root routers and both workflow versions are preserved byte-for-byte in
[source-routing](source-routing). The live root router combines the two finite test
allowlists and adds three integration tests. The two live synthetic workflows call
scripts/verify_m224.py where the old commands assumed their own draft-specific router.
All other imported files, including every source report, result, freeze, configuration,
training module, runner, original verifier and test, remain byte-identical to the drafts.
Neither old frozen manifest was edited to bless the integration overlay.

The new verifier checks all 16 PR #62 snapshot entries and all 12 PR #63 entries at
their explicitly recorded locations, and every historical main file. It then invokes
the preserved M22.1 numeric validator/comparator. Original standalone verification
commands intentionally remain bound to their original routing files; use the integrated
command below on this branch. This replaces that routing assumption with whole-snapshot
byte verification, rather than claiming the live router has either original hash.

## Readiness matrix

| Gate | Status | Evidence and limits |
| --- | --- | --- |
| Historical main / model code preservation | PASS | All 953 main files match SHA256; M12/M17 and other historical code unchanged |
| M0–M21 branch evidence / archive | PASS | All 37 pre-existing local refs unchanged, including M21.1 and archive/m203-interrupted at 808441a80488cb4a633bc4329a096929df5cd521; branch-only evidence was not imported or altered |
| Original draft evidence | PASS | All 28 snapshot entries verified; routing/workflow originals preserved separately; original reports/results/frozen manifests unchanged |
| Combined test collection | PASS | 47 tests pass together, including fail-closed scope and tampered routing-archive controls; historical tests not imported on dedicated-branch CI |
| M22.1 full/prefix/resume/disk replay | PASS | All 20 prespecified scenario/seed pairs have exact local replay, 12-epoch prefixes and 12+108 continuation; endpoint difference versus reference is 0 locally |
| M22.3 exact continuation | PASS | All 3 seeds have identical uninterrupted/resumed/fresh states; fresh-process disk continuation also tested; 1,200 endpoint updates |
| Rejection / noninterference controls | PASS | Optimizer reset, corruption, schema/dtype/shape/input identity and held-out controls retained; 22 controls pass under optimized Python |
| Portable comparison | PASS locally | Original tolerances unchanged: endpoint rtol=1e-8/atol=1e-9; M22.3 raw fixtures/preprocessing rtol=0/atol=1e-14; remote CI compares fresh runs separately from exact local resume |
| Ruff / packaging / static integrity | PASS | Check and format pass; wheel contains both modules and no arrays/weights; release-candidate/readability/M18.4 static integrity checks pass |
| One shared strict checkpoint API | PARTIAL | Deliberately not implemented: M22.1 retains its original assertion-based checks and format; M22.3's envelope has not been applied to M22.1's architecture |

No required local integration gate failed. The PARTIAL row limits the scope of the
claim; coexistence does not establish a common trainer or checkpoint compatibility.
M22.1 evidence validation requires ordinary Python and the integration verifier rejects
optimized execution of that legacy validator. M22.3 runtime rejection uses explicit
exceptions and is tested under -O. No historical weakness was silently rewritten.

[verification.json](verification.json) records the checks, wheel hash and replay hashes.
[m221-replay.json](m221-replay.json) and [m223-replay.json](m223-replay.json) contain fresh
synthetic-only integration runs. Checkpoint arrays remain ignored under runs/. The
legacy M22.1 runner records the integration import commit as the latest commit touching
its preregistration path; that metadata is not a new preregistration. Its original
protocol commit and chronology remain in the unchanged source evidence.

The local build frontend was unavailable; `pip wheel --no-deps --no-build-isolation`
succeeded using the installed backend. CI additionally performs the unchanged standard
wheel build/install check. No package was released or uploaded.

## Reproduction

Use new output directories on every replay. Existing checkpoints are never overwritten.

```sh
pytest -q tests/test_m221_synthetic.py tests/test_m223_checkpoint.py tests/test_m224_integration.py
python -O -m pytest -q tests/test_m223_checkpoint.py -k 'rejection or corrupted or changed_input or semantic or optimizer_reset'
ruff check .
ruff format --check .
python scripts/run_m221.py --output runs/m221/integration-reproduction
python scripts/verify_m224.py --m221-compare runs/m221/integration-reproduction/results.json
python scripts/run_m223.py --output runs/m223/integration-reproduction --compare benchmarks/m223/results.json
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/m224-wheel-reproduction
```

On the original checkout, add `--local-refs` to verify all original local branch refs.
CI checks the full historical/source byte inventory without requiring the local-only
archive to be pushed. Existing historical workflows remain unchanged. The dedicated
integration router aborts on any non-additive historical or unrelated change before
excluding historical tests. These commands read only artificial numeric outcomes;
no historical training replay or biological data acquisition is part of the milestone.

## One bounded methodological next step

Preregister one synthetic-only checkpoint-substitution bridge experiment: use exactly
the two existing M22.1 scenarios and seeds 0–9, with its architecture, initialization,
optimizer, preprocessing and schedules fixed, and test whether an M22.3-style envelope
preserves the same epoch-12/120 outputs and exact 12+108 continuation. Acceptance must
be exact same-environment state/output equivalence, with no tuning or scenario changes.
This is a recommendation only; no bridge implementation or additional training was run.
