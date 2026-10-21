# M22.6 — final synthetic methodology integration

**PASS for the final synthetic integration candidate.** Reuse draft
[PR #65](https://github.com/adithyakupad/proteinmod/pull/65), retargeted to main,
as the single candidate. Its original source head already contains PR #64 and the
M22.5 bridge; a new implementation branch or PR would duplicate that integration.
Only this report and opaque verification inventories are added for M22.6. No trainer,
configuration, test, checkpoint implementation, collection plugin or workflow changes.
The report lives under the existing M22.5 evidence namespace so its frozen additive
CI policy remains valid without introducing another router or exception.

## Exact heads and dependency audit

All four source drafts were open, unmerged and draft when inspected. Main was
402bd8815dcd8857d416f47b1b41fdb922cf471d, matching the local baseline.

| PR | Verified source head | Base at inspection | Dependency / content |
| --- | --- | --- | --- |
| [#62](https://github.com/adithyakupad/proteinmod/pull/62) | 2e82ff477979fffa398ae778ba2bac1162e24205 | main | Independent M22.1 horizon diagnostic |
| [#63](https://github.com/adithyakupad/proteinmod/pull/63) | df58fe0dbeb0df772465533659a8c5c7b3a9fd78 | main | Independent M22.3 checkpoint fixture |
| [#64](https://github.com/adithyakupad/proteinmod/pull/64) | 3482d76961115078e97a1489c0d80f3325da8402 | main | Imports both prior snapshots; resolves the one root-router overlap |
| [#65](https://github.com/adithyakupad/proteinmod/pull/65) | 5f69fbf3450aaf993e8f0c6955ef7161d2af7924 | #64 branch at 3482d76961115078e97a1489c0d80f3325da8402 | Direct Git descendant of #64; adds preregistered envelope substitution |

#62/#63 are source-snapshot dependencies of #64, not merged Git parents. Their exact
original routing files are preserved in #64's source-routing packet. #65 already has
all #64 files unchanged plus its own additive work. Retargeting #65 to main exposes
that complete content for one review and requires no prior merge of #62, #63 or #64.
Those drafts remain unchanged and open. #65's candidate head advances only for the
three M22.6 documentation/verification additions; its original 5f69fbf source commit
and every file from that commit remain preserved. No rebase, squash or branch merge.

## Verification matrix

| Gate | Decision | Final local evidence |
| --- | --- | --- |
| Original protocols/reports/results | PASS | All 1,008 files at original #65 head checked against exact Git blob SHA256; no original evidence rewritten |
| Historical code and M0–M21 evidence | PASS | All 953 integrated-main files and the full parent inventories retained; 38 non-candidate local refs unchanged, including M21.1 and local archive/m203-interrupted at 808441a80488cb4a633bc4329a096929df5cd521 |
| Synthetic tests | PASS | All 77 existing tests pass together on the original branch-specific finite CI route |
| M22.1 120 vs 12+108 replay | PASS | Both original scenarios, seeds 0–9: 20 exact replay/prefix/disk/resume cases; original verifier/reference metric difference 0 locally |
| M22.3 replay/rejection | PASS | All 3 registered seeds have identical uninterrupted/resumed/repeated states; original dtype/shape/schema/corruption/identity/optimizer and held-out controls pass |
| M22.5 checkpoint substitution | PASS | Both M22.1 scenarios, seeds 0–9: 20 exact bridge cases; all model/optimizer/RNG/metadata and endpoint metrics equal |
| Optimized-Python rejection | PASS | 22 M22.3 plus 24 M22.5 existing controls pass under -O; runtime guards remain explicit exceptions |
| Ruff / packaging | PASS | Check and format pass; wheel builds with all three synthetic modules and no arrays/weights/data exports; unchanged release integrity verifier passes |
| Historical static verification | PASS | Original readability, M18.4, M22.4 and M22.5 integrity checks pass; no historical experiment replay |
| Applicable remote CI | PASS criterion | Existing synthetic, bridge, integrity and packaging workflows run on the final candidate; status and logs are linked from #65 |
| Universal checkpoint/trainer unification | PARTIAL / outside scope | Original formats remain separate; M22.5 proves substitution only for the exact frozen M22.1 fixtures, not arbitrary models/environments |

The final candidate passes its required local gates; no FAIL was observed and no
threshold was relaxed. Remote CI must also be green at the latest candidate head
before any later human merge decision. The PARTIAL row limits the claim and is not
an unmet prerequisite for these frozen synthetic fixtures.

[verification.json](verification.json) records fresh replay paths/hashes and checks.
[preservation.json](preservation.json) records the exact source-file inventory, source
heads and original non-candidate refs. Replays use the already registered artificial
numeric fixtures only. They are verification of prior methods, not new training
experiments, scenarios, models, tuning or methodology development. New checkpoints
and raw replay output remain in ignored runs/; original committed result files stay
unchanged. The fresh replay metrics match the local prior reference with difference 0.
The normal pytest warning under -O does not disable production exception gates.

## Existing infrastructure retained

The original M22.1 training implementation, independent M22.3 fixture and M22.5
serialization wrapper remain exactly as reviewed. The wrapper calls M22.1's original
fit function; it does not duplicate or replace its optimizer/training loop. The existing
pytest plugin provides synthetic-only collection on #65's branch and rejects parent
or unrelated modifications before suppressing historical test collection. Its narrow
policy is why retaining #65 avoids a new collection layer and repeated infrastructure.

All original preregistration chronology, numerical failures, license records and
blinding records remain preserved. The archived restricted investigation remains
local and untouched. Opaque integrity hashing does not resume or inspect its outcomes.

## Reproduction

These are only existing registered numeric replays; use new output directories.

```sh
pytest -q tests/test_m221_synthetic.py tests/test_m223_checkpoint.py tests/test_m224_integration.py tests/test_m225_bridge.py
python -O -m pytest -q tests/test_m223_checkpoint.py -k 'rejection or corrupted or changed_input or semantic or optimizer_reset'
python -O -m pytest -q tests/test_m225_bridge.py -k 'rejection or changed_identity or corruption or optimizer_reset'
ruff check .
ruff format --check .
python scripts/run_m221.py --output runs/m221/final-reproduction
python scripts/verify_m224.py --m221-compare runs/m221/final-reproduction/results.json
python scripts/run_m223.py --output runs/m223/final-reproduction --compare benchmarks/m223/results.json
python scripts/run_m225.py --output runs/m225/final-reproduction --compare benchmarks/m221/results.json
python scripts/verify_m225.py --result runs/m225/final-reproduction/results.json
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/m226-wheel-reproduction
```

Add --local-refs to the integrity verifiers on the original checkout. CI does not
require uploading the local archive. Fresh-process and exact same-environment replay
claims remain separate from the unchanged portable endpoint tolerances. No universal
cross-platform checkpoint compatibility, convergence explanation, model-quality or
biological-generalization conclusion follows from this candidate. No further experiment
is proposed or executed here. No biological data/model fitting, mutation prediction,
variant generation, automatic merge, tag or release is part of M22.6.
