# M22.3 — synthetic reproducible checkpoint infrastructure

Decision: **PASS for the tested synthetic-only infrastructure scope.** No historical
M12/M17 trainer has been changed or qualified for resumable training by this work.
No biological arrays, phenotypes, variant generation, protein fitting or optimization
were used. The fixed architecture is a numeric engineering fixture, not architecture search.

Implementation/protocol freeze: `894fb7ce65d27c12fbd651ef3cf090f34b56c1ac`. Results were recorded only after that
commit; the source hash map in [implementation-freeze.json](implementation-freeze.json)
remains unchanged. [PROTOCOL.md](PROTOCOL.md) defines the design and tolerance policy.
[execution.json](execution.json) records local checks; [results.json](results.json)
contains full identities, state/file SHA256s and train/validation/test MSE and Spearman.
The three small checkpoint files remain ignored under `runs/m223/frozen-run/` and can
be recreated with the commands below. No checkpoint or research dataset is packaged.

| Seed | Updates at epoch 120 | Full / disk-resumed / fresh repeat state | Epoch-12 test MSE | Epoch-120 test MSE | Epoch-120 test Spearman |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 1200 | identical | 0.0324196884 | 0.0152193911 | 0.9936614646 |
| 1 | 1200 | identical | 0.0221477604 | 0.0041463529 | 0.9966386555 |
| 2 | 1200 | identical | 0.0743262807 | 0.0194178688 | 0.9780072029 |

Each uninterrupted state is byte-identical under canonical serialization to its
12-epoch disk checkpoint plus 108-epoch continuation and independent fresh 120 run.
This includes model weights, Adam moments/step, RNG states, counters, preprocessing,
full grouped/ragged plan and identity. Prefixes have 120 updates; endpoints have 1,200.
Endpoint paired differences (resume minus uninterrupted) in every train/validation/test
MSE and Spearman are zero for every seed. Seed mean and sample SD are zero.
A t interval would degenerate to [0,0] for these fixtures; this does not estimate a
population of hardware, tasks or random seeds. These synthetic metric values carry
no biological or scientific generalization interpretation.

| Contract / control | Status | Evidence |
| --- | --- | --- |
| Versioned strict envelope; trusted file hash + internal payload hash | PASS | JSON duplicate/unknown/missing keys, corrupt/truncated data, wrong checksum and size limit tested |
| Model/optimizer/RNG/counter consistency | PASS | Exact 120 vs 12+108 for seeds 0–2, fresh reruns, disk round trip and fresh-process continuation |
| Grouped/ragged batching | PASS | 17/33/9 rows, batch 8; each epoch covers all 59 rows exactly once, groups never mix, three singleton batches retained |
| Input / config / preprocessing / source / environment binding | PASS | Changed values, row order, groups, seed, identity fields and mutated contract rejected before updates |
| dtype / shape / schema / finiteness | PASS | float32, wrong shapes, coercible strings/bools/null, nonfinite values, negative second moments and unsupported schema rejected |
| Optimizer reset | PASS | Reset Adam counter/moments rejected; test-only bypass with fresh Adam from epoch-12 weights produces a different trajectory |
| Held-out noninterference | PASS | Evaluation accepts separate arrays; adversarial held-out replacement changes no checkpoint or continuation state |
| Explicit runtime guards | PASS | 22 rejection/control tests under Python -O; an additional subprocess verifies unsupported-version rejection without assertions |
| Same-environment exact replay | PASS | Canonical state equality and metric equality; all source/config/input hashes fixed |
| Cross-platform exact resume | Unsupported | Environment identity mismatch is rejected; fresh-run metrics have separately declared rtol 1e-8 / atol 1e-9 |
| Historical trainer resumability | Not evaluated | M12/M17 unchanged; no historical training test was run |

All **34** focused tests pass, including **34** through the dedicated-branch default
CI collection route, and **22** under optimized Python. Ruff check and format pass.
The usual pytest warning under -O describes disabled ordinary assertions; production
validation uses explicit exceptions, and the dedicated subprocess confirms rejection.
A separate local fresh-run comparison has maximum metric difference **0**. Remote
CI runs the same exact replay locally and separately compares fresh-run endpoints
with this committed reference. Remote job outcomes are available on the draft PR.

All **953** historical main files remain byte-identical by SHA256, and all
**36** pre-existing local branch refs remain unchanged, including archive/m203-interrupted
at `808441a80488cb4a633bc4329a096929df5cd521`, M21.1 and M22.1.
[preservation.json](preservation.json) contains the opaque hash inventory and refs.
The historical readability and M18.4 integrity checks pass without training replay.
M22.2 was a read-only audit and created no tracked artifact. Existing research drafts
are unmerged. Only this new dedicated branch is proposed for review.

Reproduction, from an environment with the declared project development dependencies:

```sh
pytest -q tests/test_m223_checkpoint.py
python -O -m pytest -q tests/test_m223_checkpoint.py -k 'rejection or corrupted or changed_input or semantic or optimizer_reset'
ruff check .
ruff format --check .
python scripts/check_m223_preservation.py
python scripts/run_m223.py --output runs/m223/reproduction --compare benchmarks/m223/results.json
```

Use a new output directory on each run; overwrites are rejected. On the original
checkout, `python scripts/check_m223_preservation.py --local-refs` additionally verifies
all pre-existing local refs. CI verifies historical file bytes without requiring the
local-only archive to be uploaded. The dedicated-branch collection guard permits only
additions in the explicit milestone allowlist; any historical edit aborts collection
instead of suppressing its tests. Existing workflows remain byte-identical.

Limits: checkpoint compatibility is tied to this source and environment, not just
architecture names. Only completed epochs are supported. Global Python/NumPy RNG,
GPU state, dropout, schedulers, distributed workers and stochastic preprocessing are
outside this implementation because none are used. No cryptographic signature or
adversarial-process protection is claimed: a caller controlling both a fabricated
valid payload and its trusted digest can substitute arbitrary weights. Interrupted
writes are detected on loading rather than guaranteed atomic. Cross-platform tolerance
is a policy for tested fresh-run metrics, not a universal guarantee. This milestone
does not remedy, rerun or reinterpret historical biological experiments.
