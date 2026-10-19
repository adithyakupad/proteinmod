# Readability refactor contract

Baseline: `8a07008ba427cb9cd82c84b5508a1326b562afa0`, branch
`codex/m18-3-masked-structural-admission`; initial working tree clean. Work is on
`codex/readability-refactor`, never merged into main. BASELINE.json inventories
all 821 tracked files before edits, including 158 Python files, all configurations,
reports, checkpoint schemas, source hashes and benchmark evidence.

Baseline validation: 215 tests passed in 9. 36 seconds; Ruff check and format passed.
M16, M16. 1, M17, M18, M18. 1, M18. 2 and M18. 3 offline guards passed. Existing M17
records contain 1,096 fits / 34,304 predictions; no external outcomes accessed.

Stages:
1. Data and structural geometry: descriptive internal names while retaining
   mutation indexing, mapping status, graph edges, array order, dtype and masks.
2. Model kernels: reveal message passing, attention, task masking and LoRA tensor
   flow without changing public argument names, registered attributes, parameter
   construction/order, state_dict keys, operation order or RNG consumption.
3. Experiment/calibration/report utilities: clarify local state and indices;
   preserve train-only scaling, selections, roles, loops, exception paths and
   serialization fields. Audit tests/scripts; leave clear or historical code alone.

Every change is declared in refactor_plan.json and checked against baseline ASTs.
Local renames cannot collide with existing bindings or public parameters. Small
expression expansions must inline back to the exact original AST. This is a
restricted transformation check, not a general proof of Python equivalence.
Differential execution, model gradients/state restoration, tests and local replay
supply additional evidence. No numerical reassociation, sorted-order changes,
new calls to randomness, altered model layers, or changes to biological gates.

Cache-identity-sensitive embeddings.py, residue_cache.py and lora_data.py remain
byte-identical; cache schemas, scientific JSON/CSV fields, checkpoints, public APIs
and previous source hashes are never rewritten. Original historical verifiers
remain byte-identical. Because those verifiers intentionally pin live source
bytes, run them against the frozen revision in a temporary Git checkout; run the
new refactor verifier against current source and protected files. Running an old
live-source hash guard directly on intentionally refactored files should fail.
That expected difference must be explicit in CI/docs, never silently bypassed.

All benchmark/configuration/data-fixture/scientific-report files stay byte-identical.
M18 outcomes remain blinded. No training on external data, final admission
manifest, new scientific claim, architecture, tag or release. Full historical
retraining and independent cold-source reproduction will only be claimed if run;
otherwise report limitations. Preserve M17 scaling and degree-null definitions,
M15 LoRA gradients/weights and M18 failed admission records.
