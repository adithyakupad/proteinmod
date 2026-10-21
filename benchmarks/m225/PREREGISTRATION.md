# M22.5 — prospective synthetic checkpoint-substitution bridge v1

Parent: PR #64 at 3482d76961115078e97a1489c0d80f3325da8402. This registration
must be committed before new bridge implementation/tests or bridge numerical execution.
All parent files, standalone implementations and historical refs remain unchanged.

Use the exact configs/m221_synthetic.json bytes (SHA256
8c5a97563a8f13e5dbd80c88095ced96af28f8fc9ec5fce6e2cdd2670d66c231):
ample_clean and scarce_noisy, seeds 0–9, 16→64 tanh→1 float64 MLP, original Glorot
initialization, Adam .003/.9/.999/1e-8, batch size 32, training-only mean/population SD,
no target scaling, no dropout/scheduler/early stopping, fixed epochs 12 and 120.
Data generation and optimization use the unmodified M22.1 functions. No new scenario,
seed replacement, hyperparameter change, architecture search or biological data.

For every one of the 20 registered pairs:
1. Run original uninterrupted 120 with its epoch-12 snapshot, plus an independent 12
   prefix. Require prefix equality in every model/optimizer/RNG/metadata field.
2. Encode the 12 prefix in a new versioned, bounded, non-pickle fail-closed envelope,
   save without overwriting, reload with a trusted external file SHA256 and expected
   training-only contract, and continue original fit for exactly 108 more epochs.
3. Require exact model arrays, Adam moments/counter, full PCG64 order state,
   preprocessing, complete training-loss trace and original snapshot metadata at 120.
   Require identical train/validation/test MSE and Spearman at fixed 12/120. Also
   round-trip the terminal checkpoint. Compare with independent original reruns.
4. Record state/file/source/config/input/schedule hashes, counters, environment and
   exact per-component comparisons; report every seed without replacements.

The envelope binds exact config, scenario/seed, ordered training data dtype/shape/hash,
preprocessing, source implementation and environment identity. Record the complete
120-epoch original M22.1 permutation/chunk schedule and expected order-RNG states;
never substitute M22.3's grouped sampler. Accept only completed epochs 12 and 120.
Capture all original state atomically in one logical snapshot; no best-checkpoint restore.
No evaluation arrays enter the contract, serialization validation or continuation API.

Controls: corrupted/truncated/checksum-invalid/duplicate-key checkpoints; unsupported
schema/version; unknown/missing keys; dtype/shape/coercion/nonfinite arrays; changed
training values/order/config/scenario/seed/preprocessing/source/environment; inconsistent
optimizer counter or order RNG; negative second moments; tampered schedule; held-out
features/targets replaced adversarially. Validate before calling fit. Exercise rejection
under Python -O using explicit exceptions, plus cold-process continuation.
A reset optimizer must be rejected as inconsistent. Checksums do not authenticate a
caller who also controls the trusted external hash; document that boundary.

PASS requires every mandatory exact comparison for all 20 pairs and all controls to
pass, with focused software tests, Ruff, preservation verification and relevant CI.
Otherwise report FAIL and retain the failed cases. No numerical tolerance may rescue
a local substitution difference. Paired resumed-minus-full endpoint effects must be
zero for all seeds; report mean, sample SD and conditional degenerate interval [0,0]
only as engineering fixtures, not a population/generalization inference.

Exact equivalence is within each captured Python/NumPy/BLAS/thread environment. CI
must independently obtain exact original-versus-bridge equality on its platform.
Cross-platform bitwise checkpoint compatibility is unsupported and environment
mismatches are rejected. Existing M22.1 fresh-run endpoint comparisons may use its
original rtol=1e-8/atol=1e-9 separately; they never override the local exact gate.

Original standalone files, manifests, results and PR #64 remain preserved. New results
live only in benchmarks/m225 and ignored runs/m225. No biological fitting, outcomes,
variant generation, protein optimization, tuning, merge, tag or release is authorized.
