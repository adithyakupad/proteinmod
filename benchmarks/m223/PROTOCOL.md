# M22.3 synthetic checkpoint protocol v1

Standalone infrastructure milestone based on main `402bd8815dcd8857d416f47b1b41fdb922cf471d`.
M22.1 remains a separate draft. No historical trainer or research evidence is changed.
M22.2 was a read-only audit with no new repository artifact. The scope is artificial
numeric arrays, with no biological inputs or scientific generalization claim.

Prespecified replay: seeds 0, 1, 2; 59 training rows, 6 independent normal features;
synthetic target sin(x0) + 0.5*x1 + normal noise SD 0.1. Additional 100 independent
normal rows have noiseless targets and are split into 50 validation and 50 test rows.
These are evaluation-only and cannot enter Contract or train. No checkpoint selection.
The fixed 6→12 tanh→1 float64 MLP uses Adam, learning rate .003, betas .9/.999,
epsilon 1e-8; no weight decay, dropout, scheduler, early stopping or tuning.
Training-only population mean/SD preprocessing (zero SD replaced by 1).

Groups a/b/c contain 17/33/9 rows. Within each epoch: sorted group traversal,
permutation within each group, chunks of 8 including remainders, then batch-list
shuffle. Exactly 10 batches per epoch, including three singleton remainders.
PCG64 streams use SeedSequence([223, seed, namespace]), namespaces 1=data,
2=initialization, 3=ordering. No global NumPy or Python RNG is used by training.
The complete 120-epoch plan and its SHA256 are part of the checkpoint identity.
The replay cursor is the completed epoch; the recorded order RNG must equal that
boundary's expected state. Mid-epoch continuation is unsupported and rejected.

Compare fresh uninterrupted 120, independent 12 prefix saved to disk and reloaded
with a trusted external file SHA256 then continued to 120, and another fresh 120.
Require byte-identical canonical states (all parameters, moments, RNG, counters,
preprocessing and identity), and identical train/validation/test MSE and Spearman.
Three paired seed effects must be exactly zero; report mean/sample SD, without
interpreting three engineering fixtures as an inferential scientific sample.

Checkpoint v1 is bounded JSON, not pickle. An envelope payload SHA256 detects internal
corruption, but is not authentication: loading additionally requires a trusted
caller-side file SHA256 and reconstructed expected Contract. Reject unknown/missing
keys, duplicate JSON keys, versions, dtype/shape/coercion/nonfinite errors, negative
second moments, inconsistent counters/RNG, changed input/order/groups/config,
preprocessing, implementation and environment. No model is updated before validation.
A loader cannot detect an intentionally substituted internally consistent model when
the attacker also controls the trusted digest; digest custody is the caller's duty.
Write-once saves reject overwrites; interrupted/truncated writes fail loading.

Negative controls: reset Adam step/moments must be rejected as an inconsistent
checkpoint. Separately bypass validation in test code only to demonstrate fresh
optimizer continuation from epoch-12 weights diverges from correct continuation.
Adversarial held-out feature/target replacement must leave all training state identical.
Schema rejection is also exercised under optimized Python; runtime guards use explicit
exceptions and are never assertions. Rejection does not change the supplied snapshot.

Exact replay is promised only in the captured environment: Python/NumPy versions,
platform/machine, NumPy build and thread settings. Cross-platform checkpoint resume
is deliberately rejected. Fresh local 120/12+108 runs on CI independently require exact
replay; their endpoint MSE/Spearman may differ from the committed reference by
rtol=1e-8, atol=1e-9. Protocol/input/source identity must still match. This tolerance
is an engineering comparison policy, not evidence of universal numerical equivalence.

Historical files are verified as opaque SHA256 bytes. All additions must belong to
this milestone before dedicated-branch CI can exclude historical test collection;
any modification/deletion of an old main file aborts collection. No old trainer test,
research replay, acquisition or outcome analysis is authorized here. CI also runs
the unchanged static historical-integrity and code-only packaging checks.

## Portable-run clarification after initial CI

The initial CI runs passed all exact replay and rejection tests but stopped at the
fresh-run identity gate before comparing endpoints. That gate required byte-identical
floating-point input/preprocessing metadata across macOS and Linux, conflating the two
policies. The portable comparator now records all artificial input arrays and checks
them, plus preprocessing mean/scale, with **rtol=0, atol=1e-14**. It first verifies each
input hash against its recorded arrays. Source/config/seed/group/order/schedule identity
remains exact, and endpoint tolerance is unchanged (rtol=1e-8, atol=1e-9). Local checkpoint
loading still rejects any input/preprocessing hash change, including differences within
the portable tolerance. A dedicated test distinguishes these policies. No training code,
architecture, optimization setting, input generation formula, scenario or endpoint
threshold changed. The original freeze/run remains preserved in commits 894fb7c/80b98e2.
