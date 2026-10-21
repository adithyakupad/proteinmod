# M22.1: deterministic synthetic regression at epochs 12 and 120

**Both prespecified scenarios behaved as expected for MSE.** Longer training improved generalization with plentiful, low-noise synthetic examples; with few noisy examples, training error fell while validation/test MSE rose. The latter scenario's rank-correlation change was inconclusive. This is a non-biological software/optimization demonstration, not an explanation or independent validation of any protein-model result.

## Registration and scope

Protocol: commit `6413046`, before the first synthetic trajectory. Implementation: `b96c8d8`, with static-only unused-import cleanup `034a6e6` before execution. [PREREGISTRATION.md](PREREGISTRATION.md), [preregistration.json](preregistration.json) and [implementation-freeze.json](implementation-freeze.json) preserve hashes. Import ordering/unused-import issues found by Ruff were corrected before any numerical run; neither the registered scenarios nor numerical settings changed after results.

Base is integrated main `402bd8815dcd8857d416f47b1b41fdb922cf471d`. Work is additive on `codex/m221-synthetic-horizon`; M21.1 remains on its original branch and commit. No historical model/checkpoint, biological input, protein sequence, phenotype table, variant generation, biological optimization, architecture search, dataset acquisition or release is used. The implementation imports NumPy/SciPy and stdlib only, plus the package's inert version initializer. Raw synthetic arrays and binary weights are not committed.

## Fixed experiment

One fully trainable MLP: 16 real inputs → 64 tanh units → one linear scalar; **1,153 parameters**. Glorot-uniform weights, zero biases, float64. Adam learning rate .003, betas .9/.999, epsilon 1e-8, batch32. No weight decay, scheduler, gradient clipping or early stopping. Feature mean/population SD are fitted on training inputs only; targets are not transformed.

Inputs are independent standard-normal real arrays. The artificial regression function is `sin(x0) + .5*x1 + independent normal noise`. The following two scenarios were fixed before execution:

| Scenario | Training rows | Validation / test rows | Noise SD | Registered expectation |
| --- | ---: | ---: | ---: | --- |
| ample_clean | 512 | 256 / 2,048 | .05 | Longer training lowers test MSE |
| scarce_noisy | 32 | 256 / 2,048 | 3 | Training MSE falls while test MSE rises |

Within a scenario, all three splits have the same generating distribution; no artificial train/test distribution shift is introduced. Scenarios jointly change sample size and noise and are not a factorial test separating their causal effects.

Ten fixed seeds independently regenerate data, initialization and minibatch order in separate PCG64 namespaces. Initialization is identical between scenarios at the same seed. For each seed/scenario, a single trajectory reaches120, retaining fixed checkpoints12 and120. Both endpoints share initialization, preprocessing, examples and the first12 epochs. Validation/test evaluation happens only after training has completed. Neither evaluation nor any validation-selected best checkpoint controls optimization.

The training trace reports full-training-set MSE at each frozen epoch state, not online minibatch loss. The primary comparison is paired MSE120 − MSE12 per seed, separately in each scenario. Rank correlations use average ranks for ties; undefined correlations remain null. Positive rank differences mean improvement, whereas negative MSE differences mean improvement.

## Results

Means across the ten seeds; all individual split/epoch metrics and paired effects are retained in [results.json](results.json).

| Scenario / split | MSE, epoch12 | MSE, epoch120 | Spearman, epoch12 | Spearman, epoch120 |
| --- | ---: | ---: | ---: | ---: |
| ample_clean — train | .047065 | .001490 | .970451 | .998774 |
| ample_clean — validation | .072486 | .021767 | .957333 | .982853 |
| ample_clean — test | .071349 | .022101 | .959712 | .984507 |
| scarce_noisy — train | 5.426309 | .060595 | .618365 | .992375 |
| scarce_noisy — validation | 10.179026 | 18.252425 | .072327 | .048488 |
| scarce_noisy — test | 10.765453 | 18.670595 | .077077 | .051561 |

Paired test effects, always epoch120 minus epoch12:

| Scenario / metric | Mean difference | Seed sample SD | Nominal 95% paired t interval | Seed directions |
| --- | ---: | ---: | --- | --- |
| ample_clean MSE | −.049248 | .004838 | [−.052708, −.045787] | 10/10 improve |
| ample_clean Spearman | +.024795 | .004053 | [.021896, .027694] | 10/10 improve |
| scarce_noisy MSE | +7.905142 | 3.457627 | [5.431705, 10.378579] | 10/10 worsen |
| scarce_noisy Spearman | −.025516 | .074634 | [−.078906, .027874] | 6 worsen, 4 improve |

Both registered MSE expectations pass: the stipulated direction holds for at least8/10 seeds and its mean paired interval excludes zero. The noisy scenario additionally lowers mean training MSE by5.365715. Validation MSE changes agree with the respective test directions in all seeds, but validation was never used to select the horizon.

The noisy scenario demonstrates MSE overfitting, **not a consistent rank-correlation deterioration**. Its nominal rank interval includes zero, and four seeds improve rank correlation. Squared-error and ranking conclusions must remain distinct. The clean result compares only the declared horizons; it does not establish convergence, an optimal epoch or that still longer training would continue improving.

## Replay, leakage and uncertainty

For all20 seed/scenario pairs:

- An independently initialized full rerun reproduces checkpoint state digests and split metrics exactly in the same environment.
- A separate12-epoch prefix exactly matches the12-epoch checkpoint of the120-epoch trajectory.
- Both checkpoints survive a no-pickle disk roundtrip without state changes.
- Restoring epoch12 parameters, both Adam moments, update count, scaler and batch RNG and continuing to120 exactly reproduces the uninterrupted state.

The focused tests independently check analytic gradients against finite differences; changed training/config/seed resume identities are rejected. An adversarial held-out-input/label test leaves training, scalers and checkpoint digests unchanged. The training API accepts only training arrays; evaluation cannot feed held-out information into gradients or preprocessing. The CI scope guard rejects modified historical or unrelated paths rather than silently omitting their tests. Nine focused tests pass; these are not claims that the historical biological test suites were rerun.

For each metric, compute seed-level paired differences first, then their mean, sample SD, SE and `mean ± t(.975, df9)*SE`. The ten seeds are the uncertainty units; thousands of test rows are not counted as independent seed replications. Seeds combine generator/sample, initialization and order variability. Intervals are conditional on the fixed artificial generator and model, nominal per metric/scenario, and not simultaneous multiple-comparison bounds. Neither biological generalization nor a population of architectures is identified.

The registered run took3.474 seconds locally including independent full/prefix/resume work, on Python3.12.15, NumPy2.5.4 and SciPy1.18.1, float64 with single-thread numerical environment settings. This is a small warm-environment diagnostic, not a hardware benchmark. Cross-platform CI uses precommitted metric tolerances `rtol=1e-8`, `atol=1e-9`, separate from strict same-environment replay. No universal bitwise portability claim is made.

## Preservation and CI

[Preservation results](preservation-result.json) confirm all953 main-tracked files and all991 files at the preserved M21.1 commit are byte-identical, with previous local branch references unchanged. The archive remains `808441a80488cb4a633bc4329a096929df5cd521`. The historical readability and M18.4 integrity guards pass; no scientific runner is invoked by them.

All original workflow files are unchanged. The additive root `conftest.py` limits pytest collection only on the exact dedicated CI branch, and only after verifying that the diff from pinned main contains exclusively allowlisted additions. It collects only `test_m221_synthetic.py` there; normal local and other-branch collection is unchanged. Any old/unrelated file change aborts scoped CI. This prevents existing broad CI commands from importing historical model tests while allowing normal GitHub job reporting. The new `m221-synthetic` workflow reruns the complete registered synthetic diagnostic and compares all endpoint metrics with the committed evidence.

Local checks and CI status are recorded in [verification.json](verification.json). Protocol and source hashes are verified by `scripts/verify_m221.py`; result effects/intervals are independently recomputed from endpoint metrics. No old integrity guard is weakened or rewritten. No merges, tags or releases.

## Reproduction — synthetic only

From this repository's existing development environment:

```sh
python scripts/run_m221.py --output runs/m221/new-replay
python scripts/verify_m221.py --compare runs/m221/new-replay/results.json
pytest -q tests/test_m221_synthetic.py
ruff check .
ruff format --check .
```

The output directory must be new and located under ignored `runs/m221`; existing output is refused. The runner acquires nothing and reads only its fixed synthetic configuration and registration. It writes40 small checkpoint files plus JSON in that isolated ignored directory. Committed evidence contains only aggregate metrics, per-seed effects, training-loss traces, hashes and environment/protocol records.

**Conclusion:** a fixed neural training continuation can improve or worsen synthetic generalization depending on the declared regime. The paired trajectory design, exact replay and leakage checks make that observation inspectable. It does not settle why any historical protein model underperformed ridge and does not authorize a biological follow-up.
