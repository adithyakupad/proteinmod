# Reproduction and working inference

Python3.11–3.14 offline core; measured full neural environment Python3.12/macOS
arm64/CPU. From a full Git clone (history is required for provenance guards):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-m3.txt -e '.[dev,esm,m3,structure]'
LOKY_MAX_CPU_COUNT=4 pytest -q
ruff check .
ruff format --check .
python -m pip check
python scripts/verify_readability.py --historical
python scripts/verify_m184.py
python scripts/verify_sod1.py
python scripts/verify_release_candidate.py --output /tmp/proteinmod-integrity-new.json
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/proteinmod-wheel-new .
```

Use fresh destinations; evidence exporters refuse overwrite. Setuptools/wheel must
be installed for --no-build-isolation; omit that flag for normal isolated builds
with network access. The resulting wheel contains code and licenses, no benchmarks,
model weights or datasets. Command-line benchmark workflows require the repository
configs and explicitly acquired sources; a wheel alone is not the evidence archive.

## Cached selected-state replay, no authentic-data training

Acquire/prepare pinned M12/M14/M15 assets using docs/M12.md, docs/M14.md and docs/M15.md.
The original M17 and M19 selected-state caches must also be present at recorded paths.
Binary files are intentionally absent from Git; verify hashes before loading.

```sh
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 python scripts/replay_readability.py --output runs/release-replay-new --m17 --m15-fold
LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 python scripts/verify_release_candidate.py --replay-m19 --output runs/release-m19-new.json
```

The first command also exercises synthetic kernels/gradients and a tiny synthetic
training fixture; it does not retrain an authentic-data benchmark sweep. M19 replay
loads existing states only. If historical weights are absent, report unavailable;
do not call a newly trained replacement an exact checkpoint replay.
Offline tests reconstruct archived metrics without model or data downloads.

## Real bounded single/multiple inference

With pinned M12/M15 sources, frozen/prefix caches and selected M15 G3 checkpoint:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 python scripts/infer_m16.py --protein BBC1 --mutations P56N
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 python scripts/infer_m16.py --protein BBC1 --mutations 'D45S;N47R'
```

Actual outputs0.05477935075759888 and0.5626708269119263 respectively, from checkpoint
ce9c93d404dcfa7180162c4950110837b5f36f1fea4636e569ecab6c72145bd2.
See inference_examples.json for protein/assay/reference SHA and PDB1TG0 mapping
provenance. This LoRA path is sequence-only; the PDB identifies the benchmark
construct, not a structural input to that checkpoint. Outputs are unbounded
train-assay rank predictions, **not kcal/mol**, and have no valid M13 interval.
Unknown proteins/unvalidated variants and duplicate mutations are rejected.
The tool is not a universal arbitrary-protein inference API.

## Fail-closed thermodynamic output

```sh
python -m proteinmod.stability_inference --output runs/safe-stability-new.json
```

All13848 archived M8 diagnostic records export prediction=null, unit=null,
thermodynamic_eligible=false, retaining source/checkpoint identity. No clipping,
unit relabeling or new thermodynamic validity is introduced. Missing provenance,
wrong source checksum or an existing destination rejects export.

## Cold-source boundary

M0 quickstart in README explicitly downloads one FLIP2 file. M16.1 previously
regenerated M0/M1 features from an empty cache on CPU/MPS and downloaded/rebuilt real
structure/native adapter sources; see benchmarks/m161. Other milestones retain
individual acquisition instructions and cached-state dependence. Complete cold
M0–M19 reproduction is NOT verified. No external SOD1/GEO phenotype acquisition is
part of these commands; maintain the external firewall and original admission gates.
