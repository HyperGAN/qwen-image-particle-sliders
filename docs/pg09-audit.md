# Native ParticleGAN 0.9.1 audit

The public training CLI now calls installed `ConditionalPG09` with the shared
`QwenImage21ResponseAdapter` and conditional readout critic. The original Qwen main
was verified as `b418ff5b709dd687fdc7d9fbb5498c592ec21a58`. The dependency is proposed
shared commit `1bfafe193005af2db91472268be3e555964f36e1` (core 0.6.0); its review and
publication are separate from this product draft. ParticleGAN is the latest official
released 0.9.1, tag commit `9fc8fb51cba27c26d61f2b6a7d139f55cc726c35`. No PG source
overlay is used for the native spatial response domain.

[Machine receipt](pg09-audit.json), [34-test JUnit](pg09-cpu-tests.xml), and
[fresh-process resume receipt](pg09-fresh-process.json) record exact source/wheel
hashes and results. Product and shared wheels were installed into an isolated
consumer environment, with native Diffusers built at its retained source pin.
Tests ran outside the checkout under `python -I`, with `pythonpath=''`; all Qwen,
core, PG, Diffusers and Transformers imports resolved to that environment. Existing
PyTorch/NumPy dependencies were reused. This is source-free package installation,
not a claim of complete fresh dependency resolution or a new Python-version matrix.
Every installed product Python/JSON file and shared Python file matched its tested
source bytes. No engine was copied into the product.

All **34 tests passed**. Coverage includes native rectangular FP32/BF16 teacher
parity with differing text lengths and None/explicit masks; both PG presets; actual
particle identity and unique optimizer ownership; frozen base weights; balanced
sampling; repeated checkpoint backward with exact gradient parity after scale scopes
exit; strict fixed-budget resume; EMA inference, fractional positive strength and
exact zero bypass; atomic bad-tensor rejection; and cleanup after failed construction.

Public CLI runs in separate Python processes proved four uninterrupted updates
identical to two plus two resumed updates under both presets, including complete
checkpoint fingerprints and exact EMA safetensors bytes. The same installed CLI
performed native latent inference. A separate installed `--critic hndl` CLI completed
two native GAN updates. The old script name `check_cuda_resume.py --dummy` now
provides a CPU-only proof of the new recipe and never claims a new CUDA cap test.

The optional common HNDL critic executed its released 0.8.2 graph, native GAN updates
and strict restore. The common HyperGAN develop diversity observer consumed actual
Qwen responses from two latent seeds in each of five fixed conditioning/time groups.
The detached observer preserved game and RNG state. Its variation measurements are
not a quality score or a model selection gate.

The new v2 artifact/response protocol rejects historical gmix exports/checkpoints.
The old model, Diffusers, processor, scheduler and research-license pins are retained;
old training source pins are preserved in the lock's `legacy_training` record and
all active checkouts/environments remain untouched. Historical CUDA receipts do not
qualify the new objective. No pretrained model was loaded, no CUDA context was
initialized, and no GPU job, service, permission, model publication or merge occurred.
Pretrained memory, complete sampler/decode behavior, image quality and convergence
remain admission/qualification gates for the coordinator. Parent routes exact heads
through Ada before merging.

Reproduce after installing the pinned package dependencies:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
python scripts/check_cuda_resume.py --dummy --preset bispf --outdir /tmp/qwen-resume-bispf
python scripts/check_cuda_resume.py --dummy --preset bispf_optimized --outdir /tmp/qwen-resume-optimized
qwen-slider-train --dummy --steps 2 --critic hndl --outdir /tmp/qwen-hndl
```
