# Validation evidence and limits

Validation on 2026-10-10 uses Transformers 5.17.0, Diffusers
`1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458`, core
`20fdf98d25455582e7e7191eef8ef51cf73173ae` and ParticleGAN
`82b72661c03377f348ccb76de37cd4ca02d45a8d`. CI independently installs dependencies
from their public pins. The machine-readable real-model receipt is
[docs/real-model-smoke.json](docs/real-model-smoke.json).

## Native CPU evidence

All **21 tests pass** on Python 3.12.15, PyTorch 2.14.1 CPU and torchvision 0.29.1 CPU:

- Exact projection targets, zero-initialized residuals, strength-zero bypass, cleanup,
  rejection of other backbones and invalid strengths, frozen base parameters.
- Native pipeline neutral trajectories, matched positive teacher velocities, identical
  controls and target token suffix handling.
- Variable-length prompt microbatches preserve native RoPE layouts; adding text
  padding changes image positions and is deliberately avoided.
- Shared D/G updates through actual hooks, finite gradients, cap on step four,
  base-weight immutability and native gradient checkpoint recomputation.
- BF16 native transformer with FP32 routed adapters and checkpointed backward.
- Positive-teacher error, independence from noise-shaped feature inputs, refreshed
  all-coordinate sketch coverage/moments, and calibration unaffected by added controls.
- Exact uninterrupted four-step versus two-plus-two CPU resume: every raw state field
  and exported safetensors bytes match, including optimizers, particles, EMA and RNG.
- EMA inference, exact native latent equivalence at strength zero, finite changed
  latents at strength one, strict metadata/tensor rejection, portable configuration identity.
- Actual tiny randomly initialized native VAE/pipeline decoding to RGBA, with an exact
  four-channel PNG round-trip and a non-opaque alpha channel.
- Native Qwen3-VL video processor construction without model files, covering the
  torchvision dependency needed by the official processor even for text-to-image.

Resource policy and snapshot file validation tests use synthetic reports and a
**mocked miniature manifest**. The CUDA placement options test is also mocked.
These unit checks are separate from the actual full-snapshot smoke below. The tiny
pipeline's processor is an initialization stub and its text states are synthetic.
No image-quality evidence is claimed for random fixtures.

Separate CLI smokes run four CPU training steps with batch one, and four with batch
two plus gradient checkpointing. Losses, noise and VIC remain finite. Inference writes
synthetic latent tensors. Ruff lint/format checks and wheel/sdist packaging pass.
The revised installed wheel also trains/infers outside the source checkout.
CI repeats native/unit tests and exercises the installed wheel.

## Official-weight CUDA smoke

The restricted task view cannot access CUDA; this is an execution boundary, not a
host driver failure. The normal approved host execution route exposes two working
RTX A6000s. Existing GPU jobs were not stopped or modified. Before the smoke, the
selected GPU had about 39.5 GiB free and host RAM had about 25 GiB available.
Storage had about 356 GiB free.

A pre-existing shared Hub snapshot was found at cache revision
`b3179ad355be050328e483a9dfdd9e60cd62adfa`. **Every one of the 25 required files,
including all seven weight files, matches the size and SHA256 of the pinned
`d26bb61231c349cf6b7896fa83353113880e1ba3` manifest.** No weights were downloaded.
The older cache directory name does not substitute for file verification.

The first real load exposed a missing torchvision dependency in Qwen3-VL processor
loading. It failed before GPU weight allocation. The dependency and matching-wheel
setup/CI instructions are fixed, with a native processor regression test.

The successful retry used Python 3.12.15, PyTorch **2.13.0+cu126**, torchvision
**0.28.0+cu126**, CUDA 12.6, and driver 580.173.02. Native `device_map="cuda"` and
`low_cpu_mem_usage=True` placed all three model components directly on the selected
GPU, avoiding full-pipeline CPU staging. The original 40 GiB host-RAM admission
policy was unnecessarily restrictive for this path and is replaced by a conservative
16 GiB policy. Admission policies remain unmeasured headroom rules for larger workloads.

The run used a 10-minute supervisor deadline, a 34 GiB process GPU allocation cap,
an 18 GiB process RSS guard, and self-abort below 4 GiB available host RAM. It finished
successfully in **324.13 seconds**, with **8.17 GiB peak process RSS**, **30.35 GiB peak
GPU allocations** and **30.45 GiB peak GPU reservations**. RSS was sampled every second;
GPU peaks are PyTorch allocator measurements, not total machine use.

- Real tokenizer/text encoder, transformer and VAE loaded from verified weights.
- Two edit-bearing pairs and one identical-prompt control, 128×128, batch one,
  two native trajectory steps, one retained state per pair and two shared-core
  updates with native gradient checkpointing; finite D/G losses, VIC and noise.
- All **2,956 checkpoint tensors** were checked finite, including optimizer states.
- All **32 native attention output projections** were targeted. EMA export and strict
  safetensors reload succeeded; controls were excluded from critic calibration.
- Strength-zero latents were bit-for-bit equal to an unadapted call on the same native
  pipeline, prompt, seed and settings, with maximum absolute error **0.0**.
- Strengths zero and one decoded and reopened as **128×128 RGBA PNGs**, with alpha
  extrema **0–255**. There was no RGB conversion or alpha flattening.

The two-update smoke adapter and low-step images are local test artifacts, not released
trained sliders or evidence of useful semantic control.

## Remaining validation

512×512 or larger full-model memory use, exact CUDA resume, full-budget convergence,
held-out semantic direction, identity/composition preservation, and interpolation
quality remain open. The smoke did not reach the shared cap penalty's lazy fourth
step; that path is covered by native CPU tests, not by a full-weight CUDA test.
Follow [the bounded acceptance procedure](docs/real-model-validation.md) with fresh
resource checks before extending validation. CI success establishes source/package
integration, not trained-slider quality or broad hardware fit.
