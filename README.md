# Particle Sliders for Qwen-Image 2.1

**Built with Qwen.** A separate HyperGAN product for the official
[`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) checkpoint,
using `QwenImage21Pipeline` and the pinned shared
[`particle-sliders-core`](https://github.com/HyperGAN/particle-sliders/tree/20fdf98d25455582e7e7191eef8ef51cf73173ae/packages/particle-sliders-core).

The initial implementation supports text-to-image paired-prompt training, nonlinear
routed attention output adapters, shared particle inference at strengths **0–1**,
EMA safetensors export, and checkpoint/resume. Strength zero bypasses the hooks exactly.
The generator/critic update, noise, cap regularization and particle VIC come from
`winning_formulation()` and `FormulationGame`; there is no copied training engine.

**Status:** 21 CPU tests pass, and a bounded official-weight CUDA smoke passed at
128×128: training, native fourth-step cap backward, exact two-plus-two CUDA resume,
EMA reload, exact strength-zero latent equivalence, and native RGBA PNG output.
Full-resolution memory use, image quality and convergence remain unvalidated.
No pretrained slider weights are released.
See [VALIDATION.md](VALIDATION.md).
This product targets base 2.1, not older Qwen-Image, Edit/Layered, or 2.1 Turbo.
Image-conditioned training, other CFG values, ComfyUI and ordinary-LoRA distillation
are outside this initial release.

## Licensing

Qwen-Image 2.1 uses the **Qwen RESEARCH LICENSE AGREEMENT**, for research/evaluation
and non-commercial use. Commercial model usage requires a separate upstream license.
This differs from licenses of earlier Qwen models. Read
[QWEN-RESEARCH-LICENSE.txt](QWEN-RESEARCH-LICENSE.txt) and [NOTICE](NOTICE).
The original integration code is Apache-2.0; it does not relicense the model or trained
Qwen derivatives. Retain upstream terms, notices and “Built with Qwen” attribution
when distributing trained adapters. No base model weights are included.

## Install

Python 3.11 or 3.12. Install compatible PyTorch **>=2.6** and torchvision builds for
your machine first, then install the package (including exact core, ParticleGAN and
Diffusers source pins):

```bash
python -m venv .venv
source .venv/bin/activate
# CPU validation; use your selected CUDA wheel index for real model work.
python -m pip install 'torch==2.14.1' 'torchvision==0.29.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

The official model documents Transformers >=5.17 and Diffusers source support.
Qwen3-VL processor loading also requires torchvision, including its video processor
even for text-to-image use. Install torch/torchvision together from the same wheel
index for your CPU or CUDA runtime. This product tests Transformers 5.17.0 and pins
Diffusers to `1d5d056…`.
Source/model identities, weight sizes, file hashes and licenses are in
[backend.lock.json](backend.lock.json) and [PROVENANCE.md](PROVENANCE.md).

## Bounded, offline CPU validation

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  qwen-slider-train --dummy --steps 4 --outdir outputs/dummy
qwen-slider-infer --dummy --adapter outputs/dummy/adapter.safetensors \
  --prompt 'A ceramic cup on a wooden table.' --strength 1 \
  --out outputs/dummy/latents.pt
python -m pytest -q
```

`--dummy` uses the actual two-block `QwenImage21Transformer2DModel`, native
block-causal attention and native pipeline/Euler scheduler, with random weights and
synthetic masked text embeddings. Its processor is a minimal initialization stub;
real tokenization and Qwen3-VL encoding are not tested. A separate test adds an
actual tiny randomly initialized VAE and verifies the native RGBA/PNG output path.
The dummy CLI downloads nothing, runs on CPU, and caps training/inference to four
steps at 32×32.
Every export records `tiny-native-random-weights-synthetic-text`; real inference
rejects these artifacts. The `.pt` dummy output contains packed synthetic latents.

## Resources and official snapshots

Check resources before downloading or loading:

```bash
qwen-slider-preflight --cache-dir /ml2/qwen-sliders/cache --device cuda:0
```

The pinned snapshot has **33,115,613,408 weight bytes (30.84 GiB)**, plus processor
and configuration files. Initial conservative policies require 45 GiB free download
storage, 16 GiB available host RAM for a CUDA load, and 38 GiB free on the selected
GPU. A CPU FP32 full load requires 80 GiB available RAM. These are **unmeasured
headroom policies**, not fit claims. Activations, resolution and gradients can require
more. Shared GPU jobs must retain their own resources; the loader does not choose an
idle GPU, stop processes, quantize weights or download anything by default.
The bounded 128×128 checks measured up to 9.62 GiB peak process RSS and 30.45 GiB
reserved GPU memory; these measurements do not establish headroom for larger runs.

Stage the exact official snapshot in `/ml2/models/Qwen-Image-2.1`. The loader checks
all load-relevant file sizes and SHA256s, including every weight shard, before loading.
Extra loadable files are rejected to prevent an unverified variant being selected.
For an explicitly permitted pinned download, replace `--model-dir` with
`--allow-hub --cache-dir /ml2/qwen-sliders/cache`. Resource checks happen before download.
CUDA loading uses native `device_map="cuda"` and `low_cpu_mem_usage=True` inside the
selected device context, streaming shards directly to that GPU instead of first
staging the whole pipeline on CPU. The package's preflight command itself is offline
and downloads nothing. The [official model CPU-offload example](https://github.com/QwenLM/Qwen-Image-2.1#memory-optimization)
is another inference option, but its initial CPU staging and training hook behavior
are not part of this loader.

## Training

The example card trains warm light from three neutral/positive pairs plus two
identical-prompt controls. Edit-bearing records set critic calibration; controls
remain in training and cannot dilute that calibration. Customize
[configs/warm-light.json](configs/warm-light.json), retaining at least two pairs.

```bash
qwen-slider-train --model-dir /ml2/models/Qwen-Image-2.1 --device cuda:0 \
  --prompts configs/warm-light.json --height 512 --width 512 \
  --trajectory-steps 40 --records-per-pair 2 --batch 1 --steps 1600 \
  --gradient-checkpointing --outdir /ml2/qwen-sliders/warm-light
```

The native neutral trajectory supplies matched latent/time states; the positive
teacher is evaluated at those same states. Training and inference both use native
CFG=1 with KV caching disabled. Native packing uses unpatched 64-channel latents,
model time is scheduler time/1000, and dynamic timestep shifting follows the pinned
scheduler. Teachers are cached on CPU with at most 128 retained records.
See [FORMULATION.md](FORMULATION.md) for the error sketch and shared game contract.

`train.jsonl` can be tailed. `adapter.safetensors` and its JSON sidecar contain the
EMA routed maps/particles and strict provenance. `training.pt` contains raw maps,
particles, critic, optimizers, RNG, EMA and run configuration. Resume with the same
original options and a larger total `--steps`:

```bash
qwen-slider-train <same original options> --steps 2000 \
  --resume /ml2/qwen-sliders/warm-light/training.pt
```

Exact resume is tested on CPU and in one bounded 128×128 CUDA configuration with
PyTorch 2.13.0+cu126. Every checkpoint field, EMA artifact byte, step input and RNG
hash matched four uninterrupted updates versus two-plus-two resumed updates.
Device/runtime changes can alter numerics; this is not a guarantee across hardware
or processes with different settings. See the measured evidence in [VALIDATION.md](VALIDATION.md).
Resume files are loaded with `weights_only=True`; inference uses safetensors.

## Inference

```bash
qwen-slider-infer --model-dir /ml2/models/Qwen-Image-2.1 --device cuda:0 \
  --adapter /ml2/qwen-sliders/warm-light/adapter.safetensors \
  --prompt 'A ceramic cup on a wooden table.' --strength 0.75 \
  --seed 42 --height 512 --width 512 --steps 40 --out warm-light.png
```

Dimensions must be multiples of 32. Output is PNG to retain the pipeline's native
RGBA channels. Strength changes each nonlinear residual; it does not promise linear
interpolation in image space. The exporter is a routed particle artifact, not an
ordinary LoRA and cannot be loaded through `pipe.load_lora_weights()`.

The CI workflow installs pinned source dependencies on CPU, runs native/unit tests,
builds a wheel, then installs and exercises the wheel outside the source tree.
It never downloads base weights. Real-model acceptance checks remain in
[docs/real-model-validation.md](docs/real-model-validation.md).
