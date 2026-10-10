# Validation evidence and limits

Local validation on 2026-10-10 uses Python 3.12.15, CPU PyTorch 2.14.1, Transformers
5.17.0, Diffusers `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458`, core
`20fdf98d25455582e7e7191eef8ef51cf73173ae` and ParticleGAN
`82b72661c03377f348ccb76de37cd4ca02d45a8d`. The isolated task environment reuses
a read-only CPU PyTorch installation; exact source dependencies are installed from
verified local checkouts. CI independently installs dependencies from their public pins.

## Native CPU evidence

The suite has 18 tests and exercises actual tiny Qwen-Image 2.1 native code:

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

Resource policy and snapshot file validation tests use synthetic reports and a
**mocked miniature manifest**. They are unit checks, not a full snapshot hash/load test.
The tiny pipeline's processor is an initialization stub and text states are synthetic.
No actual tokenization, text encoder weights, pretrained transformer weights or VAE
weights/decoder are tested. No generated image-quality evidence is claimed.

Separate CLI smokes run four CPU training steps with batch one, and four with batch
two plus gradient checkpointing. Losses, noise and VIC remain finite. Inference writes
synthetic latent tensors. Ruff lint/format checks and wheel/sdist packaging are checked.
CI repeats native/unit tests and runs the installed wheel outside the source checkout.

## Material real-model blocker

The task sandbox exposes no NVIDIA device nodes. A read-only host query found two
working RTX A6000s (48 GiB each), both with existing active jobs. At the first check,
about 8.5 GiB and 17.8 GiB were in use; jobs were not stopped or changed. Host available
RAM was about 24 GiB, with later task checks around 27 GiB. `/ml2` had roughly 357 GiB
free storage. No matching official 2.1 snapshot was found in the specific standard
Hub and `/ml2/models/Qwen-Image-2.1` locations checked; this is not an exhaustive cache search.

The official pinned snapshot requires 30.84 GiB in weight files alone. This initial
one-device loader's conservative CUDA policy requires 40 GiB available host RAM and
38 GiB free on the selected GPU; full CPU FP32 loading requires 80 GiB host RAM.
These are unmeasured policies, not benchmarked minimums or fit guarantees.
The host RAM check fails, and the restricted execution view cannot use CUDA.
No model weights were downloaded, no GPU smoke was launched, and no long training
was started. Source/configuration metadata downloads and bounded CPU checks only.

Real snapshot load/encoding, full-model gradients, CUDA memory use/resume, native
RGBA decode, held-out edit direction and preservation, and training convergence
remain open. Follow [the bounded acceptance procedure](docs/real-model-validation.md)
after resources are available. CI success establishes source/package integration;
it does not close these full-model gaps.
