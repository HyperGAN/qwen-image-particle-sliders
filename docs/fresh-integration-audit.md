# Fresh Qwen integration source audit

This is an unpublished local candidate, inspected on 2026-10-21. It is not a
merge, a checkpoint migration, a new installed-wheel qualification, or an active
training pin update. Resource admission for tests and builds is pending.

## Source choice and preserved work

Read-only GitHub queries confirmed main at
`b418ff5b709dd687fdc7d9fbb5498c592ec21a58` and
[PR1](https://github.com/HyperGAN/qwen-image-particle-sliders/pull/1) open at
`31e797f290c58bda74bb2e7e69ad2aeef2b4244b`, targeting main. Main is inventory
evidence only. The isolated `audit/fresh-qwen-core070` branch starts from the
reference draft `a3222f8aa1f05df8063c92ae6ac256440b9b6c1e`, whose ancestry retains
PR1. No rebase or public write was performed.

The shared source was fetched read-only from the owner's local repository into
an isolated checkout at exactly
`fd89e1c780ab5210cd1be98ec3b70977e6349379` (core 0.7.0). The derivative retains
that proposed milestone in both lock copies and dependency metadata. Clean main
integration and compact inference loading are common-owner work; this candidate
does not retarget PR141/142 or substitute another common pin.

## Entrypoints and ownership

| Path | Integration and boundary |
| --- | --- |
| `particle_sliders.image_backends:qwen` | Complete owned `lifecycle.prepare`: staged/local runtime, paired native teachers, actual AdapterBank and particle parameter, response adapter, critic, shared game and cleanup. Callers provide config, not constituent objects. |
| `particle-slider` / `open_session` | Common train, split resume, EMA, export, response evaluation, artifact load and native-render dispatch. New image-session packet schemas and explicit cohort. |
| `qwen-slider-train` / `train.train` | Retained product envelope and CLI. Uses shared `ConditionalPG09`, `advance`, EMA and atomic save; writes `qwen-native-pg09-training-v2` and EMA safetensors. It is not silently redirected into the new session schema. |
| `qwen-slider-infer` / `infer.infer` | Retained strict product metadata/safetensors loader and native pipeline sampler; incompatible with common image-session packets. |
| `qwen-slider-preflight` | Retained resource report; no download or automatic device selection. |
| Historical main/gmix cohorts | Original pins/loaders and receipts required; no rebinding to this candidate. |

The generator is the **routed particle GAN** arm. There is one actual 128×4
particle parameter, owned by the shared network and passed into the native
attention-output hooks without registration aliases. The product's rank-8
down/routed/up maps are FP32. Base transformer, text encoder and VAE stay frozen.
Native Qwen packing/unpacking, unpadded per-prompt RoPE layouts, conditioning
masks, raw scheduler time and model time conversion stay in their existing
product/core native helpers. No GAN mathematics or optimizer law was copied.

The native callback continues to use QwenImage21Pipeline and the retained
FlowMatchEulerDiscreteScheduler configuration, true_cfg_scale=1, no KV cache,
and an explicitly seeded CPU generator. Rendering does not replace the native
sampler with the response-field evaluator. Reference conditioning, negative
strengths, Edit/Layered/Turbo and alternate devices are unsupported. Tiny mode
advertises latent output only; native PIL output preserves native decoding.

The fresh light edits make capabilities match actual FP32 CPU/BF16 CUDA loading,
verify staged runtime kind/device/precision, expose native checkpointing with
restoration, and preserve scheduler identity/state plus the four transient
fields assigned by the pinned native pipeline. Setup interruption, adapter
device-transfer failure and partially attached hooks now unwind through the
owned cleanup paths. Product metadata depends on core, which owns the exact
ParticleGAN 0.9.1 requirement; there are no product imports in the new core
lifecycle dependency closure.

## Evidence boundaries

- `docs/cuda-cap-resume.json` and `docs/real-model-smoke.json` are historical
  gmix/core `20fdf98d` evidence; they do not qualify PR1 or this candidate.
- `docs/pg09-audit.json` and its 64-test receipt apply to shared `4be0b5b2`.
- `docs/pg09-final-core.json` records PR1's shared `cdfb80a7`, 10 focused checks,
  and exact fresh-process resume under both presets. These receipts are retained
  unchanged and do not qualify core 0.7.0 or these fresh edits.
- The reference milestone receipt records core `fd89e1c` wheel SHA256
  `9637aede473939e4284cf05c4c8d950e5a84ab5b0fe3c482080eb10028abc4d9`
  and Qwen `a3222f8` wheel SHA256
  `2790010637c80744732c6c7948a3c73e0fb785f707eeea1b8ff5732b90c66b73`.
  Its Qwen run reports 9 passed / 1 skipped; this is prior reference evidence,
  not a run on the fresh candidate.
- This pass runs only AST syntax checks, whitespace checks and the common
  guard's pure AST reverse-import helper against the isolated shared source.
  No PyTorch import, tests, installation, wheel build, model load/download or
  accelerator work is performed. Machine-readable source/evidence hashes are
  in `fresh-integration-inspection.json`.

## Pending common interfaces and qualification

Compact inference is pending a common preparation-free factory/binding seam;
current artifact opening still builds teachers and the training game. This
candidate does not create an incompatible private loader.

HyperGAN owns run presentation via the common on_step, manifest and artifact
seams. Native inference rendering is the product callback above. No competing
run UI is introduced.

HANDL/HNDL migration must preserve matched weights, gradients and ownership for
the shared RoutedMLP and conditional readout critic and product RoutedProjection.
The existing optional `critic_kind='hndl'` is a distinct architecture and packet
identity, not matched-weight migration evidence. Native pretrained backbones
remain dependencies behind their existing adapter boundary.

When the parent schedules an isolated offline CPU slot, build/install exact
product and shared wheels outside source trees and run the lifecycle/ownership,
native checkpoint, optimizer-history, inference-transaction and dependency
guard checks. Recheck split resume including metrics/EMA/RNG/coverage, corrupt
restore before mutation, export/load, exact zero and fractional strengths,
sampler/RNG preservation and setup/update failures. No current pass claims
installed conformance, native pretrained quality, GPU parity or convergence.
