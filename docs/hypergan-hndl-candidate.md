# Qwen compatibility and HyperGAN/HNDL candidate

The supported default remains S081: core **0.8.1** at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, released PG **0.9.1**, the exact
Diffusers revision in `backend.lock.json`, and optional HNDL **0.8.2** readout.
Existing CLI, lifecycle registrations, checkpoint schemas, recipes and both
copies of the original lock are preserved. Historical receipts retain their dates.

The opt-in Python API `qwen_image_sliders.hypergan_hndl.build_game` executes real
shared **core 0.12 / PG 0.9.4 / HyperGAN develop / HNDL 0.10** graphs. This is a
new architecture, `qwen-image21-hypergan-hndl-v1`, with a separately pinned source
manifest, `qwen_image_sliders/hypergan-hndl.lock.json`:

| Component | Exact candidate |
|---|---|
| Shared core | `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` / 0.12.0 |
| ParticleGAN | official 0.9.4 |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7`, clean build provenance |
| HNDL | 0.10.0, source closure checked by shared builder |
| Native Diffusers | `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458` |

The shared candidate was local and unpublished when qualified. Default package
metadata deliberately still selects S081. The candidate is developer-only: use a
separate environment provisioned with the reviewed core wheel and exact native
dependencies, then install this product wheel with `--no-deps`. This manual lane
does not satisfy the product's default S081 dependency declaration and is not a
resolver-qualified public upgrade. Do not upgrade an existing training environment.

## Ownership and API

The shared `HNDLParticleProjection` supplies the routed graph on each native
`attn.to_out.0` projection. The product attaches and restores native forwards;
the shared `QwenImage21ResponseAdapter` binds the actual trainable leaves and one
128x4 particle parameter. The shared `HNDLFieldCritic` consumes spatial responses;
only its critic wrapper removes Qwen's singleton frame axis. The game and public
penalty always receive the original `[1,C,1,H,W]` field. Shared source admission,
domain guards, losses, recipes, optimizers and checkpoint validation stay in core.
The candidate changes the critic architecture; it does not claim equivalence to
the old readout critic or substitute a new loss.

```python
from qwen_image_sliders.hypergan_hndl import build_game
from qwen_image_sliders.teachers import load_prompts, paired_records
from qwen_image_sliders.tiny import TinyRuntime

runtime = TinyRuntime()  # random weights, synthetic text, no downloads
records = paired_records(runtime, load_prompts(), seed=7, height=32, width=32,
                         trajectory_steps=4, records_per_pair=2)
game = build_game(runtime, records, steps=4, batch=2)
try:
    game.step()
    checkpoint = game.state_dict()  # shared strict schema; not a product export
    # Use the shared context so native checkpoint recomputation captures strength.
    with game.network.scaled(0):
        latent = runtime.sample(runtime.encode('a cup'), seed=11,
                                height=32, width=32, steps=4, output_type='latent')
finally:
    game.backend.adapter.close()  # restores borrowed forwards/checkpoint settings
```

Use `game.load_state_dict(checkpoint)` only in the exact same source/runtime
cohort. PG0.9.1/S081 packets reject before state mutation; resume them with their
original loader and environment. There is no checkpoint conversion or relabeling.
The modern API has no product export/native inference lifecycle yet; the old
`train`, `infer` and common backend entrypoints continue to use S081.

## Qualification and promotion gates

Installed S081 CPU suite: **86 passed**, one separate modern-lane module skipped.
Installed candidate CPU suite: **6 passed**. An independently created actual
S081 game packet was refused by the modern loader with all live state and RNG
unchanged. The candidate CPU suite runs separately with:

```bash
QWEN_TEST_CORE012=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -I -m pytest -q -o pythonpath= /path/to/checkout/tests/test_hypergan_hndl.py
```

The modern cases use actual tiny Qwen native forwards, the verified HyperGAN
builder and HNDL graphs, both public PG presets, gradient checkpointing on/off,
unique optimizer ownership, frozen base weights, exact zero strength, exact
checkpoint continuation, profile refusal without mutation and setup cleanup.
They do not mock the builder, GAN or native model. The corruption/setup-refusal
tests inject only the failure they exercise.

Tests use task-local virtual environments and reuse read-only CPU Torch/runtime
libraries; this is not a pristine full native dependency installation. The normal
CI continues to run S081 on Python3.11/3.12. To repeat the entire suite against
a non-editable installed wheel with its source path disabled, run
`python scripts/check_installed_contracts.py`. The workflow itself is unchanged:
the available publication credential cannot update workflow files. CI cannot fetch the unpublished
modern core commit, so the modern suite remains an explicit separate lane.

Promotion requires central publication/review of the exact core cohort, a
resolver-qualified modern product profile, modern export/inference lifecycle
qualification, and separately authorized pretrained/GPU, memory, sampler and
convergence checks. No model weights were accessed or downloaded; no GPU test,
training job, image-quality claim or release is included here. Source/hash drift
requires a new reviewed profile, not an expanded version range or relaxed gate.
