# Qwen compatibility and HyperGAN/HNDL candidate

The supported default remains S081: core **0.8.1** at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, released PG **0.9.1**, the exact
Diffusers revision in `backend.lock.json`, and optional HNDL **0.8.2** readout.
Existing CLI, lifecycle registrations, checkpoint schemas, recipes and both
copies of the original lock are preserved. Historical receipts retain their dates.

The separately built modern profile executes real shared **core 0.12 / PG 0.9.4 /
HyperGAN develop / HNDL 0.10** graphs. Its architecture is
`qwen-image21-hypergan-hndl-v1`; its common backend name is `qwen-hndl`.
`qwen_image_sliders/hypergan-hndl.lock.json` binds the exact candidate:

| Component | Exact candidate |
|---|---|
| Shared core | `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` / 0.12.0 |
| Reviewed local core wheel SHA256 | `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad` |
| ParticleGAN | official 0.9.4 |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7`, clean build provenance |
| HNDL | 0.10.0, exact source closure checked by the shared builder |
| Native Diffusers | `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458` |

## Separate packaging and the public-install gate

The exact core candidate is now public in [draft PR148](https://github.com/HyperGAN/particle-sliders/pull/148).
On 2026-10-31 its remote PR head and commit API both returned `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`;
all 49 source/graph/manifest hashes and the package pyproject hash matched the
reviewed local candidate. The PR remains open/draft, not merged or released.
Default product metadata still selects S081. Build a separate local-version wheel
from this checkout using the exact verified local core artifact:

```bash
python scripts/build_hypergan_hndl_candidate.py \
  --core-wheel /path/to/reviewed/particle_sliders_core-0.12.0-py3-none-any.whl \
  --outdir dist-modern
```

The builder verifies the core archive hash, package version and all 49 source/
graph/manifest hashes before staging any product files. It transforms metadata
inside a temporary copy, never rewriting the original source or pyproject.
The result is `qwen-image-particle-sliders==0.2.0+hypergan.hndl.local`, with an
exact local wheel/hash requirement on `particle-sliders-core[hypergan-hndl]`.
That extra supplies the actual pinned HyperGAN develop and HNDL0.10 dependencies.
The profile registers only `qwen-hndl` and omits the old product train/infer CLIs;
use the shared `particle-slider` CLI or common Python API. Native preflight remains
available. Keep one product profile per environment.

The local-artifact profile resolved and installed in a new empty Python3.12
environment with normal dependency resolution, official CPU Torch2.6.0 and matching
CPU torchvision0.21.0. No `--no-deps`, inherited site-packages or shared runtime
views were used. `pip check` passes.

To qualify from the public source instead of the local wheel, use:

```bash
python scripts/build_hypergan_hndl_candidate.py --public-core --outdir dist-modern-public
```

This mode fetches and verifies the locked core pyproject plus all 49 source hashes
before staging. It writes an exact immutable VCS requirement on
`particle-sliders-core[hypergan-hndl]` at commit `4abc0eb`, never a branch head or
version range. Changed/missing public sources refuse; there is no local fallback.
A second empty Python3.12 venv successfully installed this public-source profile
with normal dependency resolution. Installed VCS provenance confirms exact core
`4abc0eb`, HyperGAN `5f368a` and Diffusers `1d5d056`; all runtime packages reside
inside that venv and `pip check` passes. The installed public-source suite passed
**19 CPU cases**. The separate generated product wheel remains a candidate
artifact, not a package release or a change to the S081 default. Core source publication has cleared the
previous availability gate for this exact commit; upstream review and default
profile promotion remain separate decisions.

For local reproduction, install the generated wheel with its dependencies in a
new venv, supplying a matched CPU torch/torchvision pair as additional requirements.
The qualified pair is the official `torch-2.6.0+cpu-cp312-cp312-linux_x86_64.whl`
and `torchvision-0.21.0+cpu-cp312-cp312-linux_x86_64.whl` for Linux x86_64/Python3.12.
This is one concrete CPU environment, not a universal platform compatibility claim.

## Ownership and native geometry

The shared `HNDLParticleProjection` supplies the routed graph on each native
`attn.to_out.0` projection. The product attaches and restores native forwards;
the shared `QwenImage21ResponseAdapter` binds actual trainable leaves and one
128x4 particle parameter. The shared `HNDLFieldCritic` consumes spatial responses;
only its critic wrapper removes Qwen's singleton frame axis. The game and public
penalty receive the original `[1,C,1,H,W]` field. Shared source admission, domain
guards, losses, recipes, optimizers, EMA and checkpoint/export validation remain
in core. This is a distinct critic architecture; no equivalence to the old
readout critic or replacement loss is claimed.

The low-level `qwen_image_sliders.hypergan_hndl.build_game` remains available;
callers must close `game.backend.adapter`. The common lifecycle handles cleanup,
observation transactions, native checkpoint settings and shared EMA automatically.
An explicit `critic_kind` selector is refused because this profile has a fixed
shared HNDL field critic.

## Training, raw/EMA export and native-only inference

In the separately installed modern profile:

```python
from particle_sliders.lifecycle import SessionConfig, open_session
from particle_sliders.inference import open_inference

config = SessionConfig(
    'qwen-hndl', 'new-qwen-hndl-cpu-v1', dummy=True, steps=4, ema_decay=.5,
    height=32, width=32, trajectory_steps=4, records_per_pair=2,
    rows=[dict(neutral='a red cup', positive='a shiny red cup'),
          dict(neutral='a blue bowl', positive='a shiny blue bowl')])
with open_session(config) as training:
    training.train()
    training.checkpoint('new-run/training.pt')
    training.export_inference('new-run/raw.pt', selection='raw')
    training.export_inference('new-run/ema.pt', selection='ema')
with open_inference('new-run/ema.pt') as inference:
    latent = inference.render(prompt='a green vase', seed=11, strength=.5)
```

The dummy path uses tiny random native weights and synthetic text; it downloads
nothing. Native weights require a verified local model directory and compatible
precision/device options. They were not loaded during this qualification.

Native v2 exports contain only the selected adapter/particle values, native
architecture/source/profile identity and an origin receipt. They contain no
teacher rows, training configuration, critic or optimizer. The inference factory
constructs a frozen native runtime and HNDL adapter only; tests forbid construction
of the training game, critic, optimizer and teacher capture. Raw and EMA values
are each restored exactly and reproduce their corresponding training-side native
render at strengths0/.5/1. Export/render preserve the training state and RNG.
Checksum, profile, model identity and late invalid tensor refusals are atomic;
borrowed forwards/buffers/RoPE/checkpoint settings are cleaned up on failure.

Training checkpoints restore the exact raw and EMA continuation in the same
modern source/runtime cohort. Native inference artifacts cannot resume training.
Actual PG0.9.1/S081 packets reject before game mutation; use their original loader
and environment. No checkpoint conversion or historical relabeling is offered.

## Qualification and remaining gates

The installed modern suite covers both public presets, native checkpointing
on/off, response-only gradients into the actual routed particles independent of
the particle prior, unique optimizer ownership, frozen base parameters, exact
zero strength, exact resume, selected raw/EMA native-only parity and atomic refusal.
It executes the actual native model, HyperGAN builder and HNDL graphs.

```bash
QWEN_TEST_CORE012=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -I -m pytest -q -o pythonpath= \
  /path/to/checkout/tests/test_hypergan_hndl.py \
  /path/to/checkout/tests/test_hypergan_hndl_lifecycle.py \
  /path/to/checkout/tests/test_candidate_packaging.py
```

The separate S081 suite passed **88 installed CPU cases**, with the two modern
modules explicitly skipped because they require their separate environment. Run
`python scripts/check_installed_contracts.py` to avoid source-path shadowing.
Normal CI runs S081 on Python3.11/3.12. The workflow is unchanged and does not
automatically promote or exercise the separate modern profile. Initial CPU2.14 fixtures reused read-only runtime
libraries; the later clean CPU2.6 install/test lane removes that limitation for
this concrete modern environment.

Default promotion still requires coordinated review of the exact public core
and modern product profile. Pretrained/GPU, memory, full native sampler and convergence checks
remain unrun. No model weights were accessed or downloaded, no GPU job was started,
and no package/model release or deployment is included. Source/hash drift requires
a new reviewed profile rather than relaxed admission.
