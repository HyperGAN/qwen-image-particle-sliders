# Qwen modern candidate: isolation and qualification

Stable remains `qwen-image-particle-sliders==0.2.0`, core 0.8.1 at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, official PG 0.9.1 and optional HNDL 0.8.2.
Its native runtime files, original lock copies, recipes and loaders are unchanged.
The packaging-only build hook excludes every `hypergan_hndl*` module and the modern
lock from the stable wheel, including stale copies in build/lib. A stable sdist
retains the development sources/builder, but rebuilding its normal wheel applies
the same exclusion. Editable checkouts may expose development files; modern
admission refuses them in a stable environment.

The modern artifact is a **different distribution and prerelease**:
`qwen-image-particle-sliders-modern-candidate==0.3.0a2`. It cannot satisfy the stable
`==0.2` requirement or replace that project through an ordinary upgrade. It owns
its native helper package plus the modern modules in a separate environment.
Never co-install both distributions: candidate package import and modern admission
refuse stable distribution metadata, and imported package/module origins are
checked against the candidate and dependency wheels before native use.
Editable modern installs and incomplete wheel manifests are not admitted.

The new product cohort is `qwen-image21-hypergan-hndl-v3`, backend `qwen-hndl`.
Earlier experimental artifacts require their original head and environment.
The v2 preview with product version 0.3.0a1 and core commit 4abc0eb remains a separate
cohort; its [archived lock](cohorts/qwen-hndl-v2-core4abc.lock.json) is preserved
verbatim. Its packets refuse in this v3 candidate. There is no automatic migration,
including from the earlier 0.2.0+local prototype.
No historical checkpoint or receipt is relabeled.

## Exact shared sources

| Component | Candidate |
|---|---|
| Core | `a794e3b4a8813b012f2c4faa3da82b78d4a90214` / 0.12.0 |
| Optional reviewed local core archive | SHA256 `113496f2b1a7e1c40eeec1df84678bb718234edf73023a4e46f515b209d95b6a` |
| ParticleGAN | official 0.9.4 |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7`, clean build provenance |
| HNDL | 0.10.0, exact source closure |
| Diffusers | `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458` |

The active core dependency is the exact reviewed commit `a794e3b4a8813b012f2c4faa3da82b78d4a90214`.
The recursive lock covers all 51 tracked runtime files under both
`particle_sliders` and `concept_slider_core`, including data and LICENSE files.
The runtime verifies the complete installed inventory, bytes, and imported origins.
The public builder verifies the immutable Git source-tree inventory and the same
bytes. Product identity also binds the core commit and recursive closure digest.
The original stable locks and prior preview lock are not rewritten or relabeled.

The shared owner cleared this core for CPU and bounded synthetic CUDA software
fixtures. That does not qualify Qwen's pretrained GPU path, native widths, model
quality, or audio behavior. This product migration runs CPU checks only.

## Build and install in a separate environment

```bash
python scripts/build_hypergan_hndl_candidate.py --public-core --outdir dist-modern
```

The builder verifies the exact public core pyproject and all 51 recursive source/data hashes
before staging. `--core-wheel /path/to/reviewed.whl` additionally verifies the
canonical local archive, but still requires public source verification. **Both
modes emit the same portable immutable public VCS requirement** on
`particle-sliders-core[hypergan-hndl]` at a794e3b4; neither embeds a file:// dependency.
Unavailable/changed public sources refuse with no local fallback.

The builder changes metadata and the candidate initializer inside a temporary
copy. The stable source initializer/version/pyproject are not rewritten by it.
It registers only `qwen-hndl`, omits old product train/infer CLIs, and uses the
shared `particle-slider` CLI. Its receipt lists only the fresh isolated build's
wheel/sdist and hashes; unrelated files already in outdir are left alone and
never counted as deliverables.

Install the explicitly named prerelease wheel with normal dependency resolution
in an empty venv. The concrete clean qualification uses Python 3.12/Linux x86_64,
official CPU Torch 2.6.0 and matching CPU torchvision 0.21.0. It uses no --no-deps,
inherited site-packages or shared runtime views. The public VCS receipts bind
core a794e3b4, HyperGAN 5f368a and Diffusers 1d5d056. This is a concrete CPU qualification,
not a universal platform or pretrained/GPU compatibility promise.

## Shared ownership and lifecycle

Shared HNDLParticleProjection supplies each routed projection graph; shared
HNDLFieldCritic supplies the spatial critic. Qwen attaches/restores native
forwards and removes only its singleton frame axis inside the critic. The game
and public penalty receive original [1,C,1,H,W] fields. ConditionalPG09 owns
losses, recipes, updates, optimizer state, EMA and checkpoint/export validation.
No GAN mathematics or replacement loss is copied. `critic_kind` overrides refuse
because this profile has one fixed shared HNDL field critic.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
from particle_sliders.inference import open_inference

config = SessionConfig(
    'qwen-hndl', 'new-qwen-hndl-v3-cpu', dummy=True, steps=4, ema_decay=.5,
    height=32, width=64, trajectory_steps=3, records_per_pair=2,
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

The native v2 descriptor records training **height, width, trajectory_steps and
total training_steps** in its configuration and model identity; completed GAN
steps remain in the shared origin receipt. Inference defaults to that exact
geometry and trajectory budget. Explicit different height/width/render steps
refuse during preflight before native allocation/encoding. A different sampling
context requires a separately qualified cohort rather than silently reusing
these results. This conservative restriction applies to raw and EMA alike.

Native-only inference constructs a verified frozen runtime and selected HNDL
adapter/particles, never teachers, a game, critic or optimizer. Exports retain
the context receipt above, but omit teacher rows, paired prompts, critic/optimizer
state and resumable game state. Both selections reproduce their
training-side latent exactly at strengths 0, 0.5, and 1 with native checkpointing on/off.
Opening either a session or inference preserves caller RNG; tests assert this
immediately after open, before render or close. New checkpoints restore exact
raw/EMA continuation in their own cohort. S081 packets retain their original
loaders and are refused by the modern game without mutation.

## Failure and qualification limits

Shared packet/tensor validation runs before copying exported adapter values.
For the exercised validation refusals, no adapter values are partially copied;
managed projection forwards, captured mode/gradient/checkpoint settings and RoPE
references are restored. Partial attachment setup attempts every owned restore. Failed forward restores
remain tracked for close retry, and native cleanup is attempted even if a forward
restore fails. An explicit operation error remains primary, with cleanup errors
attached as notes. After a failed close, retry close before any further use.
This is **not a general atomic transaction for arbitrary third-party side effects**:
a failing native loader/builder may allocate resources or mutate state outside
those captured fields. No universal CUDA allocation cleanup or third-party buffer
rollback is claimed. Low-level build_game/Bank callers own their surrounding RNG
and lifetime; use the shared lifecycle for its managed open/observation contexts.

Normal GitHub CI is **S081-only** on Python 3.11/3.12. The modern modules skip unless
QWEN_TEST_CORE012=1 and the distinct candidate environment is installed. Passing
GitHub checks do not establish modern qualification. No workflow edit or new
OAuth scope is requested. Run the modern checks explicitly:

```bash
QWEN_TEST_CORE012=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -I -m pytest -q -o pythonpath= \
  /path/to/checkout/tests/test_hypergan_hndl.py \
  /path/to/checkout/tests/test_hypergan_hndl_lifecycle.py \
  /path/to/checkout/tests/test_hypergan_hndl_isolation.py \
  /path/to/checkout/tests/test_hypergan_hndl_pin.py \
  /path/to/checkout/tests/test_candidate_packaging.py \
  /path/to/checkout/tests/test_stable_packaging.py
```

Pretrained/GPU, memory, pretrained sampler/decoding and convergence remain unrun.
No model weights are accessed or downloaded by these fixtures, and no GPU job,
package/model release or deployment is included. Root coordinates core/product
review and any default promotion; source drift requires a reviewed cohort.

The lock was regenerated once from the exact Git tree with
`scripts/regenerate_hypergan_hndl_lock.py`. The recorded local core wheel is a
qualification input, not a release artifact. Building it twice with setuptools
84.2.1 and `SOURCE_DATE_EPOCH=1793476986` produced identical archive bytes. Normal
installation uses the portable exact Git dependency, not that local wheel path.
The old preview packets were captured before this migration and retained unchanged
for refusal tests; set `QWEN_OLD_PREVIEW_DIR` to their receipt directory when running
that explicit qualification case. No pretrained model weights are included in those fixtures.
