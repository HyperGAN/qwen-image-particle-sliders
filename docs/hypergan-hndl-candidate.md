# Qwen modern candidate: isolation and qualification

Stable remains `qwen-image-particle-sliders==0.2.0`, core0.8.1 at
`32f6830f6402e85aa77b6631db64fcbbb2131f6b`, official PG0.9.1 and optional HNDL0.8.2.
Its native runtime files, original lock copies, recipes and loaders are unchanged.
The packaging-only build hook excludes every `hypergan_hndl*` module and the modern
lock from the stable wheel, including stale copies in build/lib. A stable sdist
retains the development sources/builder, but rebuilding its normal wheel applies
the same exclusion. Editable checkouts may expose development files; modern
admission refuses them in a stable environment.

The modern artifact is a **different distribution and prerelease**:
`qwen-image-particle-sliders-modern-candidate==0.3.0a1`. It cannot satisfy the stable
`==0.2` requirement or replace that project through an ordinary upgrade. It owns
its native helper package plus the modern modules in a separate environment.
Never co-install both distributions: candidate package import and modern admission
refuse stable distribution metadata, and imported package/module origins are
checked against the candidate and dependency wheels before native use.
Editable modern installs and incomplete wheel manifests are not admitted.

The new product cohort is `qwen-image21-hypergan-hndl-v2`, backend `qwen-hndl`.
Earlier experimental v1 artifacts require their original head/environment; there
is no automatic migration, including from the earlier 0.2.0+local prototype.
No historical checkpoint or receipt is relabeled.

## Exact shared sources

| Component | Candidate |
|---|---|
| Core | `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231` / 0.12.0 |
| Optional reviewed local core archive | SHA256 `9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad` |
| ParticleGAN | official 0.9.4 |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7`, clean build provenance |
| HNDL | 0.10.0, exact source closure |
| Diffusers | `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458` |

Core commit/API, PR148 head and git remote ref were independently verified as
4abc0eb; all49 locked source/graph/manifest hashes and its pyproject hash matched.
This is an immutable source qualification, not a claim that the upstream draft
was merged or released. The product lock is separate from the original S081 locks.
Admission checks actual imported origins, including already loaded core/HNDL/
HyperGAN/Diffusers submodules, as well as installed source hashes. Product Python
source hashes, including the admission helper and staged package initializer,
are included in the modern model/checkpoint identity.

## Build and install in a separate environment

```bash
python scripts/build_hypergan_hndl_candidate.py --public-core --outdir dist-modern
```

The builder verifies the exact public core pyproject and all49 source hashes
before staging. `--core-wheel /path/to/reviewed.whl` additionally verifies the
canonical local archive, but still requires public source verification. **Both
modes emit the same portable immutable public VCS requirement** on
`particle-sliders-core[hypergan-hndl]` at4abc0eb; neither embeds a file:// dependency.
Unavailable/changed public sources refuse with no local fallback.

The builder changes metadata and the candidate initializer inside a temporary
copy. The stable source initializer/version/pyproject are not rewritten by it.
It registers only `qwen-hndl`, omits old product train/infer CLIs, and uses the
shared `particle-slider` CLI. Its receipt lists only the fresh isolated build's
wheel/sdist and hashes; unrelated files already in outdir are left alone and
never counted as deliverables.

Install the explicitly named prerelease wheel with normal dependency resolution
in an empty venv. The concrete clean qualification uses Python3.12/Linux x86_64,
official CPU Torch2.6.0 and matching CPU torchvision0.21.0. It uses no --no-deps,
inherited site-packages or shared runtime views. The public VCS receipts bind
core4abc0eb, HyperGAN5f368a and Diffusers1d5d056. This is a concrete CPU qualification,
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
    'qwen-hndl', 'new-qwen-hndl-v2-cpu', dummy=True, steps=4, ema_decay=.5,
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
training-side latent exactly at strengths0/.5/1 with native checkpointing on/off.
Opening either a session or inference preserves caller RNG; tests assert this
immediately after open, before render or close. New checkpoints restore exact
raw/EMA continuation in their own cohort. S081 packets retain their original
loaders and are refused by the modern game without mutation.

## Failure and qualification limits

Shared packet/tensor validation runs before copying exported adapter values.
For the exercised validation refusals, no adapter values are partially copied;
managed projection forwards, captured mode/gradient/checkpoint settings and RoPE
references are restored. Partial attachment setup unwinds already-owned forwards.
This is **not a general atomic transaction for arbitrary third-party side effects**:
a failing native loader/builder may allocate resources or mutate state outside
those captured fields. No universal CUDA allocation cleanup or third-party buffer
rollback is claimed. Low-level build_game/Bank callers own their surrounding RNG
and lifetime; use the shared lifecycle for its managed open/observation contexts.

Normal GitHub CI is **S081-only** on Python3.11/3.12. The modern modules skip unless
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
  /path/to/checkout/tests/test_candidate_packaging.py \
  /path/to/checkout/tests/test_stable_packaging.py
```

Pretrained/GPU, memory, pretrained sampler/decoding and convergence remain unrun.
No model weights are accessed or downloaded by these fixtures, and no GPU job,
package/model release or deployment is included. Root coordinates core/product
review and any default promotion; source drift requires a reviewed cohort.
