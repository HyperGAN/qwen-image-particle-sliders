> Current adoption: [published PG0.9.5 / core0.13.0](docs/pg095-adoption.md). Versioned earlier receipts below retain their original source scope; none establishes a PG0.9.5 native result.

# Qwen Image 2.1 common lifecycle

Current and new runs disable EMA explicitly with `ema_decay=None` and use RAW
for training evaluation, export and inference. Do not create EMA comparisons.
Historical EMA artifacts and their recorded tests keep their original loaders.

Canonical shared pin: `ddb41423bcefc1fe76cd94d0f9b367353b3ea137` / core0.12.0 / ParticleGAN0.9.4.
See [the current CPU adoption audit](docs/pg094-adoption.md). This is a fresh source
cohort. The historical 26-check receipt and core0.8.1 adoption notes remain historical
and do not qualify this pin. Old runs retain their original loaders and source locks.

This derivative registers backend `qwen` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.
The dependency direction is product → particle-sliders-core → ParticleGAN;
the core owns the exact released GAN dependency.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('qwen', 'new-qwen-pg094-cpu-v1', dummy=True, steps=4,
    height=32, width=32, trajectory_steps=4, records_per_pair=2, ema_decay=None,
    rows=[dict(neutral='a red cup', positive='a shiny red cup'),
          dict(neutral='a blue bowl', positive='a shiny blue bowl')])
with open_session(config) as session:
    session.train(stop_after=2)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()
    session.export('new-run/adapter.pt', selection='raw')
    session.export_inference('new-run/native.pt', selection='raw')
```

Strengths remain [0,1]; reference-image conditioning is unsupported.
Qwen uses its native 2.1 packing/scheduler. Local native options are `model_dir`,
`cache_dir`, `device`; downloads remain disabled. The tiny fixture can render
latents using `session.render(prompt=..., height=32, width=32, steps=4,
output_type="latent")`. Native pixel decoding is a separate qualification.
Tiny render preflight now defaults to its supported latent output. Use
`native_request={...}` at open time to validate request geometry/options before
runtime allocation. Render callbacks also validate before prompt encoding.

Base precision follows the native loader: CPU uses `float32`, CUDA uses
`bfloat16`; `base_precision` may assert that value but does not cast the runtime.
Trainable adapters and the particle bank remain FP32. Dummy mode accepts no
model/cache paths and uses CPU FP32. `gradient_checkpointing=True` enables the
native transformer setting for this session; close and setup failures restore
the caller's settings. Teacher capture and native rendering use private
scheduler copies and restore the pinned pipeline's transient sampling fields.
An injected staged runtime must match the declared dummy/native kind, device
and precision. Reference images, negative strengths and nonnative devices reject.
Native observation hooks also restore registered backbone/encoder/VAE buffers,
module modes and Qwen's unregistered RoPE frequency list. An optional staged
`admission` callback receives the shared before-update context; returning False
refuses the update without closing the session. It is a decision callback, not a
place to mutate the borrowed native runtime.

Native-only v2 export opens without teacher rows, a critic, optimizer or game:

```python
from particle_sliders.inference import open_inference

# This fixture needs no local options; native weights require model_dir/device.
with open_inference('new-run/native.pt') as inference:
    latent = inference.render(prompt='a new green vase', strength=.5, seed=11)
```

The v2 artifact contains adapter values plus verified native model/source/profile
identity and an origin receipt, without training data/config/game state.
Native-only response evaluation is disabled: training-row velocity evaluation
remains on the training session. The common CLI `infer` consumes v2 artifacts;
the existing CLI `train` still writes v1 exports, so use `export_inference` for v2.
V1 artifacts and product safetensors keep their original loaders and schemas.

Write `config.as_dict()` as JSON to use `particle-slider capabilities`,
`particle-slider train`, `particle-slider evaluate` and `particle-slider render`.
Run `particle-slider --help` for the common arguments. Use a fresh output
directory for each CLI continuation to preserve old metrics and receipts.
Training packets and inference artifacts have different versioned schemas;
exports cannot resume training. New source pins require a new cohort or a
reviewed explicit migration.

The core sdist includes IMAGE-LIFECYCLE.md with the public factory, capability,
packet and error contracts. Backend dependencies stay isolated: Qwen and Fibo
use different Diffusers Git revisions and are not a joint environment promise.
Use CPU thread limits and offline mode for these fixtures. They do not qualify
pretrained behavior, GPU memory, convergence or image quality.

The modern candidate is documented in [hypergan-hndl-candidate.md](docs/hypergan-hndl-candidate.md).
It is a distinct prerelease distribution, `qwen-image-particle-sliders-modern-candidate==0.3.0a2`,
for a separate environment. The stable wheel excludes its modules/lock. The candidate
registers `qwen-hndl` for shared sessions and native-only raw/EMA v2 export/inference.
V2 records training geometry and both trajectory/training step budgets; incompatible
render contexts refuse. Normal CI qualifies canonical PG094 only; modern CPU checks are an
explicit manual lane. The active modern core pin is `a794e3b4a8813b012f2c4faa3da82b78d4a90214`; default promotion and
pretrained/GPU qualification remain separate review decisions.
