# Qwen Image 2.1 common lifecycle

Proposed local shared pin: `32f6830f6402e85aa77b6631db64fcbbb2131f6b` / core 0.8.1.
Unpublished; exact-head review is required before draft publication.
The source-only adoption and outstanding gates are recorded in
[docs/core081-adoption.md](docs/core081-adoption.md). The prior core0.7.0 source
pass is preserved at `bf8a6c160153c80f7c4155f96ed3f943114615d7`.

This derivative registers backend `qwen` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, EMA, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.
The dependency direction is product → particle-sliders-core → ParticleGAN;
the core owns the exact released GAN dependency.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('qwen', 'new-qwen-core081-cpu-v1', dummy=True, steps=4,
    height=32, width=32, trajectory_steps=4, records_per_pair=2, ema_decay=.9,
    rows=[dict(neutral='a red cup', positive='a shiny red cup'),
          dict(neutral='a blue bowl', positive='a shiny blue bowl')])
with open_session(config) as session:
    session.train(stop_after=2)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()
    session.export('new-run/adapter.pt', selection='ema')
    session.export_inference('new-run/native.pt', selection='ema')
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
