# Qwen Image 2.1 common lifecycle

Proposed local shared pin: `fd89e1c780ab5210cd1be98ec3b70977e6349379` / core 0.7.0.
Unpublished; exact-head review is required before draft publication.

This derivative registers backend `qwen` for `particle_sliders.lifecycle`.
The core owns preparation dispatch, updates, EMA, checkpoints, export,
evaluation and cleanup. The product factory owns native loading/conditioning,
geometry and sampler differences. This is a new source cohort; keep original
pinned runs and receipts with their original loaders.

```python
from particle_sliders.lifecycle import SessionConfig, open_session
config = SessionConfig('qwen', 'new-qwen-cpu-v1', dummy=True, steps=4,
    height=32, width=32, trajectory_steps=4, records_per_pair=2, ema_decay=.9,
    rows=[dict(neutral='a red cup', positive='a shiny red cup'),
          dict(neutral='a blue bowl', positive='a shiny blue bowl')])
with open_session(config) as session:
    session.train(stop_after=2)
    session.checkpoint('new-run/training.pt')
with open_session(config, resume='new-run/training.pt') as session:
    session.train()
    session.export('new-run/adapter.pt', selection='ema')
```

Strengths remain [0,1]; reference-image conditioning is unsupported.
Qwen uses its native 2.1 packing/scheduler. Local native options are `model_dir`,
`cache_dir`, `device`; downloads remain disabled. The tiny fixture can render
latents using `session.render(prompt=..., height=32, width=32, steps=4,
output_type="latent")`. Native pixel decoding is a separate qualification.

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
