# Qwen Image HNDL candidate

The separate `qwen-image-particle-sliders-modern-candidate==0.3.0a3` wheel
registers `qwen-hndl` for the shared lifecycle. Use a separate environment
from the canonical package.

Build the candidate against the verified public core source.

```bash
python scripts/build_hypergan_hndl_candidate.py --public-core --outdir dist-modern
python -m pip install dist-modern/qwen_image_particle_sliders_modern_candidate-0.3.0a3-py3-none-any.whl
```

The builder verifies the immutable core revision and every installed runtime
file against the package lock. The candidate uses core 0.15.2, ParticleGAN
0.11.1, HNDL 0.10.0 and HyperGAN's pinned graph builder. The installed
candidate also verifies its own source and its dependency imports.

Open new sessions with `ema_decay=None`. Use `selection='raw'` for adapter
exports and inference. The product owns native Qwen hooks and spatial
geometry. The shared engine owns the adversarial game and checkpoint state.

Use the [captured candidate guide and source locks](https://github.com/HyperGAN/qwen-image-particle-sliders/blob/7ea24d2072324bb707e0f7ceb6e5f30d048dff19/docs/hypergan-hndl-candidate.md)
to reproduce earlier artifacts. Pretrained GPU execution needs its own
qualification for the current source.

The [closed HNDL cohort source and restore utility](https://github.com/HyperGAN/qwen-image-particle-sliders/tree/7ea24d2072324bb707e0f7ceb6e5f30d048dff19/cohorts/pre-lr-hndl)
remain available from that checkout.
