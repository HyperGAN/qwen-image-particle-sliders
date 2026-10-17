# Source, model and license provenance

New native PG0.9.1 cohort audited on 2026-10-17. Original model/runtime pins are retained. `backend.lock.json` and its packaged
copy are identical; full commits and all official snapshot hashes are authoritative.

| Source | Pin | Role / license |
|---|---|---|
| HyperGAN/particle-sliders | `cdfb80a76b882b4b73861b12f555263bd64ffeb7` | Installable shared core 0.6.0 candidate; MIT |
| 255BITS/ParticleGAN | `9fc8fb51cba27c26d61f2b6a7d139f55cc726c35` | Released ParticleGAN 0.9.1; MIT |
| huggingface/diffusers | `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458` | Native pipeline/transformer/scheduler; Apache-2.0 |
| QwenLM/Qwen-Image-2.1 | `1993c3a4821f977df0b6dd2661ff5b40239aa55a` | Official requirements and license reference |
| Qwen/Qwen-Image-2.1 | `d26bb61231c349cf6b7896fa83353113880e1ba3` | Official model snapshot; Qwen research license |
| Transformers | `5.17.0` | Native Qwen3-VL encoder/processor; Apache-2.0 |

Inspected public Iris `6a226126…`, Anima `778ef38b…`, Krea2 `0b489f63…` and current
core public API documentation. The old gmix source pins are preserved under
`legacy_training` in the lock and remain historical; they are not resumed by v2. Iris product hooks informed this adaptation; their Apache license and
attribution are retained. Older local Anima/Krea pins were not used as the engine.

The official 2.1 transformer implements single-stream block-causal attention,
64-channel unpatched latents, `causal_condition`, differentiable native forward,
`_supports_gradient_checkpointing = True`, and `PeftAdapterMixin`. The native
pipeline supports LoRA loading and prefix caching, but this initial routed adapter
uses projection hooks and uncached forwards, not a guessed older-Qwen LoRA target
list. Its pipeline call is no-grad; training calls the transformer directly within
the shared game. Tiny tests verify actual backward propagation through those hooks.

The model manifest lists sizes and published LFS SHA256s for all seven weight shards.
Small load-relevant files were retrieved at the exact snapshot and hashed, including
processor, model index, scheduler and all component configs. The Qwen research
license is copied unmodified from that snapshot and included in wheel/sdist licenses.
No base weights, dependency engine files, credentials or private sources are shipped.

Official references:
- [Qwen source and documented requirements](https://github.com/QwenLM/Qwen-Image-2.1/tree/1993c3a4821f977df0b6dd2661ff5b40239aa55a)
- [Official model snapshot](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d26bb61231c349cf6b7896fa83353113880e1ba3)
- [Qwen license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE)
- [Native Diffusers transformer](https://github.com/huggingface/diffusers/blob/1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [Shared product contract](https://github.com/HyperGAN/particle-sliders/blob/cdfb80a76b882b4b73861b12f555263bd64ffeb7/docs/shared-core.md)

The optional `--critic hndl` route executes the shared
`HNDLConditionalReadoutCritic` built by released HNDL 0.8.2. Its resolved graph,
source and implementation identity are bound by the shared checkpoint. The
source-free qualification also executes the shared HyperGAN diversity observer
against actual Qwen responses from two latent seeds per fixed condition, using
HyperGAN develop `a879c4fbff903ccefeba2194bcc7a13fe7b5abb7`. This is a detached
observation and leaves game/RNG state unchanged; it is not a loss or quality gate.
