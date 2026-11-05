> Current source candidate: released PG0.9.5/core0.13.0, fresh cohort. Original PG0.9.4 loader/conditioning/control receipts remain historical; no PG0.9.5 pretrained qualification is claimed.

# Qwen-Image 2.1 strict native candidate

This separate opt-in package prepares current native learning-rate experiments.
The release package, historical checkpoints and closed HNDL cohorts retain their
original source and installation routes. No pretrained execution or rate is
qualified by the CPU evidence for this candidate.

Two explicit profiles retain distinct numerical identities:

| CPU encoder | DiT and VAE | Profile |
| --- | --- | --- |
| BF16 | CUDA BF16, math SDPA | `qwen-image21-cpu-bf16-pread-encoder-cuda-bf16-dit-vae-math-v2` |
| FP32 from original BF16 weights; native hidden output explicitly rounded to BF16 | CUDA BF16, math SDPA | `qwen-image21-cpu-fp32-pread-encoder-rounded-bf16-cuda-bf16-dit-vae-math-v2` |

FP32 conditioning does not claim identical output to BF16 conditioning. Each
factory verifies all original assets, loads and checks every actual CPU encoder
parameter against the declared cast, encodes only declared texts with the
unmodified native pipeline, releases the encoder, then loads and verifies the
owned CUDA DiT and VAE. Keeping the VAE on CUDA preserves the native pipeline's
execution-device selection, sampler and decoder. Source FP32 VAE weights are
explicitly cast to the canonical CUDA BF16 precision.

Current source pins are base `a0090e51a819adaae2890038767dd39c9a231928`, core
`3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official ParticleGAN 0.9.5, and
Diffusers `1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458`. The source and runtime
closures refuse changed implementation bytes before native allocation.

The native response is `[1,64,1,32,32]` at 512 pixels, with 4096-wide native text
conditioning. Each row retains its actual full-prefix length. The existing
product bank supplies 128 four-dimensional particles and the shared PG game
owns all GAN math and optimizer/checkpoint behavior. Generator, critic and bank
rates use `lr`, `d_lr_mult` and `prior_lr_mult`; EMA defaults to disabled.

The proposed unmeasured GPU window is 24 GiB total/23 GiB allocator with a 5 GiB
global floor. The DiT plus BF16 VAE weight floor is approximately 13.882 GiB;
training/decoder peaks remain unknown. CPU FP32 encoder staging alone has a
32.66 GiB parameter floor. A separately reviewed first CPU benchmark requires
48 GiB process RSS, at least 72 GiB host availability at admission, a 24 GiB host
floor, four CPU threads, no CUDA, and exactly one prompt. These are resource
policies, not measured fit guarantees or authorization to launch.

Strict policy requires deterministic errors, highest matmul precision, TF32 and
reduced-precision reductions disabled, deterministic cuDNN, math-only Torch
SDPA and the actual native Diffusers dispatch function. Every native forward
and update rechecks caller flags. Initialization scopes the CPU generator and
only the owned CUDA generator, restoring caller RNG without global seeding.

Native inference accepts numerical responses and native latent/RGBA rendering;
it constructs no teacher rows or game. Full pretrained response/render/resume
qualification remains pending. An artifact requires its exact exported source,
precision profile and declared prompt set; training cache receipts are never
used as substitutes for fresh current native parity.

For bounded CPU checks, use the pinned independent environment:

```bash
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -m pytest -q tests
```

The first random-weight native encoder tests use the already staged canonical
processor/tokenizer metadata. They do not open pretrained weight payloads.

Version0.1.1 uses the native public `from_pretrained(None, config=...,
state_dict=...)` with lazy safetensors `PySafeSlice` values from positioned-read
handles. The first full CPU mmap-loader attempt at ec9cd3c exceeded48GiB RSS
during loading and was stopped before complete weight verification or encoding;
its v1 artifacts/environment/receipts remain preserved. No retry is implicit.
The new positioned-read profile requires its own source/environment/frozen plan.

Actual tiny BF16 and FP32 positioned-read loads match the actual native mmap
loader bit for bit, including original full-prefix output and BF16 transport.
A model constructed and then cast is an invalid oracle: it also rounds native
nonpersistent rotary buffers. The new identity records/checks all registered
buffers, plain tensor/rotary attributes, actual config precision and caller
default dtype/device. Tensor hashing streams existing CPU memory without
embedding-sized byte copies. These checks do not establish full-model memory
fit or authorize another pretrained load.

Version0.1.2 separately restores the original native checkpoint function and
instance flags when the binding closes. Cleanup attempts every owned release
action even if a hook removal fails, and can retry failed actions. It moves
retained parameters, registered buffers, response rows, conditioning, nested
optimizer history and pending PG state to CPU without replacing their tensor
aliases; interrupted native VAE
feature caches are cleared. Unrelated live models and hooks are left alone.
Tiny actual native/game CPU checks cover these paths and constructor failures
after a model has returned. A loader exception before it returns a model still
requires the dedicated worker process to exit and a verified unload receipt.
Full GPU cleanup and pretrained execution remain unqualified. The frozen
v0.1.1 one-prompt CPU benchmark keeps its own original source and environment.

Version0.1.3 uses the common cached-row `session.evaluate(index=...,
split='train', strength=...)` contract for training sessions. Optional staged
`evaluation_pairs={'development': {'rows': [...], 'seed': 202}}` captures
observation-only native teachers from declared prompt strings, then prepares
and snapshots them through the same shared adapter. Geometry, native sampler
horizon and records per pair inherit the training configuration. Context
tensors, caller identities, undeclared prompts and budget overrides are refused.
Prepared teacher/OFF fields must exactly equal the native pipeline captures;
split checkpoint identity derives from actual source, texts, seed and tensors.
Observation rows are released with the owned binding. This adds native work
during construction; it does not reuse an external encoder/cache assertion.

The separate inference backend retains native requests through
`open_inference(..., operation='response', native_request=...)`. The immutable
earlier benchmark sources, environments and artifact receipts remain separate.

Version0.1.4 accepts the common public inference `seed` argument for native
numerical responses. The response has no stochastic sampling; default0 and
explicit bounded integer seeds produce the same field and preserve caller RNG
and network state. Invalid seeds and unknown response fields are refused before
the native forward. The public factory takes no seed, so this validation occurs
when calling the response, after opening its runtime. The regression uses actual
`read_inference` → `open_inference(operation='response')` →
`InferenceSession.response` with training constructors poisoned. Earlier direct
binding checks did not qualify this public path; the v0.1.3 refusal receipt is
retained. Full pretrained/GPU public inference remains unqualified.
