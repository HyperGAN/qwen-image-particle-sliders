# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Train512px default no checkpointing | 1 | 108 | 32 | 512×512; ≤512 tokens; B=2; BF16 frozen; FP32 routed adapter/critic/activation envelope; no checkpointing |
| Train512px explicit checkpointing | 1 | 56 | 32 | 512×512; ≤512 tokens; B=2; BF16 frozen; FP32 routed adapter/critic/activation envelope; checkpointed |
| Train512px dense fallback | 1 | 196 | 32 | 512×512; ≤512 tokens; B=2; BF16 frozen; FP32 routed adapter/critic/activation envelope; no checkpointing |
| Sample512px full pipeline | 1 | 52 | 32 | 512×512; ≤512 tokens; B=1; BF16 frozen; FP32 routed adapter/critic/activation envelope; no checkpointing |
| CPU FP32 sample512px | 0 | 0 | 184 | 512×512; ≤512 tokens; B=1; FP32 all weights on CPU; no checkpointing |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. They describe the finite profiles above, one worker and one learned slider, with RAW weights and **no model-weight EMA**. A larger image, token sequence, image/reference count, logical batch, cache or number of simultaneous workers needs a new calculation; there is no finite universal maximum for unbounded inputs.

## Scope and residency

Base: **`Qwen/Qwen-Image-2.1@d26bb61231c349cf6b7896fa83353113880e1ba3`**. Runtime uses device_map=cuda with low_cpu_mem_usage=True and retains DiT, full text/vision encoder and VAE on the same GPU. Host planning allows two largest serialized shards plus frozen row/prompt caches; it does not assume a full simultaneous CPU pipeline copy. The CPU path converts all components to FP32 and is separately budgeted.

Rows bind text-only prompts≤512 formatted tokens, native512px, logical batch2 and≤32 frozen rows. The public text encoder does not cap arbitrary prompts here: longer inputs need a new calculation. Sampling use_kv_cache=False; conservative text KV/hiddens are still included below. No quantized loader is present. Existing MIN_GPU_BYTES38/host16 policies are admission constants, not these calculated requirements.

The table's efficient-SDPA rows require a flash/efficient implementation. An automatic backend can fall back to dense math; use the dense row where provided, or add the dense term below before planning capacity. A planning row does not change code, precision, optimizer, rates, initialization, checkpoints or qualification.

## Frozen and trainable ledger

- DiT serialized upper bound: **14,230,284,408 bytes** (13.253 GiB).
- text encoder serialized upper bound: **17,534,339,488 bytes** (16.330 GiB).
- VAE serialized upper bound: **1,350,989,512 bytes** (1.258 GiB).

The locked seven weight files total33,115,613,408 bytes; use that as a conservative BF16 resident bound including headers. The transformer is32×4096,32 heads and FFN ratio3; the8.77B text encoder remains loaded even for preencoded sampling.

Trainable allowance: **3,419,137 parameters**. 32×(8×8192+6476)+512 bank + readout16×(65,536+4,102)+16+17 = 3,419,137; HNDL spatial critic is a separate option. FP32 trainable weights and gradients require `4A+4A` bytes. Frozen base parameters have no gradient or optimizer tensor allocation. Routed particle-bank parameters, where present, are counted once; a frozen response cache is not a trainable bank.

The current `bispf_optimized` PG0.9.5 recipe uses **DualNorm, momentum 0, direction_blend**; it is not assumed to be Adam. Persistent per-parameter moment tensors are0 in this profile; scalar counters are small. The peak ledger additionally reserves **`48A+32MiB` bytes** for protected normals, displacement/direction copies, FP64 promotion and thin-matrix SVD/Gram workspace. This is a conservative temporary-allocation allowance, not a claim that the optimizer uses no memory. Selecting historical `bispf`/Adam instead adds its actual state (typically another`8A` bytes for two FP32 moments); nonzero momentum or a different solver requires recalculation. Weight EMA adds0 bytes and must stay disabled.

## Numeric peak ledger (GiB before headroom)

| Profile | Frozen GPU weights | Activation / decode | Trainable + optimizer | Other GPU allowance | Host loading/cache subtotal |
|---|---:|---:|---:|---:|---:|
| Train512px default no checkpointing | 30.841 | 51.000 | 0.210 | 0.000 | 19.567 |
| Train512px explicit checkpointing | 30.841 | 9.188 | 0.210 | 0.000 | 19.567 |
| Train512px dense fallback | 30.841 | 123.000 | 0.210 | 0.000 | 19.567 |
| Sample512px full pipeline | 30.841 | 9.000 | 0.013 | 0.000 | 19.567 |
| CPU FP32 sample512px | 61.683 | 9.000 | 0.013 | 0.000 | 142.662 |

CPU rows place their active tensors in host RAM; their frozen/activation entries are not GPU residency. Sequential phase sub-ledgers are in the source-facts report.

Cache bound for the training profile: **32 rows**, each with three FP32 native fields, plus **32 cached prompts**. Tensor-only cache allowance: `25,165,824+134,217,728` bytes; masks/indices and serialization overhead remain covered by the explicit host subtotal and reserve. These bounds are inputs to this calculation, not new runtime limits.

## Activation, cache and host calculation

The auditable inputs and per-profile subtotals are in [source-facts.json](docs/memory/source-facts.json). For a transformer envelope, let `N=image tokens+bounded text tokens`, `D=hidden width`, `I=FFN width`, `H=attention heads`, `L=covered blocks`, and `B=simultaneously retained native graphs`:

- One-block inference allowance: `4B(16ND+6NI)` bytes. The4-byte coefficient deliberately covers FP32 intermediates even for BF16 weights;16 and6 allow Q/K/V/output, normalization/residual, gated FFN and adapter intermediates.
- Uncheckpointed training allowance: `4LB(16ND+6NI)`; checkpointed allowance: `4(4LBND+2B(16ND+6NI))`. Checkpointing saves layer inputs and recomputes block internals; it does not remove the frozen model, adapter or critic.
- Dense math fallback adds `8BH N²` bytes for inference and `16LBH N²` for uncheckpointed training, or `32BH N²` for the checkpointed current-block envelope. These are conservative materialized-score/probability allowances; flash/efficient SDPA avoids that quadratic storage. GQA expansion uses the query-head count for this bound.
- Decode allowance is separately based on bounded output/tile area and channel width; profile notes record the bound. Peak is the **maximum of sequential phases**, with the actual overlapping model weights added inside each phase. Sampling steps are sequential and do not multiply resident peak.
- Frozen CPU rows are bounded by the profile (three FP32 native fields per cached row, plus explicitly bounded prompt embeddings). Loader staging/serialization copies and the largest published shard are included in host subtotals. Host cache/page residency and kernel workspaces vary, so these are planning allowances, not an allocator theorem.

Final GPU budget is `ceil_to_4GiB(1.25×GPU subtotal+2GiB)`; host is `ceil_to_4GiB(1.25×host subtotal+4GiB)`. The explicit25% plus runtime reserve covers allocator fragmentation, CUDA context/libraries, kernel workspaces, small FP32 islands and Python/serialization overhead. These chosen reserve multipliers are explicit planning assumptions; architecture metadata alone cannot prove every allocator or fused-kernel peak. It is not borrowed from scheduler admission caps, card size or global free-memory floors. Do not add every row's peak together for sequential phases; parallel workers do add their simultaneous requirements.

This profile's text-state allowance is **230,686,720 bytes** (0.215 GiB): bounded text KV plus output_hidden_states. It is already included in the phase or host allowance, not added twice. Text-only encoding has no generated-token loop. If the actual encoder returns KV state, it is explicitly included as `2×layers×B×KV_heads×tokens×head_dim×dtype_bytes`; cache-disabled encoders allocate no persistent autoregressive KV. Recurrent/linear-attention state is accounted separately where applicable. Arbitrary output-token generation and image-conditioned encoders are outside these profiles.

## Sources and reproducibility

- [src/qwen_image_sliders/backend.lock.json](src/qwen_image_sliders/backend.lock.json)
- [src/qwen_image_sliders/runtime.py](src/qwen_image_sliders/runtime.py)
- [src/qwen_image_sliders/adapters.py](src/qwen_image_sliders/adapters.py)
- [src/qwen_image_sliders/train.py](src/qwen_image_sliders/train.py)
- [src/qwen_image_sliders/scoped/cache.py](src/qwen_image_sliders/scoped/cache.py)
- [Captured architecture, published file/index sizes and arithmetic](docs/memory/source-facts.json) retains exact model revisions, metadata SHA256s and direct Hugging Face config URLs. No weights were downloaded or opened for this calculation.
- Optimizer facts: the pinned ParticleGAN0.9.5 `recipes.py` (`bcap`/`bispf_optimized`, lines1435–1451), `optim/dualnorm.py` momentum/state branch, and `optim/direction_blend.py` FP64 normals. These are part of the dependency selected by the repository's existing core lock; their source facts are recorded in the combined calculation report.

Documentation was calculated from source and published metadata on 2026-11-14. No model construction, import of Torch/Diffusers, forward pass, benchmark, sampling, training or new GPU work was performed.
