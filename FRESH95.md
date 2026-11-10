# Opt-in cached fresh initialization for Qwen-Image 2.1

`qwen_image_sliders.fresh95` is an additional Python API for a deliberately new cohort. Four installed CPU software checks passed, including one tiny original-game update, strict FULL resume, RAW response inference and cleanup faults. They used random tiny native geometry and synthetic completed data. They do not qualify a pretrained model, CUDA execution, finished-image quality, public packaging or an available native completed-response cache. Existing CLIs, saved LR recommendations, defaults and old cohorts retain their original behavior.

## Required inputs and ownership

Use the exact original canonical `Runtime` (or the explicitly synthetic `TinyRuntime`) and original pinned core0.13.0/ParticleGAN0.9.5/native Source closure. The API keeps the complete core577/PG68 guards and requires a genuine owning `qwen-image-particle-sliders`0.3.0 installation whose generated RECORD owns all nine new namespace modules. A source overlay, a generic newer dependency set or handmade metadata is insufficient. No new command or registration is added; public build/install validation of this addition remains separate from the installed CPU result.

The caller loads and owns the native runtime under its own resource admission. `owner_admission(request)` must authenticate the requested operation and retained native/cache/Source provenance and return exactly `True` before any adapter/bank/critic attachment. It does not load or recover teachers. Do not replace that authentication with an unconditional approval for native work. The runtime is borrowed; closing restores only owned hooks/settings. An optional transferred `release()` callback runs after cleanup succeeds. A cleanup exception retains `cleanup_owner`; retry its `.close()` without replaying an update or inference.

Training requires an already completed, authentic bank of original `PreparedResponse`/`ResponseRow` objects, a whole-cache byte/SHA pin, and separately retained matching provenance. This is not the request-only packed teacher record file accepted by older entrypoints. Missing or mismatched completed provenance is refused before attachment; the API never calls `adapter.prepare`, an encoder, a teacher collector or a native trajectory to fill a gap. Native cache availability remains unestablished by the software checks.

The cache preserves full FP32 `[1,C,1,H/8,W/8]` VAE fields, raw time0..1000, true CFG1, negative=neutral text context, original embedding/mask/image-pad-mask geometry and exact native/Source/per-row origins. The original median noncontrol edit calibration is retained. `save_prepared_cache(...)` serializes supplied completed data and binds provenance; it does not establish that arbitrary caller claims are authentic. Synthetic origins remain explicitly synthetic.

For a separately retained cache binding:

```python
from qwen_image_sliders.fresh95.cache import CacheBinding
from qwen_image_sliders.fresh95.util import Checkpoint

completed_cache = CacheBinding(
    checkpoint=Checkpoint(
        path=cache_metadata["path"],
        bytes=cache_metadata["bytes"],
        sha256=cache_metadata["sha256"],
    ),
    provenance=retained_completed_provenance,
)
```

`cache_metadata` and `retained_completed_provenance` above are authenticated retained inputs, not values inferred from a request. To serialize a genuine existing bank, the exact helper signature is `save_prepared_cache(path, *, runtime, config, prepared, teacher_origins, cache_id)`.

## Fresh, FULL resume and RAW inference

Construct `config` with `configuration(**original_options)`. It accepts the original rows, fixed horizon, seed, batch, geometry, trajectory/record counts, checkpoint choice, preset and explicit rate overrides; set `ema_decay=None`. Keep those values and native runtime identity fixed across restore. Choose rates from the existing documented results when applicable; this initialization profile adds no LR search or new recommendation.

```python
from qwen_image_sliders.fresh95 import configuration, open_training, open_inference

config = configuration(**dict(original_options, ema_decay=None))
run = open_training(
    runtime, completed_cache, new_run_directory,
    mode="fresh", cohort_id=new_cohort_id, config=config,
    owner_admission=admit_cached_operation, ema_decay=None,
)
try:
    metrics = run.update()
    full = run.save()                 # New FULL at the returned logical step.
    raw = run.export(new_raw_path)    # Current RAW weights, no EMA.
finally:
    run.close()

resumed = open_training(
    runtime, completed_cache, new_resume_directory,
    mode="resume", resume=full, cohort_id=new_cohort_id, config=config,
    owner_admission=admit_cached_operation, ema_decay=None,
)
try:
    pass                            # Restore only; no update is implied.
finally:
    resumed.close()

inferred = open_inference(
    runtime, raw=raw, owner_admission=admit_cached_operation,
)
try:
    velocity = inferred.response(original_response_row, strength=1.0, seed=7)
finally:
    inferred.close()
```

Owners use explicit `.close()`; they are not context managers. Use fresh immutable output directories and artifact paths. Fresh refuses a saved packet; resume requires an exact new-profile FULL. Ordinary original component constructors still run. Fresh then applies public PG0.9.5 deterministic orthogonal initialization once before optimizers, preserving original Parameter objects and native weights. The active AdapterBank down constructor uses Kaiming `a=sqrt(5)`: its target variance is `1/(3*fan_in)`. Down QR retains that scale; UP remains zero/KEEP. The existing128x4 bank receives its declared R2 initialization at target standard deviation0.02. The original router/readout and ConditionalReadoutCritic remain constructor KEEP. No replacement layer, extra bank or pretrained initializer is introduced. FULL0/RAW0 and the receipt are saved before update1.

FULL resume skips the fresh initializer and restores the whole typed network/bank/critic/optimizers/sampling/exposure/model-RNG state, retaining the original initialization receipt and exact cache identity. It does not migrate or reinitialize a completed legacy cohort. RAW restores only the adapter/bank state and skips cache loading, critic/game/optimizer construction and the fresh initializer. Responses retain native scale0..1 and the original row geometry. The optional `inferred.render(...)` route delegates to the original native consumer; the new profile's pretrained/full-image/public-wheel execution is unqualified by these CPU response checks. No pretrained weights or new finished-output LR result are supplied here.
