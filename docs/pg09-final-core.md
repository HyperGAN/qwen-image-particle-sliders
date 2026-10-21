# Final shared-core pin acceptance

Qwen now pins shared core `cdfb80a76b882b4b73861b12f555263bd64ffeb7`
from [shared draft PR 141](https://github.com/HyperGAN/particle-sliders/pull/141).
The immutable producer wheel SHA256 is
`4abc83988fdeb6d7478de830e7ee8ec5d9bd99c20d52f89e4ce594d50dfd39a3`.
This final delta fixes original Adam generator clocks and sparse critic clock
bounds, and removes the Iris metadata dependency on Diffusers. Qwen retains its
required native Diffusers dependency.

[The installed receipt](pg09-final-core.json), [10-test JUnit](pg09-final-core-tests.xml)
and [fresh-process receipt](pg09-final-core-fresh-process.json) qualify this exact
pin. All **10 focused native checks passed**: invalid original Adam G/D clocks at
zero or beyond the completed update count reject before mutation; initial-empty,
started and honestly unused critic histories restore with exact next updates under
both presets. Tests use the real Qwen model, actual particle bank and native gradient
checkpointing, through the installed public game API.

Fresh Python processes repeated the installed training CLI for both presets:
four uninterrupted updates equal two plus two resumed updates, including complete
checkpoint state and exact EMA safetensors bytes. Exported adapters passed native
latent inference. The new product wheel SHA256 is
`bc880a9aa16bb390acc5f99bbc7d7fd6327ce9b7bad85c443254541fa379cbc4`.

The producer wheel was installed unchanged into the consumer environment. Every
installed shared Python/JSON file and every product Python/JSON file matched its
source manifest. A separate installation from the declared public Git pin matched
the same 32 shared Python/JSON files. Qwen, core, PG, Diffusers and Transformers
imports resolved to installed wheels outside the checkout under isolated Python.
Existing PyTorch/NumPy dependencies were reused; complete fresh dependency resolution
was not repeated locally.

The [previous 64-test baseline](pg09-audit.md) and all its source/wheel/resume/optional
HNDL and HyperGAN receipts remain unchanged and apply specifically to shared `4be0b5b2`.
Those results are not counted as a new full-suite run on `cdfb80a7`. This final check
targets the relevant shared delta and actual Qwen update/restore/export path.

Reproduce the focused tests after installing the pinned dependencies:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q \
  tests/test_native_optimizer_history.py \
  -k 'clock_rejection or initial_empty or valid_started or unused_critic'
python scripts/check_cuda_resume.py --dummy --preset bispf --outdir /tmp/qwen-final-bispf
python scripts/check_cuda_resume.py --dummy --preset bispf_optimized --outdir /tmp/qwen-final-optimized
```

No historical checkpoint or active source cohort was retargeted. No pretrained
model, CUDA context, GPU job, service or permission change was involved. Pretrained
sampler/decode behavior, memory, quality and convergence remain coordinator gates.
The product and shared dependency remain drafts pending exact-head Ada review and
parent-owned merges.
