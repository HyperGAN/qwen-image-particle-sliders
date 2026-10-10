# Remaining real-model validation

Run only after checking storage, host RAM, selected GPU headroom and other users'
active jobs. Do not stop existing experiments or equate synthetic tests with model
quality. A no-download 128×128 official-weight smoke has passed; see
[the RGBA receipt](real-model-smoke.json), [cap/resume receipt](cuda-cap-resume.json)
and [VALIDATION.md](../VALIDATION.md). Native fourth-step cap backward and exact CUDA
resume have also passed in that bounded runtime.
The checks below extend that evidence to the intended resolution and training budget.

To repeat the bounded correctness check, with fresh resource checks and a new output
directory, use the same package/runtime environment and an already staged snapshot:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  timeout --signal=TERM --kill-after=10 610 python scripts/check_cuda_resume.py \
  --model-dir <verified-snapshot> --device cuda:0 --outdir <new-validation-dir>
```

The script fixes real-model resolution at 128×128, uses a 10-minute internal deadline,
34 GiB GPU allocator limit, 18 GiB process RSS guard and 4 GiB available-RAM floor.
It loads weights once, runs four updates versus two-plus-two resumed updates,
observes the native full-coefficient penalty and compares all state fields, EMA
bytes, step inputs, RNG fingerprints and metrics exactly. It exits on any mismatch.
`--dummy` runs the same assertions on a 32×32 native random CPU fixture; that is the
CI check, not official-weight CUDA evidence. Other jobs must retain headroom.

1. Stage and hash-verify the exact official snapshot. Record hardware, runtime versions,
   peak host/GPU memory, snapshot identity and exit status.
2. Run two edit-bearing pairs plus one identical-prompt control, 512×512, one retained
   state per pair, batch one, two native trajectory steps, two training updates:
   `qwen-slider-train --model-dir <snapshot> --device cuda:0 --prompts <bounded-card>
   --trajectory-steps 2 --records-per-pair 1 --steps 2 --gradient-checkpointing --outdir <run>`.
   Bound the process with a supervisor timeout (for example 10 minutes) and memory
   limits that leave other jobs headroom. The 128×128 run used a 34 GiB PyTorch GPU
   allocation cap, an 18 GiB process RSS guard and a 4 GiB available-host-RAM floor.
   Its measured memory use is not a guarantee for 512×512. This checks loading,
   encoding, finite velocities/gradients and export, not convergence.
3. Generate the same prompt/seed at strength 0 and 1, 512×512, two inference steps.
   Confirm strength-zero latent equivalence against an unadapted vanilla pipeline call,
   successful PNG/RGBA decode, exact target names and finite output. Low-step samples
   do not establish image quality.
4. At the larger resolution, repeat at least four updates and matched checkpoint/resume
   to revalidate the native lazy cap path, memory and exact state/artifact comparisons.
5. Separately train and assess a held-out prompt/seed bank at the full intended budget.
   Compare semantic edit direction, identity/composition preservation, controls and
   interpolation. Record failures; do not call a synthetic check a quality benchmark.

This release makes no broad memory-fit, commercial-use or convergence claim.
The real smoke establishes one bounded full-weight configuration; native fixtures
cover additional API/gradient paths without establishing model quality.
