# Remaining real-model validation

Run only after checking storage, host RAM, selected GPU headroom and other users'
active jobs. Do not stop existing experiments or equate synthetic tests with model
quality. Initial validation has not downloaded the 30.84 GiB weight snapshot.

1. Stage and hash-verify the exact official snapshot. Record hardware, runtime versions,
   peak host/GPU memory, snapshot identity and exit status.
2. Run two edit-bearing pairs plus one identical-prompt control, 512×512, one retained
   state per pair, batch one, two native trajectory steps, two training updates:
   `qwen-slider-train --model-dir <snapshot> --device cuda:0 --prompts <bounded-card>
   --trajectory-steps 2 --records-per-pair 1 --steps 2 --gradient-checkpointing --outdir <run>`.
   Bound the process with a supervisor timeout (for example 10 minutes). This checks
   loading, encoding, finite velocities/gradients and export, not convergence.
3. Generate the same prompt/seed at strength 0 and 1, 512×512, two inference steps.
   Confirm strength-zero latent equivalence against an independent vanilla pipeline,
   successful PNG/RGBA decode, exact target names and finite output. Low-step samples
   do not establish image quality.
4. Check CUDA resume on the same runtime/device and compare state/artifact hashes.
5. Separately train and assess a held-out prompt/seed bank at the full intended budget.
   Compare semantic edit direction, identity/composition preservation, controls and
   interpolation. Record failures; do not call a synthetic check a quality benchmark.

This release makes no memory-fit, commercial-use, full-weight compatibility or
convergence claim. Native fixture success shows model API/gradient wiring only.
