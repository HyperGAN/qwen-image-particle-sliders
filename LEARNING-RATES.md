# Learning rates

Current finished-output decision and sole selection method:
[scoped learning-rate guide](docs/finished-output-learning-rates.md) and
[portable rates/case scores](docs/finished-output-rate.json). Use RAW weights,
no weight EMA, equal-case intended-effect ON−matched OFF, and reuse completed
work; no additional fresh confirmation. The option/default and CPU integration
descriptions below retain their own source-cohort scope.


Current training accepts optional `lr` (generator/base rate), `d_lr_mult` (critic multiplier), and `prior_lr_mult` (routed particle-bank multiplier). Omitted values retain the existing recipe defaults. Inputs must be finite positive numbers; booleans are rejected. The shared ParticleGAN recipe constructs every optimizer: D = `lr * d_lr_mult`; particle bank = `lr * prior_lr_mult`.

For the default `bispf_optimized` preset, this product keeps G = **0.012**, D multiplier = **1.5**, particle multiplier = **2.5**. These are defaults, not a claim of native LR calibration. Selecting `bispf` uses that official preset's existing values (base 0.00425, D multiplier 1, particle multiplier 2). 

Common interface: put the same keys in session `options`, for example `"lr": 0.00002, "d_lr_mult": 1.5`, `"prior_lr_mult": 2.5`. The product training CLI accepts `--lr` and `--d-lr-mult` and `--prior-lr-mult`. Qwen retains `g_lr` / `--g-lr` as aliases of `lr`; equal aliases are accepted, differing aliases are rejected.

Rates become part of the exact shared recipe and checkpoint configuration. Start a fresh cohort/output for a changed rate; resume requires the original rate recipe. Do not relabel a historical packet. The separate closed HNDL build retains its pre-LR borrowed modules through checksum-verified `cohorts/pre-lr-hndl` archival build routing; this change does not retune that cohort or rewrite its source hashes.

For the current saved Warm Amber comparison, G0.012 RAW32 is best tested.
Rank only the equal-case arithmetic mean intended-effect ON−matched OFF in
finished images. Retain negative gains, zeros and ties; technical failures are
missing evidence, not zero. Teacher/student relative SSE is diagnostic only;
older FP64 subtraction, teacher-energy normalization and equal-example pooling
instructions describe that diagnostic, not the current selector. Preservation,
beauty, confidence, net displacement and completion cannot select or break ties.
Reuse the existing completed search/RAWs/outputs/first scores, with no additional
fresh confirmation. Exact source/recipe identity, legitimate FULL continuation
and RAW export semantics remain required; tiny CPU checks are not native
finished-output evidence. The linked upstream historical contract does not
supersede this current selection policy.
