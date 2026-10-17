# Shared PG0.9.1 game and native Qwen wiring

The public training command executes `particle_sliders.pg09.ConditionalPG09`.
`QwenImage21ResponseAdapter`, `ResponseRow`, `image_condition_features` and
`ConditionalReadoutCritic` come from the installed shared core. There is no product
implementation of adversarial losses, penalties, prior regularization, optimizers,
row sampling or checkpoint restoration.

The architecture still uses `winning_formulation().bridge()` for each routed
attention-output residual and its 128x4 particle bank. The architecture stamp's
historical gmix training fields do not select the new game. The actual objective
is the released ParticleGAN 0.9.1 `bispf_optimized` recipe, or explicitly selected
`bispf`. `--g-lr` overrides the public recipe's **base learning rate**, including
its derived critic and prior rates. The resolved recipe is recorded in exports
and bound in checkpoints. No reconstruction/MSE loss or legacy FormulationGame
runs through the training CLI.

The native pipeline supplies bounded captured neutral-trajectory states. Its own
scheduler controls timestep shifting and its own packing controls spatial geometry.
The product retains exact neutral/positive text encodings, masks (including `None`)
and raw scheduler times. The shared adapter reversibly restores a true
`[1,C,1,H,W]` VAE field, then evaluates the neutral-conditioned student and frozen
positive teacher at the same latent/time. No rank-eight error sketch remains.
Variable text prefix lengths retain their individual native forwards. Only
text-to-image, CFG=1 and uncached native attention are supported.

The shared game adds the normalized native student-minus-target field to Gaussian
reference samples and applies the released adversarial losses and public critic
penalty. The native field reaches that penalty without arbitrary reshaping. A
median positive noncontrol edit RMS supplies the fixed response scale. Identical
prompt controls remain training rows but do not dilute scale calibration. At least
two nondegenerate edit records are required. The critic receives globally checked
full-conditioning sketches plus time/CFG/strength coordinates from the shared API.

Generator and critic use the shared balanced row samplers and independent stored
RNG streams. The exact participating particle Parameter is registered once in the
shared adapter network and reused by every routed hook and the public prior. Only
FP32 adapter/particle and critic leaves enter optimizers; the base stays frozen.
Native checkpointing captures each forward's adapter context through the shared
checkpoint helper, including repeated backward. The product restores checkpoint
functions/flags and removes its hooks on both success and failure.

Training uses a fixed `--steps` recipe budget. `--stop-after` writes a resumable
checkpoint before that budget is exhausted; resume retains the same budget and
configuration. The shared checkpoint binds recipe, all native rows/conditions,
source/runtime/model identity, optimizer and penalty state, row coverage and RNG.
The product additionally validates its EMA state before restoring the game.

EMA safetensors exports remain separate from raw resumable checkpoints. Inference
loads the exported maps and particles into the same hooked bank, then uses the
native Qwen pipeline. Strength zero bypasses each hook exactly; positive strength
adds that fraction of the nonlinear residual. This is not an ordinary LoRA.

The new artifact/protocol format is v2. Older gmix/sketch artifacts and checkpoints
are rejected rather than relabeled. Historical CUDA measurements in this repository
belong to the former pinned implementation. New PG0.9.1 pretrained/GPU, full sampler,
quality and convergence acceptance remain unqualified.
