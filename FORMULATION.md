# Shared formulation and native 2.1 wiring

`winning_formulation()` at core `20fdf98d25455582e7e7191eef8ef51cf73173ae` selects
gmix and the **provisional** `particle-gmix-1600-v2` overlay. `stamp.require()`
rejects drift; only declared batch and optional generator learning rate change as
permitted model surfaces. A later crowned overlay is adopted by bumping the core
pin, without replacing product wiring or keeping a local set of game constants.
`stamp.bridge()` supplies every routed MLP. `FormulationGame` supplies the critic,
paired adversarial losses, Gaussian schedule, cap, particle VIC and both optimizers.

The product inserts one down → routed map → zero-initialized up residual at each
`transformer_blocks.i.attn.to_out.0` projection (32 for the official checkpoint).
Base transformer, text encoder and VAE remain frozen. Particles are registered once
in the game and referenced through an unregistered hook context. Inference uses
exactly those same maps and the shared EMA particle bank. Strength 0 returns the
original projection tensor without arithmetic; strengths 0–1 are the stamped range.

At a captured neutral-trajectory state (x,t), the desired teacher is the frozen
positive velocity at that same x,t. The student uses the neutral encoding:

```
e = P_step(v_student(x,t,neutral) - v_base(x,t,positive)) / paired_edit_scale
```

The native pipeline owns the neutral trajectory, latent packing, dynamic shift and
Euler scheduler. The native transformer emits a joint text/image sequence, so the
bridge takes the final target-image token suffix, exactly as the pipeline does.
The released protocol uses CFG=1 and no image conditioning or prefix KV caching.
Each trajectory retains a bounded, evenly spaced set of states on CPU. Each batch uses sample-wise native forwards to preserve variable-length
text/image RoPE layouts; adding text padding would change image positions.

`P_step` is a rank-eight Rademacher sketch of **every** packed latent coordinate,
with entries +/-1/sqrt(number_of_coordinates). The seed is
`31415 + 1000003*step`, fixed across one D/G update and changing deterministically
between updates. In expectation, projected mean-square error equals full latent
mean-square error. Finite sketches and finite training do not prove image quality.
The sketch does not receive the game's Gaussian noise and cannot cancel that noise
by conditioning on it. There is no extra MSE objective.

For each non-control teacher edit of RMS r, calibration contains the balanced rows
`+/-sqrt(rank)*r*basis_j`. Their first/second moments equal those of fresh sketches:
zero mean, covariance r²I, row RMS r. The unchanged core normalizer owns its floors,
median RMS gain and resulting noise calibration. Identical-prompt controls are
trained but excluded from calibration; at least two nondegenerate edit records
are needed. Permanent spatial pooling and fixed-coordinate whitening are avoided.

The hook context stays active for the complete core update and backward, including
native non-reentrant gradient checkpoint recomputation. The base model is held as
an unregistered reference, so the game optimizer owns only product projections,
routed maps and particles. Raw resume state is separate from FP32 EMA exports.
