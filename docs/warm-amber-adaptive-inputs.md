# Warm Amber adaptive fit inputs

The local `vibe-validation-20261117/qwen-image-native-vibe-v1/collection-result.json`
manifest has SHA-256
`9442bd9a2a94d630f2c985d0c3cd356c3262a811964127d9902a1d62b705cfc3`.
It records eight matched prompt pairs at seeds 923501 through 923504. Each
`rows[i].neutral.output.sample` and `rows[i].positive.output.sample` receipt
binds a native `latents.pt` with BF16 packed `[1,1024,64]` states from the
same 512-pixel Euler40, CFG1, uncached profile. Use
`teacher_states[*]["kwargs"]["hidden_states"]` and retain the recorded
indices and raw timesteps. The `finished` tensor is after the terminal update. The native base revision is
`d26bb61231c349cf6b7896fa83353113880e1ba3`.

`teacher-fields-result.json` in the same collection binds sixteen saved target
and neutral velocity references for diagnostics. Its SHA-256 is
`b0557613e76ac24cff36cef9304f1fd79825b75b3d19a22f52eabac9d787effc`.
The paired prompt capture can inform a packed-latent direction. It covers two
positions per trajectory. Calibrate the gate separately on active trajectories
with the same RAW adapter and native sigma grid.

Keep the archived RAW192 cases `dev0-seed204`, `dev0-seed205`, `dev1-seed203`,
and `dev1-seed207` separate from direction fitting and active calibration. Their
fixed-strength finished outputs bind RAW SHA-256
`3132d426e96f94e358b9ff4efa2c2c08abed51be93e997c9a223877cfba39c1e`.
The saved RAW192 workflow uses its captured ParticleGAN 0.10.1 native owner.
Replay historical tensors in that owner. The current scoped package tests
use a tiny software fixture and establish exact OFF and recorded-strength
replay on the live scheduler. [Per-case fixed comparison](warm-amber-raw192-native-schedule-comparison.json)

## Warm Amber RAW192 native pilot

The pilot used the saved RAW192 adapter at nominal strength 2/3 on the original
Qwen Image 2.1 base. Each 512 × 512 image used CFG 1 and 40 native FlowMatch
Euler intervals with K/V caching disabled. The resident sampler reproduced an
archived constant-strength terminal latent from each held-out scene family
bit for bit.

The direction fit used all 40 native inputs from eight trajectories across
four paired prompt families. The four bicycle-workshop pairs were excluded
before fitting to keep the bicycle evaluation family separate. The ten
features selected by the paper’s signed contrast rule all had positive
measured contrast. The mug and chair cases came from a separate archived
Qwen Warm Amber prompt roster, with fresh native CUDA encodings. Live
calibration targeted the paper’s 50% intervention rate. CFP fired on 40 of
80 calibration intervals. CLABS fired on 41 of 80, within one interval of
the target. [Right Token, Right Time](https://arxiv.org/html/2611.18274v1#A2)

Each held-out case used the archived OFF image and a constant control matched
to the gate’s realized sigma-integrated strength. The reported gain is the
equal-case mean of finished Warm Amber intensity ON minus matched OFF.

| Allocation | Gate gain | Matched constant gain | Wins | Held-out firing |
| --- | ---: | ---: | ---: | ---: |
| CFP | +0.003571 | +0.002653 | 2/4 | 136/160 |
| CLABS | +0.003554 | +0.002793 | 2/4 | 138/160 |
| Random | +0.001443 | +0.001578 | 1/4 | 78/160 |

Both gates fired on all 40 intervals of each bicycle case and reproduced the
archived constant outputs exactly. Bicycle alignments stayed below the CFP
threshold and CLABS held a firing probability of 1 throughout. Both gates
improved the two teapot cases over their own matched constants. The held-out
rates for both adaptive gates exceeded the 50% target on this panel. All gate
and constant outcomes and their matched OFFs passed the independent categorical
integrity check. Realized dose varied across gates, so compare each row with its
own constant. The measured early-heavy profile remains the recommended Warm
Amber RAW192 profile. [Exact settings and per-case evidence](warm-amber-raw192-adaptive-study.json)
