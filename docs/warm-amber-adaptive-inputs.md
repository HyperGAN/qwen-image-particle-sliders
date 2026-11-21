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
replay on the live scheduler. The fixed Warm Amber scores remain the available
finished-image evidence. [Per-case fixed comparison](warm-amber-raw192-native-schedule-comparison.json)
