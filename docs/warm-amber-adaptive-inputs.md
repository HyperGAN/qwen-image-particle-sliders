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

Each held-out case used the archived OFF image and a constant control
calculated from the gate’s realized sigma-integrated strength. The reported
gain is the equal-case mean of finished Warm Amber intensity ON minus matched
OFF. The original table is retained below. The nearest-FP32 follow-up
verified the same finished results for CFP and CLABS.
[Native control receipts](warm-amber-raw192-nearest-fp32-controls.json)

| Allocation | Gate gain | Archived constant gain | Wins | Held-out firing |
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

## Replay the measured CFP plan

The published study supplies the fitted direction, live threshold, nominal
strength, native sigma grid, and per-case seeds. CFP calibrated at 40 of 80
intervals. Its four-case gain over OFF was +0.003571, and its gain over the
nearest FP32 constants was +0.000918. Both bicycle cases used the full
strength throughout. Use the recipe with the saved Warm Amber RAW192 and
the captured ParticleGAN 0.10.1 native owner. The strict
`scoped.open_inference` loader serves typed new-profile RAW exports.
[Published study](warm-amber-raw192-adaptive-study.json) ·
[Reviewed core wheel build](../SCOPED.md#activation-dependent-strength)

The native CUDA multiply uses an FP32 scalar and rounds the residual product
to BF16. On the saved 40-step sigma grid, the two bicycle controls exactly
matched the gate's integrated scalar coefficient. Seed 203 used the nearest
FP32 coefficient with a difference of −1.42 × 10⁻⁸. The seed 207 control
used `0.3183354437351227`, one FP32 step below the closest constant,
`0.31833547353744507`. The original sampler reran seed 207 at the closest
coefficient and produced a byte-identical terminal latent. The finished image,
independent integrity judgment, and effect score were reused. The corrected
integrated scalar dose is +1.04 × 10⁻⁸ from the gate dose before BF16 product
rounding. Use the recorded coefficients for replay. For a new run, use the
shared `match_binary32_interval_dose` with the native FP32 coefficients and
the actual sigma grid.
[Native control receipts](warm-amber-raw192-nearest-fp32-controls.json) · [Shared dose matcher](https://github.com/HyperGAN/particle-sliders/commit/07dbd331c3a548709949032c01ce19e7c64a2520).
[Native adapter](https://github.com/HyperGAN/qwen-image-particle-sliders/blob/0294326f7ef1dee65ef24a44797b1a06476a4940/src/qwen_image_sliders/adapters.py#L65) · [Scalar multiply kernel](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cuda/BinaryMulKernel.cu#L33-L42).

For the measured cases, supply the original BF16 prompt encoding. The
historical owner accepts this encoding as `context`, and returns the native
terminal latent to the original decoder. The public study contains the gate
plan and results. The original checkpoint, native owner, and prompt encodings
come from the archived run.

| Cases | Original prompt |
| --- | --- |
| `dev0-seed204`, `dev0-seed205` | An editorial photograph of a blue bicycle beside a stone wall. |
| `dev1-seed203`, `dev1-seed207` | An editorial photograph of a white porcelain teapot on a linen cloth. |

Run the code from the repository root in the captured owner's environment.
Pass the open archived owner, the archived RAW and encoding files, the local
reviewed core wheel, and one measured case ID. The function checks the
checkpoint, attached adapter tensors, base identity, and native profile.
It returns the latent, exact gate receipt, and the case's pre-rounding
constant strength.

```python
import json
from hashlib import file_digest
from importlib.metadata import version
from math import fsum, ulp
from pathlib import Path

import torch
from qwen_image_sliders import adaptive_inference as adaptive


ENCODING_SHA256 = {
    "dev0": "ca3214f993ad2b23c40bfb1c5f7c2a367fdf8b5b1b89d0e0fcd50c4bb1b7f6ff",
    "dev1": "0825463a674ab5d4ed97d7b87f925f7b597e74f0d09c1399d425bdfe5b102564",
}


def checked_tensor_file(path, expected_sha256):
    with Path(path).open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"Archived artifact hash differs for {path}")
        source.seek(0)
        return torch.load(source, map_location="cpu", weights_only=True)


def measured_gate_inputs(raw_path, encoding_path, core_wheel, case_id, *, method):
    if method not in {"cfp", "clabs"}:
        raise ValueError("Choose a measured CFP or CLABS plan")
    path = Path("docs/warm-amber-raw192-adaptive-study.json")
    study = json.loads(path.read_text(encoding="utf-8"))
    profile = study["profile"]
    calibration = study["calibration"][method]
    case, = (row for row in study["methods"][method]["cases"] if row["case"] == case_id)
    raw = checked_tensor_file(raw_path, profile["checkpoint_sha256"])
    metadata = raw["metadata"]
    base = metadata["model_identity"]
    if (
        (base["model_id"], base["revision"]) != (profile["base"], profile["revision"])
        or raw["completed_steps"] != profile["update_count"]
        or raw["weights"] != "raw" or profile["weights"] != "raw"
        or raw["ema_decay"] is not None or profile["ema_decay"] is not None
        or metadata["true_cfg_scale"] != profile["true_cfg_scale"]
        or metadata["kv_cache"] != profile["use_kv_cache"]
        or (profile["height"], profile["width"], profile["steps"]) != (512, 512, 40)
        or profile["true_cfg_scale"] != 1 or profile["use_kv_cache"]
    ):
        raise ValueError("The checkpoint and archived native sampling profile differ")
    if (study["sources"]["core_commit"], study["sources"]["core_module_sha256"]) != (
        adaptive.CORE_REVISION, adaptive.CORE_MEMBER_SHA256
    ):
        raise ValueError("Use the steering API with the reviewed core source")
    adaptive.gate_core(core_wheel)
    context_id = case_id.split("-seed", 1)[0]
    context = checked_tensor_file(encoding_path, ENCODING_SHA256[context_id])
    if context["prompt_embeds"].dtype != torch.bfloat16 or context["image_pad_mask"].any():
        raise ValueError("Use the archived BF16 text-only encoding")
    direction = torch.tensor(study["fit"]["direction"], dtype=torch.float32)
    if direction.shape != (64,) or not torch.isclose(direction.norm(), direction.new_tensor(1.0)):
        raise ValueError("The measured fit must be a 64-channel unit direction")
    options = dict(
        core_wheel=core_wheel, method=method, strength=profile["nominal_strength"],
        steps=profile["steps"], parameter=calibration["parameter"],
        direction=direction, evidence_ref=study["evidence_sha256"]["capture_report"],
        seed=case["render_seed"], gate_seed=int(case["gate_seed_decimal"]),
    )
    if method == "clabs":
        options["clabs"] = calibration["clabs"]
    return profile, case, raw, context, options


def measured_cfp_inputs(raw_path, encoding_path, core_wheel, case_id):
    return measured_gate_inputs(raw_path, encoding_path, core_wheel, case_id, method="cfp")


def replay_gate(owner, raw_path, encoding_path, core_wheel, case_id, *, method):
    profile, case, raw, context, options = measured_gate_inputs(
        raw_path, encoding_path, core_wheel, case_id, method=method,
    )
    owner.require_open()
    runtime = owner.runtime
    if (
        version("particlegan") != profile["particle_gan"]
        or runtime.base_identity != raw["metadata"]["model_identity"]
        or runtime.device.type != "cuda" or runtime.model.dtype != torch.bfloat16
        or type(runtime.pipe.scheduler).__name__ != profile["sampler"]
        or runtime.pipe.config.sample_sigmas is not None
    ):
        raise ValueError("Use the captured CUDA BF16 owner and Euler scheduler")
    attached = owner.native.network.state_dict()
    if attached.keys() != raw["network"].keys() or any(
        attached[name].dtype != tensor.dtype or not torch.equal(attached[name].detach().cpu(), tensor)
        for name, tensor in raw["network"].items()
    ):
        raise ValueError("The open owner must have the verified RAW192 adapter")
    context = {key: value.to(runtime.device) if value is not None else None
               for key, value in context.items()}
    latent, receipt = adaptive.render(owner, context=context, **options)
    expected_usage = {"fired": case["fired"], "eligible": case["eligible"]}
    gate = receipt["gate"]
    if (
        receipt["sigmas"] != profile["sigmas"]
        or receipt["effective_strengths"] != case["schedule"]
        or gate["usage"] != expected_usage
        or gate["method"] != method
        or gate["parameter"] != options["parameter"]
        or gate["seed"] != options["gate_seed"]
        or gate["clabs"] != options.get("clabs")
        or receipt["sigma_integrated_dose"] != case["sigma_integrated_dose"]
    ):
        raise RuntimeError(f"The render did not reproduce the measured {method.upper()} receipt")
    sigma = receipt["sigmas"]
    strength = receipt["sigma_integrated_dose"] / fsum(
        start - end for start, end in zip(sigma[:-1], sigma[1:], strict=True)
    )
    expected = case["equal_dose_constant_strength"]
    if abs(strength - expected) > ulp(expected):
        raise RuntimeError("The dose-matched control differs from the measured case")
    return latent, receipt, strength


def replay_cfp(owner, raw_path, encoding_path, core_wheel, case_id):
    return replay_gate(owner, raw_path, encoding_path, core_wheel, case_id, method="cfp")
```

To derive the API arguments without loading the native model, call
`measured_cfp_inputs(raw_path, encoding_path, core_wheel, "dev1-seed203")`.
A native constant pass uses the recorded FP32 coefficient for that case, the
same context, seed and sigma grid, and one unit gain per interval.
[Native control receipts](warm-amber-raw192-nearest-fp32-controls.json)

## Replay the measured CLABS plan

Use `replay_gate(owner, raw_path, encoding_path, core_wheel, case_id, method="clabs")`
with the same archived owner and per-case inputs. To inspect the exact API
arguments on CPU, use `measured_gate_inputs(raw_path, encoding_path, core_wheel, case_id, method="clabs")`.
The study supplies the calibrated bias, CLABS reference mean and standard
deviation, the paper's feedback settings, and the original independent seed
for each case. Keep the seed as an integer throughout. The helper returns
the realized-strength receipt and the pre-rounding constant strength.

The active calibration reached 41 of 80 firings. CLABS fired 138 times over
160 held-out intervals and improved both teapot cases over their nearest FP32
constants. The bicycle cases tied at full strength and had exact integrated
scalar matches. Seed 203 used its nearest FP32 coefficient with a difference
of −1.57 × 10⁻⁸. Seed 207 used the same gate schedule and verified closest
FP32 constant as CFP. Mean
intended effect increased by +0.003554 over OFF and +0.000761 over the
nearest FP32 constants.
CPU replay reproduced the saved probabilities and firing decisions from all
560 calibration intervals and 160 held-out intervals. The original four
matched constant strengths were recovered from the held-out sigma grid.
[Original per-case evidence](warm-amber-raw192-adaptive-study.json) · [Native control receipts](warm-amber-raw192-nearest-fp32-controls.json)
