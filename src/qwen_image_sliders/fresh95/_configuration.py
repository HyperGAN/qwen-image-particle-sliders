"""Explicit new cached profile configuration; reuse original validation/rate laws."""

from copy import deepcopy

from qwen_image_sliders.contracts import formulation
from qwen_image_sliders.learning_rates import resolve_alias
from qwen_image_sliders.lifecycle import validate


def configuration(
    rows,
    *,
    steps=128,
    seed=7,
    batch=2,
    height=32,
    width=32,
    trajectory_steps=4,
    records_per_pair=2,
    gradient_checkpointing=False,
    preset="bispf_optimized",
    lr=None,
    d_lr_mult=None,
    prior_lr_mult=None,
    device="cpu",
    dummy=False,
    ema_decay=None,
):
    if ema_decay is not None:
        raise ValueError("no weight EMA; RAW only")
    if type(steps) is not int or steps < 1 or type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("positive fixed horizon and original seed range required")
    if type(batch) is not int or not 2 <= batch <= 512:
        raise ValueError("original batch in [2,512] required")
    rates = resolve_alias(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    result = dict(
        rows=deepcopy(rows),
        steps=steps,
        seed=seed,
        batch=batch,
        height=height,
        width=width,
        trajectory_steps=trajectory_steps,
        records_per_pair=records_per_pair,
        gradient_checkpointing=gradient_checkpointing,
        preset=preset,
        device=device,
        dummy=dummy,
        ema_decay=None,
        **rates,
    )
    validate(result)  # Exact original core/PG/formulation/runtime-option guards.
    formulation(batch=batch, lr=rates.get("lr"))
    return result
