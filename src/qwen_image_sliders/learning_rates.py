"""Allocation-free learning-rate input admission; the shared recipe owns rates."""
import math


def recipe_overrides(*, lr=None, d_lr_mult=None, prior_lr_mult=None):
    values = dict(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    result = {}
    for key, value in values.items():
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive; booleans are unsupported")
            result[key] = float(value)
    return result


def resolve_alias(*, lr=None, g_lr=None, d_lr_mult=None, prior_lr_mult=None):
    # Validate both aliases, even when a canonical value is also supplied.
    recipe_overrides(lr=g_lr)
    rates = recipe_overrides(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    if lr is not None and g_lr is not None and float(lr) != float(g_lr):
        raise ValueError("lr and g_lr aliases conflict")
    if lr is None and g_lr is not None:
        rates["lr"] = float(g_lr)
    return rates
