"""Schedule adapter strength on the original Qwen-Image 2.1 sampler."""

from contextlib import ExitStack
from itertools import pairwise
from math import isfinite
from sys import float_info


def render(owner, *, strength, steps, step_gains, **request):
    """Run a scoped RAW request and return the native sigma grid and strengths."""
    owner.require_open()
    if type(steps) is not int or steps < 1:
        raise ValueError("Qwen requires a positive native step count")
    if type(strength) not in (int, float) or not 0 <= strength <= 1:
        raise ValueError("Qwen nominal strength must be in [0, 1]")
    if not isinstance(step_gains, (list, tuple)) or len(step_gains) != steps:
        raise ValueError("Qwen requires one step gain per native Euler step")
    if any(type(gain) not in (int, float) or not 0 <= gain <= float_info.max for gain in step_gains):
        raise ValueError("Qwen step gains must be finite and nonnegative")
    strengths = tuple(strength * gain for gain in step_gains)
    if any(not 0 <= value <= 1 for value in strengths):
        raise ValueError("Qwen effective strengths must be in [0, 1]")

    runtime = owner.runtime
    network = owner.native.network
    scope = ExitStack()
    sigmas = None
    position = 0

    def before(_model, _args):
        nonlocal sigmas, position
        if position >= steps:
            raise RuntimeError("Qwen made extra denoiser calls")
        if sigmas is None:
            sigmas = runtime.pipe.scheduler.sigmas.detach().cpu().tolist()
            if len(sigmas) != steps + 1 or any(
                not isfinite(start) or not isfinite(end) or start <= end for start, end in pairwise(sigmas)
            ):
                raise RuntimeError("Qwen native Euler grid differs from the requested step count")
        scope.enter_context(network.scaled(strengths[position]))
        position += 1

    def after(_model, _args, _output):
        scope.close()

    with ExitStack() as handles:
        handles.callback(runtime.model.register_forward_pre_hook(before).remove)
        handles.callback(runtime.model.register_forward_hook(after, always_call=True).remove)
        handles.callback(scope.close)
        output = owner.render(strength=strength, steps=steps, **request)
    if position != steps:
        raise RuntimeError("Qwen finished before applying all scheduled strengths")
    return output, {"sigmas": sigmas, "effective_strengths": list(strengths)}
