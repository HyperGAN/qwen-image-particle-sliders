"""Opt-in adaptive strength for the scoped Qwen Image 2.1 native RAW renderer."""

from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
from math import fsum, isfinite
from types import ModuleType
from zipfile import ZipFile

import torch

from .scheduled_inference import _render, _validate

CORE_REVISION = "b1f780decc516808cfd3019d5e5b674457161f86"
CORE_MEMBER = "particle_sliders/adaptive_steering.py"
CORE_MEMBER_BYTES = 10329
CORE_MEMBER_SHA256 = "5fc2a5256a935eca9cecc365d9ac62e5e94cdc375d2ab33c5dc26eb344241814"


@lru_cache(maxsize=1)
def _shared_module(source):
    if sha256(source).hexdigest() != CORE_MEMBER_SHA256:
        raise ValueError("Qwen adaptive strength requires the exact reviewed core #269 source")
    # A separate module preserves the installed core and the exact RAW owner.
    module = ModuleType("qwen_image_gate_core")
    exec(compile(source, CORE_MEMBER, "exec"), module.__dict__)  # noqa: S102 - exact source verified above
    return module


def gate_core(core_wheel):
    """Load only the exact reviewed #269 equations from a local core wheel."""
    with ZipFile(core_wheel) as wheel, wheel.open(CORE_MEMBER) as member:
        source = member.read(CORE_MEMBER_BYTES + 1)
    return _shared_module(source)


def packed_latent_readout(hidden_states):
    """FP32 mean per channel over the current packed pre-edit image tokens."""
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 3
        or hidden_states.shape[0] != 1
        or not all(hidden_states.shape)
        or not hidden_states.is_floating_point()
    ):
        raise ValueError("Qwen adaptive strength requires one packed conditional latent")
    return hidden_states[0].mean(dim=0, dtype=torch.float32)


@torch.no_grad()
def render(
    owner,
    *,
    core_wheel,
    strength,
    steps,
    method,
    parameter,
    evidence_ref,
    direction=None,
    gate_seed=None,
    clabs=None,
    **request,
):
    """Gate the original routed RAW adapter once per live Euler interval.

    CFP and CLABS require a fitted unit direction and active-rollout calibration.
    Matched random uses the target firing rate and an explicit gate seed.
    """
    owner.require_open()
    _validate(strength, steps)
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        raise ValueError("Qwen adaptive strength requires an explicit fit/calibration reference")
    if type(parameter) not in (int, float) or not isfinite(parameter):
        raise ValueError("Qwen adaptive strength requires an explicit finite gate parameter")
    core = gate_core(core_wheel)
    config = core.CLABSParameters(**clabs) if clabs is not None else None
    gate = core.ActivationGate(method, parameter, seed=gate_seed, clabs=config)
    if method == "random":
        if direction is not None:
            raise ValueError("Qwen matched random control does not use an alignment direction")
    else:
        if (
            not isinstance(direction, torch.Tensor)
            or not direction.is_floating_point()
            or direction.ndim != 1
        ):
            raise ValueError("Supply a fitted Qwen packed-latent unit direction")
        direction = direction.detach().to(device=owner.runtime.device, dtype=torch.float32).clone()
        if not torch.isclose(direction.norm(), direction.new_tensor(1.0)):
            raise ValueError("Qwen packed-latent direction must have finite unit norm")

    trace = []
    scheduler = None
    pipe = owner.runtime.pipe

    def resolve(position, native):
        nonlocal scheduler
        if position == 0:
            scheduler = pipe.scheduler
        latents = native.get("hidden_states")
        readout = packed_latent_readout(latents)
        timestep = native.get("timestep")
        expected_index = position if position else None
        if (
            pipe.scheduler is not scheduler
            or scheduler.begin_index != 0
            or scheduler.step_index != expected_index
            or len(scheduler.timesteps) != steps
            or native.get("kv_cache") is not None
            or native.get("kv_cache_mode") is not None
            or not isinstance(timestep, torch.Tensor)
            or timestep.shape != (1,)
            or not torch.equal(timestep, scheduler.timesteps[position : position + 1].to(latents) / 1000)
        ):
            raise RuntimeError("Qwen adaptive steering lost single-conditional native Euler alignment")
        if direction is not None and direction.shape != readout.shape:
            raise ValueError("Fitted direction must match Qwen's packed latent channel count")
        if not strength:
            return 0.0
        score = (
            None
            if direction is None
            else core.activation_projection(readout, direction, strength=strength).item()
        )
        decision = gate.step(score)
        effective = strength if decision.fired else 0.0
        trace.append(
            {
                "interval": position,
                "timestep": timestep.item(),
                "projection": score,
                "fired": decision.fired,
                "probability": decision.probability,
                "effective_strength": effective,
            }
        )
        return effective

    output, receipt = _render(owner, 0, steps, resolve, request)
    if scheduler.step_index != steps:
        raise RuntimeError("Qwen finished without advancing each native solver interval")
    sigmas = receipt["sigmas"]
    receipt["sigma_integrated_dose"] = fsum(
        scale * (start - end)
        for scale, start, end in zip(receipt["effective_strengths"], sigmas[:-1], sigmas[1:], strict=True)
    )
    receipt["gate"] = {
        "core_revision": CORE_REVISION,
        "core_source_sha256": CORE_MEMBER_SHA256,
        "readout": "packed_image_token_mean_fp32",
        "evidence_ref": evidence_ref,
        "method": method,
        "parameter": parameter,
        "seed": gate_seed,
        "clabs": None if config is None else asdict(config),
        "direction": None if direction is None else direction.tolist(),
        "nominal_strength": strength,
        "usage": asdict(gate.usage),
        "decisions": trace,
    }
    return output, receipt
