"""Native-only v2 inference for the isolated modern Qwen HNDL profile."""

from types import SimpleNamespace

import torch
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import _restore_all

from . import inference as legacy
from .hypergan_hndl import PROFILE, QwenHNDLBank, _shared
from .hypergan_hndl_runtime import runtime_cleanup
from .lifecycle import _observation, _render_native, _validate_runtime

PROTOCOL = PROFILE + "-native-v1"
NAME = "qwen-hndl"


def model_identity(runtime, context):
    _, _, _, identity = _shared()
    return {
        **legacy._model_identity(runtime),
        "qwen_hndl_profile": PROFILE,
        "shared_builder": identity,
        "training_context": dict(context),
    }


def specification(native, runtime, *, dummy, context):
    profile = {"dummy": dummy, "base_precision": str(runtime.model.dtype).removeprefix("torch.")}
    if dummy:
        profile["fixture_seed"] = runtime.base_identity["fixture_seed"]
    profile["training_context"] = dict(context)
    return INFERENCE_BACKEND.specification(
        native, model_identity=model_identity(runtime, context), protocol=PROTOCOL, configuration=profile
    )


def _context(spec):
    from .contracts import validate_shape

    context = spec.configuration.get("training_context")
    if (
        not isinstance(context, dict)
        or set(context) != {"height", "width", "trajectory_steps", "training_steps"}
        or any(type(value) is not int or value < 1 for value in context.values())
    ):
        raise ValueError("native export requires explicit training geometry and step context")
    validate_shape(context["height"], context["width"])
    model = spec.identity.get("model")
    if not isinstance(model, dict) or context != model.get("training_context"):
        raise ValueError("native training context differs from its identity")
    return context


def _legacy_spec(spec):
    return SimpleNamespace(
        configuration={key: value for key, value in spec.configuration.items() if key != "training_context"}
    )


def validate(options, spec):
    _shared()
    _context(spec)
    legacy.validate(options, _legacy_spec(spec))


def preflight(options, request, spec):
    legacy.preflight(options, request, spec)
    context = _context(spec)
    for key, expected in (
        ("height", context["height"]),
        ("width", context["width"]),
        ("steps", context["trajectory_steps"]),
    ):
        if key in request["native"] and request["native"][key] != expected:
            raise ValueError(
                "native render " + key + " differs from exported training context; qualify a new context"
            )


def _load_runtime(local):
    if local["dummy"]:
        from .tiny import TinyRuntime

        return TinyRuntime(local["fixture_seed"])
    from .runtime import Runtime

    return Runtime(
        allow_hub=False,
        **{key: value for key, value in local.items() if key in {"model_dir", "cache_dir", "device"}},
    )


def prepare(options, spec):
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter

    _shared()  # Exact source admission before touching native model/runtime state.
    validate(options, spec)
    context = _context(spec)
    local = legacy._options(options, _legacy_spec(spec))
    if local["dummy"]:
        local["fixture_seed"] = spec.configuration["fixture_seed"]
    runtime = _load_runtime(local)
    _validate_runtime(runtime, local)
    restore = runtime_cleanup(runtime)
    bank = None
    try:
        # The shared loader validates the complete exported state before copying.
        particles = torch.nn.Parameter(torch.zeros(128, 4, device=runtime.device))
        bank = QwenHNDLBank(runtime.model, particles).to(runtime.device)
        native = QwenImage21ResponseAdapter(
            runtime, bank, particles, model_identity=model_identity(runtime, context)
        )
        identity = native_identity(native, model_identity=model_identity(runtime, context), protocol=PROTOCOL)
    except BaseException as primary:
        _restore_all([*([bank.close] if bank is not None else []), restore], primary=primary)
        raise

    def close():
        _restore_all([bank.close, restore])

    def render(*, prompt, strength, seed, images=None, **request):
        defaults = {
            "height": context["height"],
            "width": context["width"],
            "steps": context["trajectory_steps"],
            "output_type": "latent" if local["dummy"] else "pil",
        }
        return _render_native(
            runtime,
            native.network,
            defaults,
            prompt=prompt,
            strength=strength,
            seed=seed,
            images=images,
            **request,
        )

    return InferenceBinding(
        native.network, identity, close, render=render, observation=lambda: _observation(runtime)
    )


INFERENCE_BACKEND = InferenceBackend(
    NAME,
    legacy.capabilities({}),
    prepare,
    validate,
    response=False,
    configure_capabilities=legacy.capabilities,
    preflight=preflight,
)
