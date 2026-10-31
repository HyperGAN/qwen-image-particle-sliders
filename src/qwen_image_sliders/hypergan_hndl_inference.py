"""Native-only v2 inference for the isolated modern Qwen HNDL profile."""

import torch
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity

from . import inference as legacy
from .hypergan_hndl import PROFILE, QwenHNDLBank, _shared
from .lifecycle import _observation, _render_native, _runtime_cleanup, _validate_runtime

PROTOCOL = PROFILE + "-native-v1"
NAME = "qwen-hndl"


def model_identity(runtime):
    _, _, _, identity = _shared()
    return {**legacy._model_identity(runtime), "qwen_hndl_profile": PROFILE, "shared_builder": identity}


def specification(native, runtime, *, dummy):
    profile = {"dummy": dummy, "base_precision": str(runtime.model.dtype).removeprefix("torch.")}
    if dummy:
        profile["fixture_seed"] = runtime.base_identity["fixture_seed"]
    return INFERENCE_BACKEND.specification(
        native, model_identity=model_identity(runtime), protocol=PROTOCOL, configuration=profile
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
    legacy.validate(options, spec)
    local = legacy._options(options, spec)
    if local["dummy"]:
        local["fixture_seed"] = spec.configuration["fixture_seed"]
    runtime = _load_runtime(local)
    _validate_runtime(runtime, local)
    restore = _runtime_cleanup(runtime)
    bank = None
    try:
        # The shared loader validates the complete exported state before copying.
        particles = torch.nn.Parameter(torch.zeros(128, 4, device=runtime.device))
        bank = QwenHNDLBank(runtime.model, particles).to(runtime.device)
        native = QwenImage21ResponseAdapter(runtime, bank, particles, model_identity=model_identity(runtime))
        identity = native_identity(native, model_identity=model_identity(runtime), protocol=PROTOCOL)
    except BaseException:
        try:
            if bank is not None:
                bank.close()
        finally:
            restore()
        raise

    def close():
        try:
            bank.close()
        finally:
            restore()

    def render(*, prompt, strength, seed, images=None, **request):
        defaults = {
            "height": 32 if local["dummy"] else 512,
            "width": 32 if local["dummy"] else 512,
            "steps": 4 if local["dummy"] else 40,
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
    legacy.validate,
    response=False,
    configure_capabilities=legacy.capabilities,
    preflight=legacy.preflight,
)
