"""Shared session/EMA/export plumbing for the separate modern Qwen cohort."""

from particle_sliders.lifecycle import Backend, Binding

from . import lifecycle as legacy
from .hypergan_hndl import _shared, build_game
from .hypergan_hndl_inference import NAME, specification


def validate(options):
    _shared()
    legacy.validate(options)
    if "critic_kind" in options:
        raise ValueError("the modern Qwen profile has a fixed shared HNDL field critic")


def prepare(options, *, runtime=None, admission=None):
    from .teachers import paired_records

    validate(options)
    if runtime is None:
        if options.get("dummy"):
            from .tiny import TinyRuntime

            runtime = TinyRuntime(options.get("seed", 7))
        else:
            from .runtime import Runtime

            runtime = Runtime(
                allow_hub=False,
                **{
                    key: value
                    for key, value in options.items()
                    if key in {"model_dir", "cache_dir", "device"}
                },
            )
    legacy._validate_runtime(runtime, options)
    restore = legacy._runtime_cleanup(runtime)
    frequencies = runtime.model.pos_embed.freqs
    game = None
    try:
        with legacy._observation(runtime):
            records = paired_records(
                runtime,
                options["rows"],
                seed=options.get("seed", 7),
                **{
                    key: options.get(key, default)
                    for key, default in (
                        ("height", 32),
                        ("width", 32),
                        ("trajectory_steps", 4),
                        ("records_per_pair", 2),
                    )
                },
            )
        if options.get("gradient_checkpointing", False):
            runtime.model.enable_gradient_checkpointing()
        game = build_game(
            runtime,
            records,
            **{
                key: value
                for key, value in options.items()
                if key in {"seed", "steps", "batch", "preset", "g_lr"}
            },
        )
        native_spec = specification(game.backend, runtime, dummy=options.get("dummy", False))
    except BaseException:
        try:
            if game is not None:
                game.backend.adapter.close()
        finally:
            restore()
        raise
    finally:
        runtime.model.pos_embed.freqs = frequencies

    def close():
        try:
            game.backend.adapter.close()
        finally:
            restore()

    def render(*, prompt, strength, seed, images=None, **native):
        defaults = {
            "height": options.get("height", 32),
            "width": options.get("width", 32),
            "steps": options.get("trajectory_steps", 4),
            "output_type": "latent" if options.get("dummy") else "pil",
        }
        return legacy._render_native(
            runtime,
            game.network,
            defaults,
            prompt=prompt,
            strength=strength,
            seed=seed,
            images=images,
            **native,
        )

    def before_update(context):
        legacy._validate_runtime(runtime, options)
        return admission(context) if admission is not None else True

    return Binding(
        game,
        close,
        render,
        inference=native_spec,
        before_update=before_update,
        observation=lambda: legacy._observation(runtime),
    )


BACKEND = Backend(NAME, legacy.capabilities({}), prepare, validate, legacy.capabilities, legacy.preflight)
