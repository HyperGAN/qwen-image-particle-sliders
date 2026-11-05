"""Opt-in Qwen binding for the separately installed core 0.13 HNDL cohort.

The shared package owns the graphs, source admission and complete GAN update.
This product owns native Qwen projection attachment and spatial field binding.
Existing S081 loaders, recipes, locks and CLI defaults remain independent.
"""

import json
from contextlib import contextmanager
from importlib.metadata import distribution, version
from importlib.resources import files

import torch
from torch import nn

from .adapters import target_modules
from .contracts import file_digest, validate_strength

PROFILE = "qwen-image21-hypergan-hndl-pg095-v1"


def _shared():
    from .hypergan_hndl_runtime import (
        check_core_inventory,
        isolated_runtime,
        loaded_dependency_origins,
        module_origin,
    )

    product_sources = isolated_runtime()
    if version("particle-sliders-core") != "0.13.0":
        raise RuntimeError(
            "Qwen HNDL requires a separate qualified core 0.13.0 environment; keep S081 intact"
        )
    profile = json.loads(files(__package__).joinpath("hypergan-hndl.lock.json").read_text())
    if profile["profile"] != PROFILE or version("particlegan") != profile["particlegan_version"]:
        raise ValueError("Qwen HNDL shared profile differs from the qualified candidate")
    core = distribution("particle-sliders-core")
    check_core_inventory(profile)
    for relative, expected in profile["core_source_sha256"].items():
        if relative.endswith(".py"):
            name = relative[:-3].replace("/", ".").removesuffix(".__init__")
            module_origin("particle-sliders-core", name, relative)
        if file_digest(core.locate_file(relative)) != expected:
            raise ValueError("Qwen HNDL shared source differs: " + relative)
    from particle_sliders.fibo_hndl import HNDLFieldCritic, HNDLParticleProjection, verified_builder

    builder, identity = verified_builder()
    loaded_dependency_origins()
    return (
        HNDLFieldCritic,
        HNDLParticleProjection,
        builder,
        {
            **identity,
            "qwen_product_sources": product_sources,
            "core_commit": profile["core_commit"],
            "core_runtime_closure_sha256": profile["core_source_closure_sha256"],
        },
    )


class QwenHNDLBank(nn.Module):
    """Borrow Qwen projections; unwind owned setup and retain failed restores for retry.

    Third-party side effects outside these captured callbacks are not rolled back.
    """

    particle_shape = (128, 4)

    def __init__(self, model, particles):
        super().__init__()
        from particle_sliders.lifecycle import native_cleanup

        _, projection, builder, identity = _shared()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Qwen HNDL requires an already frozen native backbone")
        if not isinstance(particles, nn.Parameter) or tuple(particles.shape) != self.particle_shape:
            raise ValueError("Qwen HNDL requires the actual owned 128x4 particle parameter")
        targets = target_modules(model)
        self.names = [name for name, _ in targets]
        self.layers = nn.ModuleList()
        self._restore = []
        self._native_cleanup = native_cleanup(model)
        self.strength = 0.0
        object.__setattr__(self, "_particles", None)
        try:
            for name, base in targets:
                # Build this branch before attaching it. Cleanup attempts every
                # earlier owned attachment and retains failed restores for retry.
                layer = projection(
                    name, base, particles, 8, 8, builder=builder, identity=identity, attach=False
                )
                self.layers.append(layer)
                self._restore.append((base, "forward" in vars(base), vars(base).get("forward")))
                base.forward = layer.forward
        except BaseException as primary:
            self.close(primary=primary)
            raise

    @contextmanager
    def active(self, particles, strength=1.0):
        validate_strength(strength)
        if strength and (
            not isinstance(particles, torch.Tensor)
            or tuple(particles.shape) != self.particle_shape
            or particles.device != next(self.parameters()).device
        ):
            raise ValueError("Qwen HNDL particle shape/device differs")
        previous = self._particles, self.strength
        layer_state = [(layer._particles, layer.multiplier) for layer in self.layers]
        object.__setattr__(self, "_particles", particles)
        self.strength = float(strength)
        try:
            for layer in self.layers:
                object.__setattr__(layer, "_particles", particles)
                layer.multiplier = self.strength
            yield self
        finally:
            for layer, (bank, multiplier) in zip(self.layers, layer_state):
                object.__setattr__(layer, "_particles", bank)
                layer.multiplier = multiplier
            object.__setattr__(self, "_particles", previous[0])
            self.strength = previous[1]

    def _restore_projection(self, entry):
        base, owned, forward = entry
        if owned:
            base.forward = forward
        else:
            vars(base).pop("forward", None)
        self._restore.remove(entry)

    def close(self, *, primary=None):
        from particle_sliders.lifecycle import _restore_all

        _restore_all(
            [
                *(
                    lambda entry=entry: self._restore_projection(entry)
                    for entry in reversed(tuple(self._restore))
                ),
                self._native_cleanup,
            ],
            primary=primary,
        )


class QwenHNDLCritic(nn.Module):
    """Remove only the singleton frame axis inside the shared spatial critic."""

    score_bound = 0

    def __init__(self, shape, condition_width):
        super().__init__()
        field_critic, _, _, _ = _shared()
        if len(shape) != 5 or shape[0] != 1 or shape[2] != 1:
            raise ValueError("Qwen HNDL requires native [1,C,1,H,W] responses")
        self.field = field_critic(shape[1], condition_width, shape[-2:])

    def forward(self, response, condition):
        if response.ndim != 5 or response.shape[2] != 1:
            raise ValueError("Qwen HNDL critic requires a singleton native frame axis")
        return self.field(response.squeeze(2), condition)


def build_game(
    runtime,
    records,
    *,
    seed=7,
    steps=4,
    batch=2,
    preset="bispf_optimized",
    g_lr=None,
    lr=None,
    d_lr_mult=None,
    prior_lr_mult=None,
):
    """Construct a fresh, explicitly distinct game; caller closes backend.adapter.

    No response reshaping is performed before shared domain admission. This is
    a bounded API integration, not pretrained/GPU or convergence qualification.
    """
    from particle_sliders.pg09 import ConditionalPG09
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter, image_condition_features

    from .learning_rates import resolve_alias
    from .train import response_rows

    rates = resolve_alias(lr=lr, g_lr=g_lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    _, _, _, identity = _shared()
    if type(batch) is not int or not 2 <= batch <= 512:
        raise ValueError("batch must be an integer in [2,512]")
    edits = []
    for record in records:
        delta = record["target"] - record["neutral"]
        if not torch.isfinite(delta).all():
            raise ValueError("non-finite paired teacher velocity")
        rms = float(delta.double().square().mean().sqrt())
        if not record["control"] and rms > 1e-8:
            edits.append(rms)
    if len(edits) < 2:
        raise ValueError("calibration requires at least two nondegenerate noncontrol edits")
    torch.manual_seed(seed)
    particles = nn.Parameter(torch.randn(128, 4, device=runtime.device) * 0.02)
    bank = QwenHNDLBank(runtime.model, particles)
    try:
        bank.to(runtime.device)
        adapter = QwenImage21ResponseAdapter(
            runtime,
            bank,
            particles,
            model_identity={
                **runtime.base_identity,
                "qwen_hndl_profile": PROFILE,
                "shared_builder": identity,
            },
        )
        prepared = [adapter.prepare(row) for row in response_rows(runtime, records)]
        conditions = image_condition_features(prepared)
        critic = QwenHNDLCritic(tuple(prepared[0].target.shape), conditions.shape[1]).to(runtime.device)
        game = ConditionalPG09(
            adapter,
            prepared,
            critic,
            conditions=conditions,
            particles=particles,
            response_domain="continuous_field",
            response_scale=float(torch.tensor(edits, dtype=torch.float64).median()),
            preset=preset,
            total_steps=steps,
            batch_size=batch,
            seed=seed,
            recipe_overrides=rates or None,
            sampling="balanced",
        )
        game.normalization_records = len(edits)
        return game
    except BaseException as primary:
        bank.close(primary=primary)
        raise
