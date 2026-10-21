from math import prod

import pytest
import torch
from safetensors.torch import load_file, save_file

from qwen_image_sliders.adapters import AdapterBank, target_modules
from qwen_image_sliders.checkpoint import load_adapter, metadata, save_adapter
from qwen_image_sliders.contracts import canonical, formulation
from qwen_image_sliders.infer import infer
from qwen_image_sliders.tiny import TinyRuntime


@pytest.fixture
def ema_export(tmp_path):
    runtime = TinyRuntime()
    stamp, declared = formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.linspace(-0.7, 0.6, steps=prod(bank.particle_shape)).reshape(bank.particle_shape)
    try:
        with torch.no_grad():
            for layer in bank.layers:
                layer.up.weight.copy_(
                    torch.linspace(-0.05, 0.05, layer.up.weight.numel()).reshape_as(layer.up.weight)
                )
        ema = {key: value.clone() for key, value in bank.state_dict().items()}
        ema_particles = particles.clone()
        meta = {**metadata(runtime, bank, stamp, declared), "training": {"export": "EMA"}}
        # A raw in-memory bank differs from the EMA payload selected for export.
        with torch.no_grad():
            for parameter in bank.parameters():
                parameter.add_(0.25)
            particles.add_(0.5)
        path = tmp_path / "adapter.safetensors"
        save_adapter(path, ema, ema_particles, meta)
        return path, ema, ema_particles, meta
    finally:
        bank.close()


@pytest.mark.parametrize("invalid", ["shape", "dtype", "nonfinite"])
def test_invalid_late_particle_tensor_leaves_entire_live_bank_unchanged(ema_export, tmp_path, invalid):
    path, _, _, meta = ema_export
    state = load_file(str(path))
    if invalid == "shape":
        state["particles"] = state["particles"][:1]
    elif invalid == "dtype":
        state["particles"] = state["particles"].double()
    else:
        state["particles"][0, 0] = float("nan")
    malformed = tmp_path / f"invalid-{invalid}.safetensors"
    save_file(state, str(malformed), metadata={"qwen_image_sliders": canonical(meta)})

    runtime = TinyRuntime()
    stamp, declared = formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.full(bank.particle_shape, -0.375)
    try:
        with torch.no_grad():
            for parameter in bank.parameters():
                parameter.fill_(-0.125)
        before = {key: value.clone() for key, value in bank.state_dict().items()}
        particles_before = particles.clone()
        with pytest.raises(ValueError, match="invalid adapter tensor particles"):
            load_adapter(malformed, bank, particles, metadata(runtime, bank, stamp, declared))
        assert all(torch.equal(value, before[key]) for key, value in bank.state_dict().items())
        assert torch.equal(particles, particles_before)
        assert bank.strength == 0 and bank._particles is None
    finally:
        bank.close()
    assert not any(module._forward_hooks for module in runtime.model.modules())


def test_loaded_ema_particles_and_positive_strength_control_native_projection(ema_export):
    path, ema, ema_particles, _ = ema_export
    runtime = TinyRuntime()
    stamp, declared = formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.nn.Parameter(torch.zeros(bank.particle_shape))
    try:
        identity = id(particles)
        load_adapter(path, bank, particles, metadata(runtime, bank, stamp, declared))
        assert id(particles) == identity and torch.equal(particles, ema_particles)
        assert all(torch.equal(value, ema[key]) for key, value in bank.state_dict().items())
        assert all(parameter is not particles for parameter in bank.parameters())
        projection = target_modules(runtime.model)[0][1]
        x = torch.linspace(-1, 1, projection.in_features * 3).reshape(1, 3, -1)
        with torch.no_grad():
            vanilla = projection(x)
            residual = bank.layers[0](x, particles)
            assert torch.count_nonzero(residual)
            with bank.active(particles, 1):
                torch.testing.assert_close(projection(x), vanilla + residual, rtol=0, atol=0)
            with bank.active(particles, 0.375):
                torch.testing.assert_close(projection(x), vanilla + 0.375 * residual, rtol=0, atol=0)
            with bank.active(particles, 0):
                assert torch.equal(projection(x), vanilla)
    finally:
        bank.close()


def test_native_inference_uses_exported_particles_and_restores_hooks(ema_export, tmp_path):
    path, ema, ema_particles, meta = ema_export
    alternate = tmp_path / "alternate-particles.safetensors"
    save_adapter(alternate, ema, torch.zeros_like(ema_particles), meta)
    runtime = TinyRuntime()
    options = {"seed": 42, "height": 32, "width": 32, "steps": 2, "output_type": "latent"}
    prompt = "A ceramic cup."
    vanilla = runtime.sample(runtime.encode(prompt), **options)
    zero = infer(runtime, path, prompt, strength=0, **options)
    active = infer(runtime, path, prompt, strength=0.375, **options)
    changed_particles = infer(runtime, alternate, prompt, strength=0.375, **options)
    repeated = infer(runtime, path, prompt, strength=0.375, **options)
    assert torch.equal(zero, vanilla)
    assert torch.isfinite(active).all() and not torch.equal(active, vanilla)
    assert not torch.equal(active, changed_particles)
    assert torch.equal(active, repeated)
    assert not any(module._forward_hooks for module in runtime.model.modules())
