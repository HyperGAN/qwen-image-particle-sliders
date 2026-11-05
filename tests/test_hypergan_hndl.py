"""Run separately with QWEN_TEST_CORE012=1 and the exact modern shared cohort."""

import os
from copy import deepcopy

import pytest
import torch

if os.environ.get("QWEN_TEST_CORE012") != "1":
    pytest.skip("separate core 0.12 / HyperGAN / HNDL environment required", allow_module_level=True)

from qwen_image_sliders.hypergan_hndl import PROFILE, build_game
from qwen_image_sliders.projection import predict
from qwen_image_sliders.teachers import load_prompts, paired_records
from qwen_image_sliders.tiny import TinyRuntime


def game_fixture(preset="bispf_optimized", checkpointing=False):
    runtime = TinyRuntime()
    captured = paired_records(
        runtime, load_prompts(), seed=7, height=32, width=32, trajectory_steps=4, records_per_pair=2
    )
    if checkpointing:
        runtime.model.enable_gradient_checkpointing()
    game = build_game(runtime, captured, preset=preset)
    return runtime, captured, game


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_real_shared_graphs_own_gradients_and_exact_resume(preset, checkpointing):
    runtime, captured, game = game_fixture(preset, checkpointing)
    bank = game.backend.adapter
    original = {key: value.clone() for key, value in runtime.model.state_dict().items()}
    vanilla = predict(runtime.model, captured[0]["kwargs"])
    initial = {key: value.clone() for key, value in bank.state_dict().items()}
    try:
        assert game.response_shape == (1, 8, 1, 2, 2)
        assert game.backend.model_identity["qwen_hndl_profile"] == PROFILE
        assert game.backend.model_identity["shared_builder"]["hndl_version"] == "0.10.0"
        assert game.state_dict()["config"]["pg_version"] == "0.9.5"
        owned = [parameter for group in game.opt_g.param_groups for parameter in group["params"]]
        assert len(owned) == len({id(parameter) for parameter in owned})
        assert {id(parameter) for parameter in owned} == {
            id(parameter) for parameter in game.network.parameters()
        }
        assert sum(parameter is game.backend.particles for parameter in owned) == 1
        assert all(parameter is not game.backend.particles for parameter in bank.parameters())
        game.step()
        assert any(not torch.equal(value, initial[key]) for key, value in bank.state_dict().items())
        assert all(layer.lora_up.weight.grad.abs().sum() > 0 for layer in bank.layers)
        assert game.backend.particles.grad is not None and game.backend.particles.grad.abs().sum() > 0
        # Isolate the native response path from the shared particle prior: a
        # nonzero particle gradient here must traverse the actual HNDL route.
        named = list(game.network.named_parameters())
        generated = game.backend.generated(game.rows[1])
        gradients = torch.autograd.grad(
            generated.square().mean(), [parameter for _, parameter in named], allow_unused=True
        )
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        particle_gradient = next(
            gradient
            for (_, parameter), gradient in zip(named, gradients)
            if parameter is game.backend.particles
        )
        assert particle_gradient.abs().sum() > 0
        assert all(parameter.grad is None for parameter in runtime.model.parameters())
        assert all(torch.equal(value, original[key]) for key, value in runtime.model.state_dict().items())
        with game.network.scaled(0):
            assert torch.equal(predict(runtime.model, captured[0]["kwargs"]), vanilla)
        with pytest.raises(RuntimeError), bank.active(game.backend.particles):
            raise RuntimeError("restore active state")
        assert bank.strength == 0 and bank._particles is None
        packet = game.state_dict()
        expected = game.step()
        expected_state = game.state_dict()
        game.load_state_dict(packet)
        assert_exact(game.step(), expected)
        assert_exact(game.state_dict(), expected_state)
        before = game.state_dict()
        for key, value in (("pg_version", "0.9.1"), ("pg_source_profile", "unqualified")):
            bad = deepcopy(packet)
            bad["config"][key] = value
            with pytest.raises(ValueError, match="original pinned|source"):
                game.load_state_dict(bad)
            assert_exact(game.state_dict(), before)
    finally:
        bank.close()
        bank.close()
    assert not any(
        "forward" in vars(module)
        for name, module in runtime.model.named_modules()
        if name.endswith("attn.to_out.0")
    )
    assert torch.equal(predict(runtime.model, captured[0]["kwargs"]), vanilla)
    assert not torch.cuda.is_initialized()


def test_failed_game_construction_restores_borrowed_projections(monkeypatch):
    from particle_sliders import pg09

    runtime = TinyRuntime()
    captured = paired_records(runtime, load_prompts(), seed=7, height=32, width=32, trajectory_steps=4)
    before = {name: module.forward for name, module in runtime.model.named_modules()}

    def fail(*args, **kwargs):
        raise RuntimeError("injected shared game refusal")

    monkeypatch.setattr(pg09, "ConditionalPG09", fail)
    with pytest.raises(RuntimeError, match="injected shared game refusal"):
        build_game(runtime, captured)
    assert all(module.forward == before[name] for name, module in runtime.model.named_modules())


def test_unqualified_core_sources_reject_before_native_access(monkeypatch):
    from qwen_image_sliders import hypergan_hndl

    monkeypatch.setattr(hypergan_hndl, "file_digest", lambda path: "unqualified")
    with pytest.raises(ValueError, match="shared source differs"):
        build_game(None, [])
