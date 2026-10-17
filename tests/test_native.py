import copy

import pytest
import torch

from qwen_image_sliders.adapters import AdapterBank, target_modules
from qwen_image_sliders.contracts import formulation
from qwen_image_sliders.projection import predict
from qwen_image_sliders.teachers import batch_records, load_prompts, paired_records, with_encoding
from qwen_image_sliders.tiny import TinyRuntime
from qwen_image_sliders.train import build_game, step_context


def records(runtime):
    return paired_records(
        runtime, load_prompts(), seed=7, height=32, width=32, trajectory_steps=4, records_per_pair=2
    )


def test_native_projection_contract_and_exact_zero_bypass():
    runtime = TinyRuntime()
    stamp, _ = formulation()
    assert [n for n, _ in target_modules(runtime.model)] == [
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.1.attn.to_out.0",
    ]
    kwargs, _ = batch_records(records(runtime), 1, 1, "cpu")
    base = predict(runtime.model, kwargs)
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.randn(128, 4)
    try:
        with bank.active(particles):
            assert torch.equal(predict(runtime.model, kwargs), base)
        for layer in bank.layers:
            torch.nn.init.normal_(layer.up.weight, std=0.05)
        with bank.active(particles, 0):
            assert torch.equal(predict(runtime.model, kwargs), base)
        with bank.active(particles):
            assert not torch.equal(predict(runtime.model, kwargs), base)
        assert "_particles" not in dict(bank.named_parameters())
        with pytest.raises(RuntimeError), bank.active(particles):
            raise RuntimeError("restore context on exceptions")
        assert bank.strength == 0 and bank._particles is None
    finally:
        bank.close()
    assert torch.equal(predict(runtime.model, kwargs), base)
    assert not any(p.requires_grad for p in runtime.model.parameters())


def test_reject_other_backbones_and_invalid_strengths():
    with pytest.raises(TypeError, match="2DModel"):
        target_modules(torch.nn.Linear(8, 8))
    stamp, _ = formulation()
    bank = AdapterBank(TinyRuntime().model, stamp)
    try:
        for value in [-1, 1.1, float("nan"), float("inf")]:
            with pytest.raises(ValueError), bank.active(torch.randn(128, 4), value):
                pass
        with pytest.raises(ValueError, match="shape"), bank.active(torch.randn(1, 4)):
            pass
    finally:
        bank.close()


def test_teachers_are_matched_native_trajectory_velocities():
    runtime = TinyRuntime()
    bank = records(runtime)
    assert len(bank) == 10
    for row in bank:
        torch.testing.assert_close(predict(runtime.model, row["kwargs"]), row["neutral"], rtol=0, atol=0)
        assert 0 < row["kwargs"]["timestep"].item() <= 1
        assert row["kwargs"]["hidden_states"].shape == (1, 4, 8)
        if row["control"]:
            assert torch.equal(row["neutral"], row["target"])
        else:
            assert not torch.equal(row["neutral"], row["target"])
    positive = with_encoding(bank[0]["kwargs"], runtime.encode(load_prompts()[0]["positive"]))
    assert torch.equal(predict(runtime.model, positive), bank[0]["target"])


def test_variable_length_native_microbatches_preserve_individual_forward():
    runtime = TinyRuntime()
    bank = records(runtime)[:2]
    short = copy.deepcopy(bank[0])
    short["kwargs"]["encoder_hidden_states"] = short["kwargs"]["encoder_hidden_states"][:, :3]
    short["kwargs"]["encoder_hidden_states_mask"] = torch.ones(1, 3, dtype=torch.bool)
    short["kwargs"]["img_mask"] = torch.tensor([[False, False, False, True]])
    kwargs, _ = batch_records([short, bank[1]], 1, 2, "cpu")
    assert [row["encoder_hidden_states"].shape[1] for row in kwargs] == [3, 5]
    together = predict(runtime.model, kwargs)
    for i, record in enumerate([short, bank[1]]):
        torch.testing.assert_close(
            together[i : i + 1], predict(runtime.model, record["kwargs"]), atol=2e-6, rtol=2e-6
        )


def test_native_response_matches_captured_teachers_and_supports_unmasked_text():
    from diffusers import QwenImage21Pipeline
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter

    from qwen_image_sliders.train import response_rows

    runtime = TinyRuntime()
    captured = records(runtime)
    stamp, declared = formulation(batch=2)
    game = build_game(runtime, captured, stamp, declared, 7)
    try:
        assert isinstance(game.backend, QwenImage21ResponseAdapter)
        assert game.response_shape == (1, 8, 1, 2, 2)
        for source, prepared in zip(captured, game.rows):
            expected = QwenImage21Pipeline._unpack_latents(source["target"], 16, 16, 8)
            assert torch.equal(prepared.target, expected)
            assert torch.equal(game.backend.generated(prepared), prepared.neutral)
        plain = copy.deepcopy(captured[0])
        plain["kwargs"]["encoder_hidden_states_mask"] = None
        plain["neutral_condition"]["prompt_embeds_mask"] = None
        row = response_rows(runtime, [plain])[0]
        response = game.backend.velocity(row.neutral, row.latent, row.timestep, scale=0)
        native = QwenImage21Pipeline._unpack_latents(predict(runtime.model, plain["kwargs"]), 16, 16, 8)
        assert torch.equal(response, native)
    finally:
        game.backend.adapter.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rectangular_response_preserves_raw_times_and_each_prompt_mask(dtype, monkeypatch):
    from diffusers import QwenImage21Pipeline
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter

    from qwen_image_sliders.train import response_rows

    runtime = TinyRuntime()
    runtime.model.to(dtype)
    original_encode = runtime.encode

    def encode(prompt):
        encoding = original_encode(prompt)
        positive = "warm" in prompt
        length = 3 if positive else 4
        encoding["prompt_embeds"] = encoding["prompt_embeds"][:, :length]
        encoding["prompt_embeds_mask"] = torch.tensor([[True, True, False]]) if positive else None
        encoding["image_pad_mask"] = torch.zeros(1, length, dtype=torch.bool)
        return encoding

    monkeypatch.setattr(runtime, "encode", encode)
    captured = paired_records(
        runtime,
        [
            {"neutral": "A cup.", "positive": "A cup in warm light."},
            {"neutral": "A room.", "positive": "A room in warm light."},
        ],
        seed=7,
        height=64,
        width=96,
        trajectory_steps=3,
        records_per_pair=3,
    )
    stamp, _ = formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.nn.Parameter(torch.zeros(bank.particle_shape))
    try:
        backend = QwenImage21ResponseAdapter(runtime, bank, particles)
        rows = response_rows(runtime, captured)
        assert len(rows) == 6
        assert any(float(row.timestep) % 1 for row in rows)
        with torch.no_grad():
            for source, row in zip(captured, rows):
                assert row.latent.shape == (1, 8, 1, 4, 6)
                assert row.neutral["embeds"].shape[1] == 4 and row.neutral["mask"] is None
                assert row.positive["embeds"].shape[1] == 3
                assert torch.equal(row.positive["mask"], torch.tensor([[True, True, False]]))
                assert torch.equal(row.timestep, source["raw_timestep"])
                prepared = backend.prepare(row)
                for field in ("target", "neutral"):
                    expected = QwenImage21Pipeline._unpack_latents(
                        source[field], 64, 96, runtime.pipe.vae_scale_factor
                    )
                    assert torch.equal(getattr(prepared, field), expected)
    finally:
        bank.close()


def test_shared_binding_captures_checkpoint_scale_after_response_context_exits():
    def gradients(checkpointing):
        runtime = TinyRuntime()
        captured = records(runtime)
        if checkpointing:
            runtime.model.enable_gradient_checkpointing()
        stamp, declared = formulation(batch=2)
        game = build_game(runtime, captured, stamp, declared, 7)
        bank = game.backend.adapter
        try:
            with torch.no_grad():
                for layer in bank.layers:
                    layer.up.weight.copy_(
                        torch.linspace(-0.03, 0.03, layer.up.weight.numel()).reshape_as(layer.up.weight)
                    )
            # No step_context/outer active scope can mask a missing shared checkpoint binding.
            generated = game.backend.generated(game.rows[1])
            assert bank.strength == 0 and bank._particles is None
            named = list(game.network.named_parameters())
            result = torch.autograd.grad(
                generated.square().mean(), [parameter for _, parameter in named], allow_unused=True
            )
            assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in result)
            assert sum(float(gradient.abs().sum()) for gradient in result) > 0
            assert bank.strength == 0 and bank._particles is None
            return {name: gradient.detach().clone() for (name, _), gradient in zip(named, result)}
        finally:
            bank.close()

    baseline = gradients(False)
    checkpointed = gradients(True)
    assert baseline.keys() == checkpointed.keys()
    for name in baseline:
        torch.testing.assert_close(checkpointed[name], baseline[name], rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_shared_core_updates_actual_bank_and_keeps_base_frozen(checkpointing, preset):
    from particle_sliders.pg09 import ConditionalPG09

    runtime = TinyRuntime()
    captured = records(runtime)
    if checkpointing:
        runtime.model.enable_gradient_checkpointing()
    base = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    stamp, declared = formulation(batch=2)
    game = build_game(runtime, captured, stamp, declared, 7, preset=preset)
    bank = game.backend.adapter
    initial = {k: v.clone() for k, v in bank.state_dict().items()}
    particles = game.backend.particles
    seen = []
    handle = bank.layers[0].route.register_forward_pre_hook(
        lambda module, args: seen.append(args[1] is particles)
    )
    try:
        assert isinstance(game, ConditionalPG09)
        assert game.prior.z is particles
        for i in range(1, 5):
            stats = step_context(game)
            assert all(torch.isfinite(torch.tensor(v)).all() for v in stats.values())
        assert seen and all(seen)
        assert game.normalization_records == 6
        assert game.state_dict()["config"]["objective"] == "g_adv+public_prior;d_adv+public_penalty"
        assert game.state_dict()["config"]["pg_version"] == "0.9.1"
        assert all(layer.up.weight.grad.abs().sum() > 0 for layer in bank.layers)
        assert any(not torch.equal(v, initial[k]) for k, v in bank.state_dict().items())
        assert all(torch.equal(v, base[k]) for k, v in runtime.model.state_dict().items())
        assert all(p.grad is None for p in runtime.model.parameters())
        owned = [p for group in game.opt_g.param_groups for p in group["params"]]
        assert len(owned) == len(set(map(id, owned)))
        assert set(map(id, owned)) == set(map(id, game.network.parameters()))
        assert sum(p is particles for p in owned) == 1
        assert game.exposures["g"].sum() == game.exposures["d"].sum() == 8
    finally:
        handle.remove()
        bank.close()


def test_bfloat16_base_checkpointing_and_fp32_adapters():
    runtime = TinyRuntime()
    runtime.model.to(torch.bfloat16)
    captured = records(runtime)
    runtime.model.enable_gradient_checkpointing()
    stamp, declared = formulation(batch=2)
    game = build_game(runtime, captured, stamp, declared, 7)
    try:
        stats = step_context(game)
        assert all(torch.isfinite(torch.tensor(v)).all() for v in stats.values())
        assert all(p.dtype == torch.float32 for p in game.network.parameters())
        assert all(layer.up.weight.grad.abs().sum() > 0 for layer in game.backend.adapter.layers)
    finally:
        game.backend.adapter.close()


def test_controls_do_not_dilute_calibration_and_degenerate_edits_fail():
    runtime = TinyRuntime()
    captured = records(runtime)
    stamp, declared = formulation(batch=2)
    game = build_game(runtime, captured, stamp, declared, 7)
    controls = [row for row in captured if row["control"]]
    try:
        game.backend.adapter.close()
        more_controls = build_game(runtime, captured + controls * 20, stamp, declared, 7)
        try:
            assert game.response_scale == more_controls.response_scale
        finally:
            more_controls.backend.adapter.close()
        with pytest.raises(ValueError, match="calibration"):
            build_game(runtime, controls, stamp, declared, 7)
    finally:
        game.backend.adapter.close()
