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


def test_native_error_depends_on_positive_teacher_and_not_feature_noise():
    runtime = TinyRuntime()
    stamp, declared = formulation()
    bank = records(runtime)
    game = build_game(runtime, bank, stamp, declared, 7)
    try:
        kwargs, target = batch_records(bank, 1, 1, "cpu")
        game.bridge.set_context(kwargs, target, game.critic.target_std, 1)
        a = game.bridge(torch.zeros(1, 8), game.particles)
        b = game.bridge(torch.randn(1, 8), game.particles)
        assert torch.equal(a, b)
        expected = (
            game.bridge.projector(
                predict(runtime.model, kwargs) - target, seed=game.bridge.projector.seed_for_step(1)
            )
            / game.critic.target_std
        )
        torch.testing.assert_close(a, expected, atol=0, rtol=0)
        game.bridge.set_context(kwargs, predict(runtime.model, kwargs).detach(), game.critic.target_std, 1)
        assert torch.count_nonzero(game.bridge(torch.zeros(1, 8), game.particles)) == 0
    finally:
        game.bridge.bank.close()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_core_updates_native_hooks_and_keeps_base_frozen(checkpointing):
    runtime = TinyRuntime()
    bank = records(runtime)
    if checkpointing:
        runtime.model.enable_gradient_checkpointing()
    base = {k: v.clone() for k, v in runtime.model.state_dict().items()}
    stamp, declared = formulation()
    game = build_game(runtime, bank, stamp, declared, 7)
    initial = {k: v.clone() for k, v in game.bridge.bank.state_dict().items()}
    try:
        for i in range(1, 5):
            stats = step_context(game, bank, i, 1)
            assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
        assert game.normalization_records == 6
        for layer in game.bridge.bank.layers:
            assert layer.up.weight.grad is not None and layer.up.weight.grad.abs().sum() > 0
        assert any(not torch.equal(v, initial[k]) for k, v in game.bridge.bank.state_dict().items())
        assert all(torch.equal(v, base[k]) for k, v in runtime.model.state_dict().items())
        assert all(p.grad is None for p in runtime.model.parameters())
        assert len(game.opt_g.param_groups[0]["params"]) == len(list(game.bridge.bank.parameters()))
    finally:
        game.bridge.bank.close()


def test_bfloat16_native_base_with_fp32_adapter_and_gradient_checkpointing():
    runtime = TinyRuntime()
    runtime.model.to(torch.bfloat16)
    bank = records(runtime)
    runtime.model.enable_gradient_checkpointing()
    stamp, declared = formulation(batch=2)
    game = build_game(runtime, bank, stamp, declared, 7)
    try:
        stats = step_context(game, bank, 1, 2)
        assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
        assert all(p.dtype == torch.float32 for p in game.bridge.bank.parameters())
        assert all(layer.up.weight.grad.abs().sum() > 0 for layer in game.bridge.bank.layers)
    finally:
        game.bridge.bank.close()


def test_controls_do_not_dilute_calibration_and_degenerate_edits_fail():
    runtime = TinyRuntime()
    bank = records(runtime)
    stamp, declared = formulation()
    game = build_game(runtime, bank, stamp, declared, 7)
    controls = [row for row in bank if row["control"]]
    more_controls = build_game(runtime, bank + controls * 20, stamp, declared, 7)
    try:
        assert torch.equal(game.critic.target_std, more_controls.critic.target_std)
        assert game.edit_rms == more_controls.edit_rms
        with pytest.raises(ValueError, match="calibration"):
            build_game(runtime, controls, stamp, declared, 7)
    finally:
        game.bridge.bank.close()
        more_controls.bridge.bank.close()
