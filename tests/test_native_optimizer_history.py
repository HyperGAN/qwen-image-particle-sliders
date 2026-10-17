from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic

from qwen_image_sliders.contracts import formulation
from qwen_image_sliders.teachers import load_prompts, paired_records
from qwen_image_sliders.tiny import TinyRuntime
from qwen_image_sliders.train import build_game


def assert_exact(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


class CriticWithUnusedParameter(ConditionalReadoutCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unused = torch.nn.Parameter(torch.ones(2))


@contextmanager
def native_game(preset, *, unused_critic_parameter=False):
    runtime = TinyRuntime()
    captured = paired_records(
        runtime, load_prompts(), seed=7, height=32, width=32, trajectory_steps=4, records_per_pair=2
    )
    runtime.model.enable_gradient_checkpointing()
    stamp, declared = formulation(batch=2)
    critic_class = CriticWithUnusedParameter if unused_critic_parameter else ConditionalReadoutCritic
    with patch("qwen_image_sliders.train.ConditionalReadoutCritic", critic_class):
        game = build_game(runtime, captured, stamp, declared, 7, steps=4, preset=preset)
    try:
        yield game
    finally:
        game.backend.adapter.close()
    assert not any(module._forward_hooks for module in runtime.model.modules())
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module", params=["bispf", "bispf_optimized"])
def started_native_checkpoint(request):
    with native_game(request.param) as game:
        game.step()
        packet = game.state_dict()
        expected = game.step()
        return request.param, packet, expected, game.state_dict()


@pytest.mark.parametrize("optimizer", ["opt_g", "opt_d"])
@pytest.mark.parametrize(
    "corruption",
    [
        "negative_second_moment",
        "missing_field",
        "extra_field",
        "extra_parameter_history",
        "deleted_history",
        "empty_history",
    ],
)
def test_native_started_optimizer_history_rejection_is_atomic(
    started_native_checkpoint, optimizer, corruption
):
    preset, original, _, _ = started_native_checkpoint
    packet = deepcopy(original)
    history = packet[optimizer]["state"]
    key = next(
        key for group in packet[optimizer]["param_groups"] for key in group["params"] if history.get(key)
    )
    slots = history[key]
    with native_game(preset) as receiver:
        if corruption == "negative_second_moment":
            parameter = next(
                parameter
                for saved, live in zip(
                    packet[optimizer]["param_groups"], getattr(receiver, optimizer).param_groups
                )
                for saved_key, parameter in zip(saved["params"], live["params"])
                if saved_key == key
            )
            slots["exp_avg_sq"] = torch.full_like(parameter, -1)
        elif corruption == "missing_field":
            del slots["exp_avg_sq" if "exp_avg_sq" in slots else "step"]
        elif corruption == "extra_field":
            slots["unexpected_history_field"] = torch.tensor(1.0)
        elif corruption == "extra_parameter_history":
            unknown = 1 + max(key for group in packet[optimizer]["param_groups"] for key in group["params"])
            history[unknown] = deepcopy(slots)
        elif corruption == "deleted_history":
            del history[key]
        else:
            history[key] = {}
        before = receiver.state_dict()
        ambient_rng = torch.get_rng_state().clone()
        flags = [(parameter, parameter.requires_grad) for parameter in receiver.critic.parameters()]
        with pytest.raises(ValueError, match="optimizer|Adam history"):
            receiver.load_state_dict(packet)
        # Includes native adapter leaves, critic/prior, both optimizers, model RNG,
        # all independent sampling/noise streams, row orders/cursors and exposures.
        assert_exact(before, receiver.state_dict())
        assert torch.equal(ambient_rng, torch.get_rng_state())
        assert all(parameter.requires_grad == flag for parameter, flag in flags)
        assert receiver.backend.adapter.strength == 0 and receiver.backend.adapter._particles is None


@pytest.mark.parametrize("started_native_checkpoint", ["bispf"], indirect=True)
@pytest.mark.parametrize("optimizer", ["opt_g", "opt_d"])
@pytest.mark.parametrize("invalid_clock", ["zero", "ahead"])
def test_native_original_adam_clock_rejection_is_atomic(started_native_checkpoint, optimizer, invalid_clock):
    preset, original, _, _ = started_native_checkpoint
    packet = deepcopy(original)
    history = packet[optimizer]["state"]
    key = next(
        key for group in packet[optimizer]["param_groups"] for key in group["params"] if history.get(key)
    )
    clock = history[key]["step"]
    invalid_count = 0 if invalid_clock == "zero" else packet["completed_steps"] + 1
    history[key]["step"] = (
        clock.new_tensor(invalid_count) if isinstance(clock, torch.Tensor) else type(clock)(invalid_count)
    )
    with native_game(preset) as receiver:
        assert isinstance(getattr(receiver, optimizer), torch.optim.Adam)
        before = receiver.state_dict()
        ambient_rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match="Adam step count differs"):
            receiver.load_state_dict(packet)
        assert_exact(before, receiver.state_dict())
        assert torch.equal(ambient_rng, torch.get_rng_state())
        assert receiver.backend.adapter.strength == 0 and receiver.backend.adapter._particles is None


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_native_initial_empty_optimizer_histories_accept_exact_first_update(preset):
    with native_game(preset) as source, native_game(preset) as receiver:
        packet = source.state_dict()
        assert packet["opt_g"]["state"] == packet["opt_d"]["state"] == {}
        receiver.load_state_dict(packet)
        assert_exact(source.step(), receiver.step())
        assert_exact(source.state_dict(), receiver.state_dict())


def test_native_valid_started_checkpoint_preserves_exact_next_update(started_native_checkpoint):
    preset, packet, expected, expected_state = started_native_checkpoint
    snapshot = deepcopy(packet)
    with native_game(preset) as receiver:
        receiver.load_state_dict(packet)
        assert_exact(expected, receiver.step())
        assert_exact(expected_state, receiver.state_dict())
    assert_exact(snapshot, packet)


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_native_unused_critic_parameter_can_honestly_resume_without_history(preset):
    with native_game(preset, unused_critic_parameter=True) as source:
        source.step()
        packet = source.state_dict()
        unused = next(
            key
            for saved, actual in zip(packet["opt_d"]["param_groups"], source.opt_d.param_groups)
            for key, parameter in zip(saved["params"], actual["params"])
            if parameter is source.critic.unused
        )
        assert unused not in packet["opt_d"]["state"]
        assert packet["optimizer_history_schema"]["d"][unused] is None
        expected = source.step()
        expected_state = source.state_dict()
    with native_game(preset, unused_critic_parameter=True) as receiver:
        receiver.load_state_dict(packet)
        assert_exact(expected, receiver.step())
        assert_exact(expected_state, receiver.state_dict())
        assert receiver.critic.unused.grad is None
