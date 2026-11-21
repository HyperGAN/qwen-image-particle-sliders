from types import SimpleNamespace

import pytest
import torch
from particle_sliders.backends import AdapterParameters

from qwen_image_sliders.adapters import AdapterBank
from qwen_image_sliders.contracts import formulation
from qwen_image_sliders.lifecycle import _observation, _render_native
from qwen_image_sliders.scheduled_inference import render
from qwen_image_sliders.tiny import TinyRuntime


def test_native_sampler_preserves_constant_render_and_applies_gains_in_order():
    runtime = TinyRuntime()
    stamp, _ = formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.randn(128, 4)
    for layer in bank.layers:
        torch.nn.init.normal_(layer.up.weight, std=0.05)
    network = AdapterParameters(bank.named_parameters(), lambda scale: bank.active(particles, scale))
    original_scheduler = runtime.pipe.scheduler

    def original_render(**request):
        with _observation(runtime):
            return _render_native(runtime, network, {"height": 32, "width": 32}, **request)

    owner = SimpleNamespace(
        runtime=runtime,
        native=SimpleNamespace(network=network),
        require_open=lambda: None,
        render=original_render,
    )
    request = {"strength": 0.5, "steps": 4, "prompt": "Room with an open window", "seed": 7}
    try:
        before = original_render(**request)
        uniform, receipt = render(owner, **request, step_gains=[1.0] * 4)
        assert torch.equal(before, uniform)
        assert len(receipt["sigmas"]) == 5
        assert receipt["effective_strengths"] == [0.5] * 4
        timed, receipt = render(owner, **request, step_gains=[1.5, 1.5, 0.5, 0.5])
        assert receipt["effective_strengths"] == [0.75, 0.75, 0.25, 0.25]
        assert not torch.equal(before, timed)
        assert torch.equal(before, original_render(**request))
        assert runtime.pipe.scheduler is original_scheduler
        assert bank.strength == 0 and bank._particles is None
        assert not runtime.model._forward_hooks and not runtime.model._forward_pre_hooks
    finally:
        bank.close()


def test_effective_strength_out_of_range_refuses_before_opening_native_sampler():
    owner = SimpleNamespace(require_open=lambda: None)
    with pytest.raises(ValueError, match="effective strengths"):
        render(owner, strength=1.0, steps=2, step_gains=[1.5, 0.5], prompt="room", seed=7)
