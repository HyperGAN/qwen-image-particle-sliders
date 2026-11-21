import os
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from particle_sliders.backends import AdapterParameters

from qwen_image_sliders.adapters import AdapterBank
from qwen_image_sliders.adaptive_inference import (
    gate_core,
    packed_latent_readout,
)
from qwen_image_sliders.adaptive_inference import (
    render as render_adaptive,
)
from qwen_image_sliders.contracts import formulation
from qwen_image_sliders.lifecycle import _observation, _render_native
from qwen_image_sliders.scheduled_inference import render
from qwen_image_sliders.tiny import TinyRuntime


@pytest.fixture
def native():
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
    try:
        yield owner, bank, original_render, original_scheduler
    finally:
        bank.close()


REQUEST = {"strength": 0.5, "steps": 4, "prompt": "Room with an open window", "seed": 7}


def test_native_sampler_preserves_constant_render_and_applies_gains_in_order(native):
    owner, bank, original_render, original_scheduler = native
    runtime = owner.runtime
    request = REQUEST
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


def test_effective_strength_out_of_range_refuses_before_opening_native_sampler():
    owner = SimpleNamespace(require_open=lambda: None)
    with pytest.raises(ValueError, match="effective strengths"):
        render(owner, strength=1.0, steps=2, step_gains=[1.5, 0.5], prompt="room", seed=7)


@pytest.fixture
def gate_wheel():
    value = os.environ.get("QWEN_GATE_CORE_WHEEL")
    if not value:
        pytest.skip("set QWEN_GATE_CORE_WHEEL to the reviewed core #269 wheel")
    return value


def test_gate_loader_rejects_unreviewed_source(tmp_path):
    changed = tmp_path / "changed.whl"
    with ZipFile(changed, "w") as wheel:
        wheel.writestr("particle_sliders/adaptive_steering.py", "raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="reviewed core"):
        gate_core(changed)


def test_adaptive_native_cfp_resolves_live_scores_and_replays(native, gate_wheel):
    owner, bank, original, scheduler = native
    seen = []

    def observe(_, args, kwargs):
        assert not args and bank.strength == 0
        assert kwargs["kv_cache"] is None
        seen.append(packed_latent_readout(kwargs["hidden_states"]))

    hook = owner.runtime.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        off = original(**{**REQUEST, "strength": 0})
    finally:
        hook.remove()
    channel = max(
        range(seen[0].numel()), key=lambda i: float(max(h[i] for h in seen) - min(h[i] for h in seen))
    )
    direction = torch.eye(seen[0].numel())[channel]
    threshold = sorted(float(hidden @ direction) for hidden in seen)[1]
    outcome, receipt = render_adaptive(
        owner,
        **REQUEST,
        method="cfp",
        parameter=threshold,
        evidence_ref="tiny-native-readout-fixture",
        core_wheel=gate_wheel,
        direction=direction,
    )
    decisions = receipt["gate"]["decisions"]
    assert len(decisions) == REQUEST["steps"] and decisions[0]["projection"] == float(seen[0] @ direction)
    assert all(row["fired"] == (row["projection"] <= threshold) for row in decisions)
    assert any(
        row["projection"] != float(state @ direction) for row, state in zip(decisions, seen, strict=True)
    )
    assert 0 < receipt["gate"]["usage"]["fired"] < REQUEST["steps"]
    gains = [value / REQUEST["strength"] for value in receipt["effective_strengths"]]
    replay, fixed_receipt = render(owner, **REQUEST, step_gains=gains)
    assert torch.equal(outcome, replay) and not torch.equal(outcome, off)
    assert receipt["sigmas"] == fixed_receipt["sigmas"]
    assert receipt["sigma_integrated_dose"] == pytest.approx(
        sum(
            value * (start - end)
            for value, start, end in zip(
                receipt["effective_strengths"], receipt["sigmas"][:-1], receipt["sigmas"][1:], strict=True
            )
        )
    )
    assert owner.runtime.pipe.scheduler is scheduler
    assert bank.strength == 0 and bank._particles is None
    assert not owner.runtime.model._forward_hooks and not owner.runtime.model._forward_pre_hooks


def test_native_gate_off_constant_and_clabs(native, gate_wheel):
    owner, _, original, _ = native
    options = {"evidence_ref": "tiny-native-endpoint-fixture", "core_wheel": gate_wheel, "gate_seed": 44}
    off, receipt = render_adaptive(owner, **REQUEST, method="random", parameter=0.0, **options)
    assert torch.equal(off, original(**{**REQUEST, "strength": 0}))
    assert receipt["gate"]["usage"] == {"fired": 0, "eligible": REQUEST["steps"]}
    constant, receipt = render_adaptive(owner, **REQUEST, method="random", parameter=1.0, **options)
    assert torch.equal(constant, original(**REQUEST))
    assert receipt["gate"]["usage"] == {"fired": REQUEST["steps"], "eligible": REQUEST["steps"]}
    clabs = {
        "mean": 0,
        "std": 1,
        "epsilon": 1e-6,
        "setpoint": 0,
        "proportional_gain": 1,
        "integral_gain": 0.2,
        "beta": 0.25,
    }
    direction = torch.eye(owner.runtime.model.config.in_channels)[0]
    _, first = render_adaptive(
        owner, **REQUEST, method="clabs", parameter=0.5, direction=direction, clabs=clabs, **options
    )
    _, second = render_adaptive(
        owner, **REQUEST, method="clabs", parameter=0.5, direction=direction, clabs=clabs, **options
    )
    assert first == second and first["gate"]["clabs"] == clabs
    decisions = first["gate"]["decisions"]
    reference = gate_core(gate_wheel).ActivationGate(
        "clabs", 0.5, seed=44, clabs=gate_core(gate_wheel).CLABSParameters(**clabs)
    )
    for observed in decisions:
        expected = reference.step(observed["projection"])
        assert observed["fired"] == expected.fired and observed["probability"] == expected.probability


def test_native_gate_rejects_wrong_interval_and_restores_owner(native, gate_wheel):
    owner, bank, original, scheduler = native
    model = owner.runtime.model

    def wrong_timestep(_, args, kwargs):
        if owner.runtime.pipe.scheduler.step_index == 1:
            return args, {**kwargs, "timestep": kwargs["timestep"] + 1}

    original_result = original(**REQUEST)
    handle = model.register_forward_pre_hook(wrong_timestep, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="native Euler alignment"):
            render_adaptive(
                owner,
                **REQUEST,
                evidence_ref="tiny-native-clock-fixture",
                core_wheel=gate_wheel,
                method="random",
                parameter=1.0,
                gate_seed=44,
            )
        assert list(model._forward_pre_hooks) == [handle.id]
        assert not model._forward_hooks
    finally:
        handle.remove()
    assert owner.runtime.pipe.scheduler is scheduler
    assert bank.strength == 0 and bank._particles is None
    assert torch.equal(original_result, original(**REQUEST))
