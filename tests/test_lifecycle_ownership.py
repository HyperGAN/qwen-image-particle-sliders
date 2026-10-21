"""Factory ownership regressions; run only in an admitted offline CPU slot."""

from dataclasses import replace

import pytest
import torch
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation

from qwen_image_sliders.lifecycle import BACKEND
from qwen_image_sliders.tiny import TinyRuntime


def config(**options):
    return SessionConfig(
        "qwen",
        "fresh-factory-ownership-v1",
        **{
            "dummy": True,
            "steps": 4,
            "height": 32,
            "width": 32,
            "trajectory_steps": 2,
            "records_per_pair": 1,
            "rows": [
                {"neutral": "a red cup", "positive": "a shiny red cup"},
                {"neutral": "a blue bowl", "positive": "a shiny blue bowl"},
            ],
            **options,
        },
    )


@pytest.mark.parametrize(
    "options",
    [
        {"rows": [None, {}]},
        {"height": 32.0},
        {"dummy": "false"},
        {"gradient_checkpointing": 1},
        {"device": "mps"},
        {"model_dir": "/unused"},
    ],
)
def test_invalid_options_reject_before_factory(options):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid configuration reached model factory")

    with pytest.raises(ValueError):
        replace(BACKEND, prepare=forbidden).open(config(**options))


def test_precision_capability_matches_runtime_device_before_allocation():
    assert BACKEND.describe(config()).base_precisions == ("float32",)
    native = config(dummy=False, device="cuda:3")
    assert BACKEND.describe(native).base_precisions == ("bfloat16",)
    with pytest.raises(UnsupportedOperation, match="base precision"):
        BACKEND.open(config(base_precision="bfloat16"))


def test_capture_failure_restores_staged_scheduler_and_hooks(monkeypatch):
    from qwen_image_sliders import teachers

    runtime = TinyRuntime()
    scheduler = runtime.pipe.scheduler
    timesteps = scheduler.timesteps.clone()
    runtime.pipe._num_timesteps = 17

    def fail_capture(actual_runtime, *args, **kwargs):
        assert actual_runtime.pipe.scheduler is not scheduler
        actual_runtime.pipe.scheduler.timesteps.zero_()
        actual_runtime.pipe._num_timesteps = 19
        raise RuntimeError("capture failed")

    monkeypatch.setattr(teachers, "paired_records", fail_capture)
    with pytest.raises(RuntimeError, match="capture failed"):
        BACKEND.open(config(), runtime=runtime)
    assert runtime.pipe.scheduler is scheduler
    assert torch.equal(scheduler.timesteps, timesteps)
    assert runtime.pipe._num_timesteps == 17
    assert not any(module._forward_hooks for module in runtime.model.modules())


def test_interrupt_after_attachment_cleans_hooks_and_checkpoint_settings(monkeypatch):
    from qwen_image_sliders import train

    runtime = TinyRuntime()
    scheduler = runtime.pipe.scheduler
    before = [(module, getattr(module, "gradient_checkpointing", None)) for module in runtime.model.modules()]

    def interrupt(*args, **kwargs):
        assert any(module._forward_hooks for module in runtime.model.modules())
        raise KeyboardInterrupt("interrupted during game setup")

    monkeypatch.setattr(train, "ConditionalPG09", interrupt)
    with pytest.raises(KeyboardInterrupt):
        BACKEND.open(config(gradient_checkpointing=True), runtime=runtime)
    assert runtime.pipe.scheduler is scheduler
    assert not any(module._forward_hooks for module in runtime.model.modules())
    assert all(getattr(module, "gradient_checkpointing", None) == old for module, old in before)


def test_render_failure_and_close_restore_owned_state(monkeypatch):
    runtime = TinyRuntime()
    scheduler = runtime.pipe.scheduler
    timesteps = scheduler.timesteps.clone()
    runtime.pipe._num_timesteps = 17
    before = [(module, getattr(module, "gradient_checkpointing", None)) for module in runtime.model.modules()]
    with BACKEND.open(config(gradient_checkpointing=True), runtime=runtime) as session:
        assert runtime.pipe.scheduler is scheduler
        assert torch.equal(scheduler.timesteps, timesteps)
        assert runtime.pipe._num_timesteps == 17
        assert runtime.model.is_gradient_checkpointing
        packet = session.checkpoint()
        output = session.render(prompt="a red cup", output_type="latent")
        assert torch.isfinite(output).all()
        assert session.checkpoint()["sha256"] == packet["sha256"]
        assert runtime.pipe._num_timesteps == 17

        def fail_sample(*args, **kwargs):
            assert runtime.pipe.scheduler is not scheduler
            runtime.pipe.scheduler.timesteps.zero_()
            runtime.pipe._num_timesteps = 19
            raise RuntimeError("sampler failed")

        monkeypatch.setattr(runtime, "sample", fail_sample)
        with pytest.raises(RuntimeError, match="sampler failed"):
            session.render(prompt="a red cup", output_type="latent")
        assert runtime.pipe.scheduler is scheduler
        assert torch.equal(scheduler.timesteps, timesteps)
        assert runtime.pipe._num_timesteps == 17
        assert session.checkpoint()["sha256"] == packet["sha256"]
    session.close()
    assert not any(module._forward_hooks for module in runtime.model.modules())
    assert all(getattr(module, "gradient_checkpointing", None) == old for module, old in before)
