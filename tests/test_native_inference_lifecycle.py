"""Native v2 adoption cases; not executed until the parent admits a CPU slot."""

import pytest
import torch
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import UnsupportedOperation
from test_lifecycle_ownership import config

from qwen_image_sliders.inference import INFERENCE_BACKEND
from qwen_image_sliders.lifecycle import BACKEND
from qwen_image_sliders.tiny import TinyRuntime


def test_native_export_load_without_any_training_construction(monkeypatch):
    from qwen_image_sliders import teachers, train

    with BACKEND.open(config(ema_decay=0.9)) as training:
        training.train(stop_after=1)
        packet = training.export_inference(selection="ema")
        assert "rows" not in packet["specification"]["configuration"]
        assert "game_signature" not in packet["specification"]["identity"]
        assert packet["specification"]["response"] is False

    def forbidden(*args, **kwargs):
        pytest.fail("native-only inference reconstructed training")

    monkeypatch.setattr(teachers, "paired_records", forbidden)
    monkeypatch.setattr(train, "build_game", forbidden)
    monkeypatch.setattr(train.ConditionalPG09, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    before = torch.get_rng_state().clone()
    runtime = TinyRuntime()
    with torch.no_grad():
        vanilla = runtime.sample(
            runtime.encode("a red cup"), seed=11, height=32, width=32, steps=4, output_type="latent"
        )
    # TinyRuntime construction above is outside the inference RNG contract.
    torch.set_rng_state(before)
    with open_inference(packet, backend=INFERENCE_BACKEND) as inference:
        assert not any(hasattr(inference, name) for name in ("game", "rows", "critic", "optimizer"))
        zero = inference.render(prompt="a red cup", seed=11, strength=0.0)
        assert torch.equal(zero, vanilla)
        assert torch.isfinite(inference.render(prompt="a new blue bowl", strength=0.5)).all()
        with pytest.raises(UnsupportedOperation):
            inference.response()
        with pytest.raises(UnsupportedOperation):
            inference.train()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_native_observation_restores_backbone_buffers_and_rope(monkeypatch):
    runtime = TinyRuntime()
    original = torch.tensor([3.0])
    runtime.model.register_buffer("observation_probe", original, persistent=False)
    frequencies = runtime.model.pos_embed.freqs
    with BACKEND.open(config(), runtime=runtime) as session:

        def fail(*args, **kwargs):
            runtime.model.observation_probe.add_(2)
            runtime.model.observation_probe = torch.tensor([9.0])
            runtime.model.pos_embed.freqs = []
            raise RuntimeError("native failure")

        monkeypatch.setattr(runtime, "sample", fail)
        with pytest.raises(RuntimeError, match="native failure"):
            session.render(prompt="a red cup")
        assert runtime.model.observation_probe is original
        assert original.item() == 3.0
        assert "observation_probe" in runtime.model._non_persistent_buffers_set
        assert runtime.model.pos_embed.freqs is frequencies


def test_denied_update_admission_leaves_training_usable():
    with BACKEND.open(config(), admission=lambda context: False) as session:
        before = session.checkpoint()["sha256"]
        with pytest.raises(UnsupportedOperation, match="admission"):
            session.step()
        assert not session.closed
        assert session.checkpoint()["sha256"] == before
