"""CPU native-factory integration; no pretrained/model downloads."""

import pytest
import torch
from particle_sliders.engine import _hash
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation, backend, open_session


def test_common_factory_roundtrip(tmp_path):
    torch.set_num_threads(1)
    config = SessionConfig(
        "qwen",
        "lifecycle-cpu-new-v1",
        dummy=True,
        steps=4,
        ema_decay=None,
        height=32,
        width=32,
        trajectory_steps=4,
        records_per_pair=2,
        rows=[
            {"neutral": "a red cup", "positive": "a shiny red cup"},
            {"neutral": "a blue bowl", "positive": "a shiny blue bowl"},
        ],
    )
    before = torch.get_rng_state().clone()
    with open_session(config) as full:
        assert torch.equal(before, torch.get_rng_state())
        full.train()
        expected = full.checkpoint()
    with open_session(config) as partial:
        partial.train(stop_after=2)
        state = partial.checkpoint(tmp_path / "training.pt")
        for strength in (0.0, 0.5, 1.0):
            response = partial.evaluate(strength=strength)
            assert torch.isfinite(response).all()
            assert partial.checkpoint()["sha256"] == state["sha256"]
    with open_session(config, resume=tmp_path / "training.pt") as resumed:
        resumed.train()
        assert resumed.checkpoint()["sha256"] == expected["sha256"]
        resumed.export(tmp_path / "adapter.pt", selection="raw")
        expected_live_hash = _hash(resumed.game.network.state_dict())
    with open_session(config, artifact=tmp_path / "adapter.pt", operation="evaluate") as inference:
        assert _hash(inference.game.network.state_dict()) == expected_live_hash
        assert torch.isfinite(inference.evaluate()).all()
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


def test_preflight_unsupported_before_factory(monkeypatch):
    contract = backend("qwen")
    with pytest.raises(UnsupportedOperation):
        contract.open(SessionConfig("qwen", "new", dummy=True), references=True)
