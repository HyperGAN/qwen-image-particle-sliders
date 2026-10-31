"""Modern native-only raw/EMA lifecycle; separate exact candidate environment."""

import os
from copy import deepcopy

import pytest
import torch

if os.environ.get("QWEN_TEST_CORE012") != "1":
    pytest.skip("separate core 0.12 / HyperGAN / HNDL environment required", allow_module_level=True)

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.pg09 import _hash

from qwen_image_sliders.hypergan_hndl_inference import INFERENCE_BACKEND
from qwen_image_sliders.hypergan_hndl_lifecycle import BACKEND
from qwen_image_sliders.tiny import TinyRuntime


def config(**options):
    return SessionConfig(
        "qwen-hndl",
        "qwen-hndl-modern-cpu-v1",
        **{
            "dummy": True,
            "steps": 4,
            "ema_decay": 0.5,
            "height": 32,
            "width": 32,
            "trajectory_steps": 4,
            "records_per_pair": 2,
            "rows": [
                {"neutral": "a red cup", "positive": "a shiny red cup"},
                {"neutral": "a blue bowl", "positive": "a shiny blue bowl"},
            ],
            **options,
        },
    )


@pytest.mark.parametrize("selection", ["raw", "ema"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_only_restores_selected_values_and_exact_render_without_training(
    selection, checkpointing, monkeypatch, tmp_path
):
    from particle_sliders import pg09

    from qwen_image_sliders import hypergan_hndl, hypergan_hndl_lifecycle, teachers

    opening_rng = torch.get_rng_state().clone()
    with BACKEND.open(config(gradient_checkpointing=checkpointing)) as training:
        assert torch.equal(opening_rng, torch.get_rng_state())
        training.train(stop_after=3)
        before = training.checkpoint()["sha256"]
        raw = training.export_inference(selection="raw")
        ema = training.export_inference(selection="ema")
        assert _hash(raw["network"]) != _hash(ema["network"])
        packet = training.export_inference(tmp_path / "native.pt", selection=selection)
        assert training.checkpoint()["sha256"] == before
        assert packet["origin"]["pg_version"] == "0.9.4"
        assert "rows" not in packet["specification"]["configuration"]
        assert "game_signature" not in packet["specification"]["identity"]
        assert set(packet) == {"schema", "specification", "origin", "selection", "network", "sha256"}
        raw_values = deepcopy(training.game.network.state_dict())
        try:
            training.game.network.load_state_dict(packet["network"])
            expected = {
                strength: training.render(prompt="a green vase", seed=11, strength=strength)
                for strength in (0.0, 0.5, 1.0)
            }
        finally:
            training.game.network.load_state_dict(raw_values)
        assert training.checkpoint()["sha256"] == before
        assert not torch.equal(expected[0.0], expected[1.0])

    def forbidden(*args, **kwargs):
        pytest.fail("native-only inference reconstructed training")

    monkeypatch.setattr(pg09.ConditionalPG09, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    monkeypatch.setattr(hypergan_hndl.QwenHNDLCritic, "__init__", forbidden)
    monkeypatch.setattr(hypergan_hndl_lifecycle, "build_game", forbidden)
    monkeypatch.setattr(teachers, "paired_records", forbidden)
    before_rng = torch.get_rng_state().clone()
    with open_inference(tmp_path / "native.pt") as inference:
        assert torch.equal(before_rng, torch.get_rng_state())
        assert inference.selection == selection
        assert not any(hasattr(inference, name) for name in ("game", "critic", "rows", "optimizer"))
        assert _hash(inference.binding.network.state_dict()) == _hash(packet["network"])
        for strength, value in expected.items():
            assert torch.equal(inference.render(prompt="a green vase", seed=11, strength=strength), value)
        assert _hash(inference.binding.network.state_dict()) == _hash(packet["network"])
    assert torch.equal(before_rng, torch.get_rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("corruption", ["checksum", "profile", "late_tensor", "model"])
def test_native_refusal_preserves_managed_runtime_and_owned_values(corruption, monkeypatch):
    from qwen_image_sliders import hypergan_hndl_inference as native
    from qwen_image_sliders.hypergan_hndl import QwenHNDLBank

    with BACKEND.open(config()) as training:
        training.step()
        packet = training.export_inference(selection="ema")
    packet = deepcopy(packet)
    if corruption == "checksum":
        packet["sha256"] = "0" * 64
    elif corruption == "profile":
        packet["specification"]["identity"]["protocol"] = "old-cohort"
    elif corruption == "model":
        packet["specification"]["identity"]["model"]["qwen_hndl_profile"] = "other-cohort"
    else:
        key = next(key for key, value in packet["network"].items() if tuple(value.shape) == (128, 4))
        packet["network"][key][0, 0] = float("nan")
    if corruption != "checksum":
        packet["sha256"] = _hash({key: value for key, value in packet.items() if key != "sha256"})
    runtime = TinyRuntime()
    base = deepcopy(runtime.model.state_dict())
    forwards = {name: module.forward for name, module in runtime.model.named_modules()}
    frequencies = runtime.model.pos_embed.freqs
    banks = []
    original = QwenHNDLBank.__init__

    def observed(bank, *args, **kwargs):
        original(bank, *args, **kwargs)
        banks.append((bank, deepcopy(bank.state_dict())))

    monkeypatch.setattr(QwenHNDLBank, "__init__", observed)
    monkeypatch.setattr(native, "_load_runtime", lambda local: runtime)
    rng = torch.get_rng_state().clone()
    with pytest.raises((CheckpointError, ValueError)):
        open_inference(packet, backend=INFERENCE_BACKEND)
    assert _hash(runtime.model.state_dict()) == _hash(base)
    assert all(module.forward == forwards[name] for name, module in runtime.model.named_modules())
    assert runtime.model.pos_embed.freqs is frequencies
    assert torch.equal(rng, torch.get_rng_state())
    assert len(banks) == (0 if corruption == "checksum" else 1)
    for bank, state in banks:
        assert _hash(bank.state_dict()) == _hash(state)
        assert bank.strength == 0 and bank._particles is None


def test_modern_session_checkpoint_restores_exact_raw_and_ema():
    with BACKEND.open(config()) as full:
        full.train()
        expected = full.checkpoint()
        expected_ema = full.export_inference(selection="ema")
    with BACKEND.open(config()) as partial:
        partial.train(stop_after=2)
        checkpoint = partial.checkpoint()
    with BACKEND.open(config()) as resumed:
        resumed.restore(checkpoint)
        resumed.train()
        assert resumed.checkpoint()["sha256"] == expected["sha256"]
        assert resumed.export_inference(selection="ema")["sha256"] == expected_ema["sha256"]


def test_candidate_distribution_registers_only_its_modern_product_entrypoints():
    import json
    from importlib.metadata import distribution
    from importlib.resources import files

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    from particle_sliders.inference import inference_backend
    from particle_sliders.lifecycle import backend

    package = distribution("qwen-image-particle-sliders-modern-candidate")
    assert package.version == "0.3.0a2"
    import qwen_image_sliders

    assert qwen_image_sliders.__version__ == package.version
    assert Version(package.version).is_prerelease
    assert not SpecifierSet("==0.2").contains(package.version, prereleases=True)
    assert backend("qwen-hndl") is BACKEND
    assert inference_backend("qwen-hndl") is INFERENCE_BACKEND
    for group in ("particle_sliders.image_backends", "particle_sliders.inference_backends"):
        assert [entry.name for entry in package.entry_points if entry.group == group] == ["qwen-hndl"]
    assert not {"qwen-slider-train", "qwen-slider-infer"} & {entry.name for entry in package.entry_points}
    (core,) = [
        Requirement(value) for value in package.requires if Requirement(value).name == "particle-sliders-core"
    ]
    profile = json.loads(files("qwen_image_sliders").joinpath("hypergan-hndl.lock.json").read_text())
    assert core.extras == {"hypergan-hndl"}
    assert core.url == (
        "git+https://github.com/HyperGAN/particle-sliders.git@"
        + profile["core_commit"]
        + "#subdirectory=packages/particle-sliders-core"
    )


def test_modern_factory_preserves_borrowed_rope_and_checkpoint_settings():
    runtime = TinyRuntime()
    frequencies = runtime.model.pos_embed.freqs
    settings = [
        (
            module,
            getattr(module, "gradient_checkpointing", None),
            getattr(module, "_gradient_checkpointing_func", None),
        )
        for module in runtime.model.modules()
    ]
    original = config()
    enabled = SessionConfig(
        original.backend, original.cohort, **{**original.options, "gradient_checkpointing": True}
    )
    with BACKEND.open(enabled, runtime=runtime) as session:
        assert runtime.model.pos_embed.freqs is frequencies
        session.step()
    assert runtime.model.pos_embed.freqs is frequencies
    for module, checkpointing, function in settings:
        assert getattr(module, "gradient_checkpointing", None) == checkpointing
        assert getattr(module, "_gradient_checkpointing_func", None) is function


def test_exported_training_context_controls_defaults_and_refuses_changes(monkeypatch):
    from qwen_image_sliders import hypergan_hndl_inference as native

    with BACKEND.open(config(width=64, trajectory_steps=3)) as training:
        training.step()
        packet = training.export_inference(selection="raw")
        context = packet["specification"]["configuration"]["training_context"]
        assert context == {"height": 32, "width": 64, "trajectory_steps": 3, "training_steps": 4}
        expected = training.render(prompt="a green vase", seed=11)
    rng = torch.get_rng_state().clone()
    with open_inference(packet) as inference:
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.equal(inference.render(prompt="a green vase", seed=11), expected)
        for changed in ({"height": 64}, {"width": 32}, {"steps": 4}):
            with pytest.raises(ValueError, match="exported training context"):
                inference.render(prompt="a green vase", **changed)

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible native request reached runtime construction")

    monkeypatch.setattr(native, "_load_runtime", forbidden)
    with pytest.raises(ValueError, match="exported training context"):
        open_inference(packet, native_request={"width": 32})
    changed = deepcopy(packet)
    changed["specification"]["configuration"]["training_context"]["width"] = 32
    changed["sha256"] = _hash({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="training context differs"):
        open_inference(changed)
