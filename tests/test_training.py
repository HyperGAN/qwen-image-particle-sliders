import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from qwen_image_sliders.adapters import AdapterBank
from qwen_image_sliders.checkpoint import load_adapter, metadata, read_metadata
from qwen_image_sliders.contracts import canonical, file_digest, formulation
from qwen_image_sliders.infer import infer
from qwen_image_sliders.teachers import load_prompts
from qwen_image_sliders.tiny import TinyRuntime
from qwen_image_sliders.train import train


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_same(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_exact_cpu_resume_and_ema_inference(tmp_path, preset):
    rows = load_prompts()
    full, split = tmp_path / "full", tmp_path / "split"
    meta = train(TinyRuntime(), rows, full, steps=4, preset=preset)
    train(TinyRuntime(), rows, split, steps=4, stop_after=2, preset=preset)
    train(TinyRuntime(), rows, split, steps=4, resume=split / "training.pt", preset=preset)
    assert_same(
        torch.load(full / "training.pt", weights_only=True),
        torch.load(split / "training.pt", weights_only=True),
    )
    assert file_digest(full / "adapter.safetensors") == file_digest(split / "adapter.safetensors")
    assert meta["quality_validated"] is False
    assert meta["validation_kind"] == "tiny-native-random-weights-synthetic-text"
    assert meta["formulation"]["architecture_id"] == "gmix"
    assert meta["protocol"]["engine"] == "particle_sliders.pg09.ConditionalPG09"
    assert meta["shared_api"]["particlegan_version"] == "0.9.1"
    kwargs = {"seed": 42, "height": 32, "width": 32, "steps": 4, "output_type": "latent"}
    runtime = TinyRuntime()
    vanilla = runtime.sample(runtime.encode("A cup."), **kwargs)
    zero = infer(runtime, full / "adapter.safetensors", "A cup.", strength=0, **kwargs)
    assert torch.equal(zero, vanilla)
    active = infer(runtime, full / "adapter.safetensors", "A cup.", strength=1, **kwargs)
    assert torch.isfinite(active).all() and not torch.equal(active, vanilla)
    assert not any(m._forward_hooks for m in runtime.model.modules())
    with pytest.raises(ValueError, match="resume"):
        train(TinyRuntime(), rows, tmp_path / "bad", steps=4, seed=8, resume=split / "training.pt")


@pytest.fixture
def export(tmp_path):
    train(TinyRuntime(), load_prompts(), tmp_path, steps=1)
    return tmp_path / "adapter.safetensors"


def test_strict_artifact_metadata_and_tensors(export, tmp_path):
    meta = read_metadata(export)
    runtime = TinyRuntime()
    stamp, config = formulation(declared=meta["formulation"]["declared"])
    bank = AdapterBank(runtime.model, stamp)
    expected = metadata(runtime, bank, stamp, config)
    particles = torch.zeros(128, 4)
    try:
        assert not any(k.startswith("transformer") for k in load_file(str(export)))
        for key in [
            "base",
            "core_commit",
            "shared_api",
            "product_sources",
            "diffusers_commit",
            "protocol",
            "targets",
            "model_config_sha256",
        ]:
            changed = {**meta, key: "tampered"}
            target = tmp_path / f"bad-{key}.safetensors"
            save_file(
                load_file(str(export)), str(target), metadata={"qwen_image_sliders": canonical(changed)}
            )
            with pytest.raises(ValueError, match="mismatch"):
                load_adapter(target, bank, particles, expected)
        for bad in [torch.zeros(1, 4), torch.full((128, 4), float("nan"))]:
            state = load_file(str(export))
            state["particles"] = bad
            target = tmp_path / "bad-tensor.safetensors"
            save_file(state, str(target), metadata={"qwen_image_sliders": canonical(meta)})
            with pytest.raises(ValueError, match="invalid"):
                load_adapter(target, bank, particles, expected)
    finally:
        bank.close()


def test_packaged_pins_and_example_match_repository():
    root = Path(__file__).resolve().parents[1]
    package = root / "src/qwen_image_sliders"
    assert (package / "backend.lock.json").read_bytes() == (root / "backend.lock.json").read_bytes()
    assert json.loads((package / "prompts.json").read_text()) == json.loads(
        (root / "configs/warm-light.json").read_text()
    )


def test_artifact_architecture_identity_is_portable_between_snapshot_paths():
    runtime = TinyRuntime()
    stamp, declared = formulation()
    bank = AdapterBank(runtime.model, stamp)
    try:
        first = metadata(runtime, bank, stamp, declared)
        runtime.model.register_to_config(_name_or_path="/different/snapshot/location")
        assert metadata(runtime, bank, stamp, declared) == first
        runtime.model.register_to_config(mlp_ratio=3)
        assert metadata(runtime, bank, stamp, declared) != first
    finally:
        bank.close()


def test_failed_build_restores_native_checkpoint_context_and_hooks(tmp_path, monkeypatch):
    import qwen_image_sliders.train as trainer

    runtime = TinyRuntime()
    before = [
        (
            module,
            getattr(module, "gradient_checkpointing", None),
            getattr(module, "_gradient_checkpointing_func", None),
        )
        for module in runtime.model.modules()
    ]

    def fail(*args, **kwargs):
        raise RuntimeError("injected common game failure")

    monkeypatch.setattr(trainer, "ConditionalPG09", fail)
    with pytest.raises(RuntimeError, match="injected common game failure"):
        train(runtime, load_prompts(), tmp_path, gradient_checkpointing=True)
    assert not any(module._forward_hooks for module in runtime.model.modules())
    for module, enabled, function in before:
        assert getattr(module, "gradient_checkpointing", None) == enabled
        assert getattr(module, "_gradient_checkpointing_func", None) is function


def test_resume_rejects_budget_drift_and_corrupt_ema_before_restore(tmp_path, monkeypatch):
    import qwen_image_sliders.train as trainer

    rows = load_prompts()
    train(TinyRuntime(), rows, tmp_path / "split", steps=4, stop_after=2)
    checkpoint = tmp_path / "split/training.pt"
    with pytest.raises(ValueError, match="fixed training budget"):
        train(TinyRuntime(), rows, tmp_path / "wrong-budget", steps=5, resume=checkpoint)
    packet = torch.load(checkpoint, weights_only=True)
    packet["ema_particles"][0, 0] = float("nan")
    bad = tmp_path / "bad-ema.pt"
    torch.save(packet, bad)
    calls = []
    original = trainer.ConditionalPG09.load_state_dict

    def observed(game, state):
        calls.append(True)
        return original(game, state)

    monkeypatch.setattr(trainer.ConditionalPG09, "load_state_dict", observed)
    runtime = TinyRuntime()
    with pytest.raises(ValueError, match="EMA"):
        train(runtime, rows, tmp_path / "wrong-ema", steps=4, resume=bad)
    assert not calls
    assert not any(module._forward_hooks for module in runtime.model.modules())


def test_hndl_critic_executes_shared_native_gan_and_resumes(tmp_path):
    pytest.importorskip("hndl")
    from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic

    from qwen_image_sliders.teachers import paired_records
    from qwen_image_sliders.train import build_game, step_context

    runtime = TinyRuntime()
    captured = paired_records(runtime, load_prompts(), seed=7, height=32, width=32, trajectory_steps=4)
    stamp, declared = formulation(batch=2)
    game = build_game(runtime, captured, stamp, declared, 7, critic_kind="hndl")
    try:
        assert isinstance(game.critic, HNDLConditionalReadoutCritic)
        assert game.critic.architecture()["hndl_version"] == "0.8.2"
        step_context(game)
        state = game.state_dict()
        expected = step_context(game)
        game.load_state_dict(state)
        assert_same(expected, step_context(game))
        assert game.completed_steps == 2
    finally:
        game.backend.adapter.close()
