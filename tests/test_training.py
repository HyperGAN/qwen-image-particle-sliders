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


def test_exact_cpu_resume_and_ema_inference(tmp_path):
    rows = load_prompts()
    full, split = tmp_path / "full", tmp_path / "split"
    meta = train(TinyRuntime(), rows, full, steps=4)
    train(TinyRuntime(), rows, split, steps=2)
    train(TinyRuntime(), rows, split, steps=4, resume=split / "training.pt")
    assert_same(
        torch.load(full / "training.pt", weights_only=True),
        torch.load(split / "training.pt", weights_only=True),
    )
    assert file_digest(full / "adapter.safetensors") == file_digest(split / "adapter.safetensors")
    assert meta["quality_validated"] is False
    assert meta["validation_kind"] == "tiny-native-random-weights-synthetic-text"
    assert meta["formulation"]["architecture_id"] == "gmix"
    assert meta["formulation"]["provisional"] is True
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
        for key in ["base", "core_commit", "diffusers_commit", "protocol", "targets", "model_config_sha256"]:
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
