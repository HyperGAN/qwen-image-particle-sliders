"""The a794 core pin: recursive admission, old-cohort refusal, and cleanup retry."""

import json
import os
from copy import deepcopy
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import pytest
import torch

if os.environ.get("QWEN_TEST_CORE013") != "1":
    pytest.skip("separate modern candidate environment required", allow_module_level=True)

from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError
from particle_sliders.pg09 import _hash
from test_hypergan_hndl_lifecycle import BACKEND, config

from qwen_image_sliders.hypergan_hndl import QwenHNDLBank, _shared, build_game
from qwen_image_sliders.tiny import TinyRuntime


def test_exact_recursive_source_and_data_admission():
    profile = json.loads(files("qwen_image_sliders").joinpath("hypergan-hndl.lock.json").read_text())
    assert profile["core_commit"] == "3dd672a7f9b8c169daef90fd63e4bb42b9ddc423"
    assert "concept_slider_core/__init__.py" in profile["core_source_sha256"]
    assert "particle_sliders/LICENSE" in profile["core_source_sha256"]
    _, _, _, identity = _shared()
    assert identity["core_commit"] == profile["core_commit"]
    assert identity["core_runtime_closure_sha256"] == profile["core_source_closure_sha256"]


@pytest.mark.parametrize("tamper", ["resource_bytes", "extra_nested_resource"])
def test_recursive_resource_tamper_refuses_before_native_use(tamper):
    core = distribution("particle-sliders-core")
    if tamper == "resource_bytes":
        path = Path(core.locate_file("particle_sliders/fibo_projection.hndl"))
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# unqualified resource change\n")
            with pytest.raises(ValueError, match="shared source differs"):
                build_game(None, [])
        finally:
            path.write_bytes(original)
    else:
        folder = Path(core.locate_file("particle_sliders/_qwen_pin_test"))
        folder.mkdir(exist_ok=False)
        path = folder / "unexpected.json"
        try:
            path.write_text("{}\n")
            with pytest.raises(ValueError, match="file inventory differs"):
                build_game(None, [])
        finally:
            path.unlink()
            folder.rmdir()
    _shared()


def test_actual_previous_preview_packets_refuse_without_relabeling(monkeypatch):
    from qwen_image_sliders import hypergan_hndl_inference as native

    location = os.environ.get("QWEN_OLD_PREVIEW_DIR")
    if not location:
        pytest.skip("provide the preserved prior-preview packet directory for this qualification")
    folder = Path(location)
    receipt = json.loads((folder / "old-preview-receipt.json").read_text())
    assert receipt["core_direct_url"]["vcs_info"]["commit_id"] == "4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231"
    with BACKEND.open(config()) as session:
        session.step()
        before = session.checkpoint()["sha256"]
        rng = torch.get_rng_state().clone()
        old_game = torch.load(folder / "old-v2-game.pt", weights_only=True)
        with pytest.raises(ValueError, match="source|conditioning"):
            session.game.load_state_dict(old_game)
        assert session.checkpoint()["sha256"] == before
        assert torch.equal(rng, torch.get_rng_state())

    def forbidden(*args, **kwargs):
        pytest.fail("old preview inference reached native construction")

    monkeypatch.setattr(native, "_load_runtime", forbidden)
    for selection in ("raw", "ema"):
        path = folder / ("old-v2-native-" + selection + ".pt")
        with pytest.raises(CheckpointError):
            open_inference(path)
    import hashlib

    for name, expected in receipt["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected


def test_bank_close_attempts_every_restore_and_retains_failed_entries(monkeypatch):
    runtime = TinyRuntime()
    bank = QwenHNDLBank(runtime.model, torch.nn.Parameter(torch.zeros(128, 4)))
    entries = tuple(bank._restore)
    original = bank._restore_projection
    cleanup = bank._native_cleanup
    seen, native_cleanup = [], []
    failed = entries[-1]

    def restore(entry):
        seen.append(entry)
        if entry is failed and seen.count(entry) == 1:
            raise RuntimeError("first projection restoration failed")
        original(entry)

    def native():
        native_cleanup.append(True)
        cleanup()

    monkeypatch.setattr(bank, "_restore_projection", restore)
    monkeypatch.setattr(bank, "_native_cleanup", native)
    with pytest.raises(RuntimeError, match="projection restoration failed"):
        bank.close()
    assert len(seen) == len(entries) and native_cleanup == [True]
    assert bank._restore == [failed]
    bank.close()
    assert bank._restore == [] and native_cleanup == [True, True]


@pytest.mark.parametrize("kind", ["training", "inference"])
def test_product_close_failure_is_retryable_and_does_not_skip_runtime_cleanup(kind, monkeypatch):
    from qwen_image_sliders import hypergan_hndl_inference as native

    runtime = TinyRuntime()
    frequencies = runtime.model.pos_embed.freqs
    if kind == "training":
        session = BACKEND.open(config(), runtime=runtime)
        bank = session.game.backend.adapter
    else:
        with BACKEND.open(config()) as training:
            training.step()
            packet = training.export_inference(selection="ema")
        monkeypatch.setattr(native, "_load_runtime", lambda local: runtime)
        banks = []
        original_init = QwenHNDLBank.__init__

        def observed(bank, *args, **kwargs):
            original_init(bank, *args, **kwargs)
            banks.append(bank)

        monkeypatch.setattr(QwenHNDLBank, "__init__", observed)
        session = open_inference(packet)
        bank = banks[0]
    original = bank._native_cleanup
    calls = []

    def fail_once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("native cleanup failed once")
        original()

    monkeypatch.setattr(bank, "_native_cleanup", fail_once)
    runtime.model.pos_embed.freqs = []
    runtime.model.train()
    try:
        raise ValueError("unrelated ambient exception")
    except ValueError:
        with pytest.raises(RuntimeError, match="native cleanup failed once"):
            session.close()
    assert not session.closed
    assert runtime.model.pos_embed.freqs is frequencies
    assert not runtime.model.training
    session.close()
    assert session.closed and len(calls) == 2


def test_context_exit_preserves_primary_and_allows_cleanup_retry(monkeypatch):
    session = BACKEND.open(config())
    bank = session.game.backend.adapter
    original = bank._native_cleanup
    calls = []

    def fail_once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("secondary cleanup failure")
        original()

    monkeypatch.setattr(bank, "_native_cleanup", fail_once)
    with pytest.raises(ValueError, match="primary operation failure") as caught, session:
        raise ValueError("primary operation failure")
    assert any("secondary cleanup failure" in note for note in caught.value.__notes__)
    assert not session.closed
    session.close()
    assert session.closed


def test_restore_ema_preflight_failure_preserves_live_state_and_retries(monkeypatch):
    with BACKEND.open(config()) as session:
        session.step()
        packet = session.checkpoint()
        expected_metrics = session.step()
        expected = session.checkpoint()["sha256"]
        before = _hash(session.ema.state_dict())
        rng = torch.get_rng_state().clone()
        original = type(session.ema).load_state_dict
        calls = []

        def fail_once(ema, *args, **kwargs):
            calls.append(True)
            assert ema is not session.ema
            if len(calls) == 1:
                ema.values = None
                torch.rand(1)
                raise RuntimeError("isolated EMA conversion failure")
            return original(ema, *args, **kwargs)

        monkeypatch.setattr(type(session.ema), "load_state_dict", fail_once)
        with pytest.raises(CheckpointError):
            session.restore(deepcopy(packet))
        assert _hash(session.ema.state_dict()) == before
        assert session.checkpoint()["sha256"] == expected
        assert torch.equal(rng, torch.get_rng_state())
        session.restore(packet)
        assert session.step() == expected_metrics
        assert session.checkpoint()["sha256"] == expected


def test_setup_failure_preserves_primary_and_attempts_native_restore(monkeypatch):
    from qwen_image_sliders import hypergan_hndl_lifecycle as lifecycle

    runtime = TinyRuntime()
    frequencies = runtime.model.pos_embed.freqs
    forwards = {name: module.forward for name, module in runtime.model.named_modules()}
    banks = []
    original_init = QwenHNDLBank.__init__

    def observed(bank, *args, **kwargs):
        original_init(bank, *args, **kwargs)
        original = bank._native_cleanup
        calls = []

        def fail_once():
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("secondary setup cleanup failure")
            original()

        bank._native_cleanup = fail_once
        banks.append(bank)

    def fail_specification(*args, **kwargs):
        runtime.model.pos_embed.freqs = []
        runtime.model.train()
        raise ValueError("primary setup failure")

    monkeypatch.setattr(QwenHNDLBank, "__init__", observed)
    monkeypatch.setattr(lifecycle, "specification", fail_specification)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="primary setup failure") as caught:
        BACKEND.open(config(), runtime=runtime)
    assert any("secondary setup cleanup failure" in note for note in caught.value.__notes__)
    assert runtime.model.pos_embed.freqs is frequencies
    assert not runtime.model.training
    assert all(module.forward == forwards[name] for name, module in runtime.model.named_modules())
    assert torch.equal(rng, torch.get_rng_state())
    banks[0].close()
