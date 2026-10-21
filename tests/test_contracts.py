import json

import pytest

from qwen_image_sliders.contracts import file_digest, formulation, validate_shape
from qwen_image_sliders.runtime import GIB, require_resources, verify_snapshot
from qwen_image_sliders.teachers import load_prompts, paired_records
from qwen_image_sliders.tiny import TinyRuntime


def test_formulation_drift_and_shape_validation():
    _stamp, declared = formulation(batch=2, g_lr=2e-5)
    assert declared["adv_batch"] == 2 and declared["g_lr"] == 2e-5
    changed = {**declared, "noise_hold_ratio": 1.0}
    with pytest.raises(ValueError):
        formulation(declared=changed)
    for size in [(16, 32), (33, 32), (0, 32)]:
        with pytest.raises(ValueError):
            validate_shape(*size)


def test_resource_guards_are_unit_checks_without_loading():
    report = {
        "host_available_bytes": 90 * GIB,
        "cuda_visible": True,
        "gpu_free_bytes": 40 * GIB,
        "disk_free_bytes": 50 * GIB,
    }
    require_resources(report, download=True, device="cuda:0")
    for key in ["host_available_bytes", "gpu_free_bytes", "disk_free_bytes"]:
        with pytest.raises(RuntimeError, match="blocked"):
            require_resources({**report, key: 1}, download=True, device="cuda:0")
    with pytest.raises(RuntimeError, match="visible"):
        require_resources({**report, "cuda_visible": False}, device="cuda:0")


def test_snapshot_checks_are_mocked_manifest_unit_checks(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}")
    manifest = {"config.json": {"bytes": 2, "sha256": file_digest(config)}}
    monkeypatch.setattr("qwen_image_sliders.runtime.lock", lambda: {"model": {"files": manifest}})
    assert verify_snapshot(tmp_path) == tmp_path
    config.write_text("[]")
    with pytest.raises(ValueError, match="hash"):
        verify_snapshot(tmp_path)
    config.write_text("{}")
    (tmp_path / "model.bin").write_bytes(b"unverified")
    with pytest.raises(ValueError, match="unverified"):
        verify_snapshot(tmp_path)


def test_prompt_and_record_budget_validation(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"rows": [{"neutral": "x", "positive": "y"}]}))
    with pytest.raises(ValueError, match="two"):
        load_prompts(path)
    with pytest.raises(ValueError, match="128"):
        paired_records(
            TinyRuntime(),
            load_prompts() * 30,
            seed=7,
            height=32,
            width=32,
            trajectory_steps=4,
            records_per_pair=2,
        )
