"""Refuse an altered core artifact before staging a candidate package."""

import importlib.util
from pathlib import Path

import pytest


def test_candidate_builder_rejects_unverified_core_before_writing_profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "candidate_builder", root / "scripts/build_hypergan_hndl_candidate.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    wheel = tmp_path / "particle_sliders_core-0.12.0-py3-none-any.whl"
    wheel.write_bytes(b"unqualified replacement")
    destination = tmp_path / "profile"
    with pytest.raises(ValueError, match="exact reviewed candidate artifact"):
        builder.stage(root, destination, wheel)
    assert not destination.exists()


def test_public_candidate_rejects_changed_metadata_before_staging(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "candidate_builder", root / "scripts/build_hypergan_hndl_candidate.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "_public_bytes", lambda url: b"unreviewed package metadata")
    destination = tmp_path / "profile"
    with pytest.raises(ValueError, match="public core package metadata differs"):
        builder.stage(root, destination, public_core=True)
    assert not destination.exists()
