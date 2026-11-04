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


def test_receipt_ignores_stale_outdir_artifacts(tmp_path):
    import json

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "candidate_builder", root / "scripts/build_hypergan_hndl_candidate.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    generated, output = tmp_path / "new", tmp_path / "out"
    generated.mkdir()
    output.mkdir()
    (generated / "new.whl").write_bytes(b"fresh wheel")
    (generated / "new.tar.gz").write_bytes(b"fresh sdist")
    (output / "old.whl").write_bytes(b"stale wheel")
    builder.copy_new_artifacts(generated, output, {})
    receipt = json.loads((output / "candidate-build.json").read_text())
    assert set(receipt["artifacts"]) == {"new.whl", "new.tar.gz"}
    assert (output / "old.whl").read_bytes() == b"stale wheel"


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


def test_public_candidate_rejects_unlocked_recursive_data(monkeypatch):
    import hashlib
    import json

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "candidate_builder", root / "scripts/build_hypergan_hndl_candidate.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    document = b"verified package metadata"
    profile = {
        "core_commit": "a794e3b4a8813b012f2c4faa3da82b78d4a90214",
        "core_pyproject_sha256": hashlib.sha256(document).hexdigest(),
        "core_src_tree": "a" * 40,
        "core_source_sha256": {"particle_sliders/LICENSE": "b" * 64},
    }

    def response(url):
        if url.endswith("pyproject.toml"):
            return document
        return json.dumps(
            {
                "sha": "a" * 40,
                "truncated": False,
                "tree": [
                    {"path": "particle_sliders/LICENSE", "type": "blob"},
                    {"path": "particle_sliders/new/data.json", "type": "blob"},
                ],
            }
        ).encode()

    monkeypatch.setattr(builder, "_public_bytes", response)
    with pytest.raises(ValueError, match="recursive source/data inventory differs"):
        builder.verified_public_core(profile)


def test_candidate_metadata_transformation_survives_canonical_version_bump(tmp_path, monkeypatch):
    import tomllib

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "candidate_builder", root / "scripts/build_hypergan_hndl_candidate.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Only the source-verification transport is mocked; metadata staging is real.
    monkeypatch.setattr(
        builder,
        "verified_public_core",
        lambda profile: (
            "git+https://github.com/HyperGAN/particle-sliders.git@"
            + profile["core_commit"]
            + "#subdirectory=packages/particle-sliders-core"
        ),
    )
    destination = tmp_path / "candidate"
    receipt = builder.stage(root, destination, public_core=True)
    metadata = tomllib.loads(receipt["pyproject"])["project"]
    assert metadata["name"] == "qwen-image-particle-sliders-modern-candidate"
    assert metadata["version"] == "0.3.0a2"
    assert "qwen-slider-train" not in metadata["scripts"]
    assert '__version__ = "0.3.0a2"' in (destination / "src/qwen_image_sliders/__init__.py").read_text()
    for name in ("contracts.py", "runtime.py"):
        source = (destination / "src/qwen_image_sliders" / name).read_text()
        assert "from .hypergan_hndl import _shared as require_current_core" in source
        assert "from .core_profile import require_current_core" not in source
