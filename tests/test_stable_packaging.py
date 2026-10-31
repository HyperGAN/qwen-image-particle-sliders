"""The stable wheel must not expose the modern runtime, even after dirty builds."""

import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def test_stable_wheel_excludes_modern_runtime_and_dirty_build_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    dirty = root / "build/lib/qwen_image_sliders"
    dirty.mkdir(parents=True, exist_ok=True)
    (dirty / "hypergan_hndl_stale.py").write_text("stale = True\n")
    (dirty / "hypergan-hndl.lock.json").write_text("{}\n")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(tmp_path), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (source,) = tmp_path.glob("*.tar.gz")
    with tarfile.open(source) as archive:
        assert any(
            name.endswith("/src/qwen_image_sliders/hypergan_hndl_runtime.py") for name in archive.getnames()
        )
        assert any(
            name.endswith("/src/qwen_image_sliders/hypergan-hndl.lock.json") for name in archive.getnames()
        )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any("hypergan_hndl" in name or name.endswith("hypergan-hndl.lock.json") for name in names)
        assert "qwen_image_sliders/train.py" in names
        assert "qwen_image_sliders/backend.lock.json" in names
        (meta,) = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = BytesParser().parsebytes(archive.read(meta))
        assert metadata["Name"] == "qwen-image-particle-sliders" and metadata["Version"] == "0.2.0"
        (core,) = [
            value for value in metadata.get_all("Requires-Dist") if value.startswith("particle-sliders-core")
        ]
        assert "32f6830f6402e85aa77b6631db64fcbbb2131f6b" in core
