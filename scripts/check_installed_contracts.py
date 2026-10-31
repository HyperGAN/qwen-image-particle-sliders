"""Run S081 tests against a non-editable installed wheel, outside the checkout."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="qwen-installed-") as directory:
        probe = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import json, qwen_image_sliders; print(json.dumps(qwen_image_sliders.__file__))",
            ],
            cwd=directory,
            text=True,
            capture_output=True,
            check=True,
        )
        if Path(json.loads(probe.stdout)).resolve().is_relative_to(root / "src"):
            raise RuntimeError("install the built wheel, rather than an editable source checkout")
        return subprocess.run(
            [sys.executable, "-I", "-m", "pytest", "-q", "-o", "pythonpath=", str(root / "tests")],
            cwd=directory,
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
