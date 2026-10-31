"""Build a separate candidate profile from a verified core wheel/public commit.

The default pyproject and S081 source tree are never rewritten. The resulting
local-version wheel binds one exact reviewed core and registers only qwen-hndl.
Default/public promotion is a separate review; --public-core verifies live sources.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from pathlib import Path


def verified_core(path, profile):
    path = path.resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["core_wheel_sha256"]:
        raise ValueError("core wheel differs from the exact reviewed candidate artifact")
    with zipfile.ZipFile(path) as archive:
        (name,) = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = BytesParser().parsebytes(archive.read(name))
        if metadata["Name"] != "particle-sliders-core" or metadata["Version"] != profile["core_version"]:
            raise ValueError("core wheel package/version differs")
        for relative, expected in profile["core_source_sha256"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError("core wheel source differs: " + relative)
    return path.as_uri() + "#sha256=" + profile["core_wheel_sha256"]


def _public_bytes(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def verified_public_core(profile):
    commit = profile["core_commit"]
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("the reviewed public core requires an immutable full commit")
    root = f"https://raw.githubusercontent.com/HyperGAN/particle-sliders/{commit}/packages/particle-sliders-core/"
    if hashlib.sha256(_public_bytes(root + "pyproject.toml")).hexdigest() != profile["core_pyproject_sha256"]:
        raise ValueError("public core package metadata differs from the reviewed candidate")

    def verify(item):
        relative, expected = item
        if hashlib.sha256(_public_bytes(root + "src/" + relative)).hexdigest() != expected:
            raise ValueError("public core source differs: " + relative)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(verify, profile["core_source_sha256"].items()))
    return (
        "git+https://github.com/HyperGAN/particle-sliders.git@"
        + commit
        + "#subdirectory=packages/particle-sliders-core"
    )


def stage(root, destination, core_wheel=None, *, public_core=False):
    profile = json.loads((root / "src/qwen_image_sliders/hypergan-hndl.lock.json").read_text())
    if public_core == (core_wheel is not None):
        raise ValueError("select one exact local core wheel or verified public core")
    core_url = verified_public_core(profile) if public_core else verified_core(core_wheel, profile)
    for name in ("src", "tests", "docs", "configs", "scripts"):
        shutil.copytree(
            root / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    for name in (
        "README.md",
        "LICENSE",
        "NOTICE",
        "QWEN-RESEARCH-LICENSE.txt",
        "MANIFEST.in",
        "LIFECYCLE.md",
        "FORMULATION.md",
        "PROVENANCE.md",
        "VALIDATION.md",
        "PUBLICATION-STATUS.md",
        "backend.lock.json",
    ):
        shutil.copyfile(root / name, destination / name)
    source = (root / "pyproject.toml").read_text()
    document = tomllib.loads(source)
    (original,) = [
        value for value in document["project"]["dependencies"] if value.startswith("particle-sliders-core ")
    ]
    changes = {
        original: "particle-sliders-core[hypergan-hndl] @ " + core_url,
        'version = "0.2.0"': 'version = "0.2.0+hypergan.hndl.local"',
        'requires-python = ">=3.11"': 'requires-python = ">=3.11,<3.15"',
        "hndl==0.8.2": "hndl==0.10.0",
        'qwen-slider-train = "qwen_image_sliders.train:main"\n': "",
        'qwen-slider-infer = "qwen_image_sliders.infer:main"\n': "",
        'qwen = "qwen_image_sliders.lifecycle:BACKEND"': 'qwen-hndl = "qwen_image_sliders.hypergan_hndl_lifecycle:BACKEND"',
        'qwen = "qwen_image_sliders.inference:INFERENCE_BACKEND"': 'qwen-hndl = "qwen_image_sliders.hypergan_hndl_inference:INFERENCE_BACKEND"',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError("default metadata changed; review the explicit modern profile transformation")
        source = source.replace(before, after)
    tomllib.loads(source)
    (destination / "pyproject.toml").write_text(source)
    readme = destination / "README.md"
    readme.write_text(
        "Local HyperGAN/HNDL candidate profile; use the qwen-hndl shared backend.\n"
        "Requires the exact reviewed core; default-profile promotion remains gated.\n\n" + readme.read_text()
    )
    return {
        "profile": profile["profile"],
        "core_url": core_url,
        "core_commit": profile["core_commit"],
        "public_sources_verified": public_core,
        "default_profile_promoted": False,
        "pyproject": source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    core = parser.add_mutually_exclusive_group(required=True)
    core.add_argument("--core-wheel", type=Path)
    core.add_argument(
        "--public-core",
        action="store_true",
        help="verify the locked public commit and use its exact VCS source",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = arguments.outdir.resolve()
    with tempfile.TemporaryDirectory(prefix="qwen-hndl-profile-") as directory:
        staged = Path(directory)
        receipt = stage(root, staged, arguments.core_wheel, public_core=arguments.public_core)
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(staged)],
            check=True,
        )
        receipt["artifacts"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
            if path.suffix in (".whl", ".gz")
        }
        (output / "candidate-build.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
