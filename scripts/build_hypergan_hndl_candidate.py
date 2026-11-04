"""Build a separate candidate profile from a verified core wheel/public commit.

The default pyproject and canonical source tree are never rewritten. The resulting
distinct prerelease wheel binds one exact reviewed core and registers only qwen-hndl.
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
        actual = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.split("/", 1)[0] in profile["core_roots"]
        }
        if actual != profile["core_source_sha256"]:
            raise ValueError("core wheel recursive source/data closure differs")
    return profile["core_wheel_sha256"]


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
    tree = json.loads(
        _public_bytes(
            "https://api.github.com/repos/HyperGAN/particle-sliders/git/trees/"
            + profile["core_src_tree"]
            + "?recursive=1"
        )
    )
    paths = {node["path"] for node in tree.get("tree", []) if node["type"] == "blob"}
    if (
        tree.get("truncated")
        or tree.get("sha") != profile["core_src_tree"]
        or paths != set(profile["core_source_sha256"])
    ):
        raise ValueError("public core recursive source/data inventory differs from the reviewed closure")

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
    local_hash = None if public_core else verified_core(core_wheel, profile)
    # Even local-artifact qualification emits a portable immutable dependency.
    # If publication is unavailable, fail rather than embedding a file:// URL.
    core_url = verified_public_core(profile)
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
        'name = "qwen-image-particle-sliders"': 'name = "qwen-image-particle-sliders-modern-candidate"',
        'version = "' + document["project"]["version"] + '"': 'version = "'
        + profile["product_version"]
        + '"',
        'requires-python = ">=3.11"': 'requires-python = ">=3.11,<3.15"',
        "hndl==0.8.2": "hndl==0.10.0",
        'qwen-slider-train = "qwen_image_sliders.train:main"\n': "",
        'qwen-slider-infer = "qwen_image_sliders.infer:main"\n': "",
        'qwen = "qwen_image_sliders.lifecycle:BACKEND"': 'qwen-hndl = "qwen_image_sliders.hypergan_hndl_lifecycle:BACKEND"',
        'qwen = "qwen_image_sliders.inference:INFERENCE_BACKEND"': 'qwen-hndl = "qwen_image_sliders.hypergan_hndl_inference:INFERENCE_BACKEND"',
        'qwen_image_sliders = ["backend.lock.json", "prompts.json"]': 'qwen_image_sliders = ["*.json"]',
        '[tool.setuptools.cmdclass]\nbuild_py = "_qwen_build_support.StableBuildPy"\n': "",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError("default metadata changed; review the explicit modern profile transformation")
        source = source.replace(before, after)
    tomllib.loads(source)
    (destination / "pyproject.toml").write_text(source)
    package_init = destination / "src/qwen_image_sliders/__init__.py"
    package_init.write_text(
        package_init.read_text().replace(
            '__version__ = "' + document["project"]["version"] + '"',
            '__version__ = "' + profile["product_version"] + '"',
        )
        + "\nfrom importlib.metadata import PackageNotFoundError as _Missing, distribution as _distribution\n"
        + 'try:\n    _distribution("qwen-image-particle-sliders")\n'
        + "except _Missing:\n    pass\n"
        + 'else:\n    raise RuntimeError("modern Qwen requires a separate environment without the stable distribution")\n'
    )
    readme = destination / "README.md"
    readme.write_text(
        "Isolated HyperGAN/HNDL prerelease; never co-install the stable distribution.\n"
        "Requires the exact reviewed core; default-profile promotion remains gated.\n\n" + readme.read_text()
    )
    return {
        "profile": profile["profile"],
        "core_url": core_url,
        "core_commit": profile["core_commit"],
        "core_src_tree": profile["core_src_tree"],
        "core_source_closure_sha256": profile["core_source_closure_sha256"],
        "public_sources_verified": True,
        "verified_local_core_sha256": local_hash,
        "default_profile_promoted": False,
        "pyproject": source,
    }


def copy_new_artifacts(generated, output, receipt):
    """Receipt only the fresh isolated build outputs, never existing outdir files."""
    artifacts = sorted(
        path
        for path in generated.iterdir()
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    )
    if len(artifacts) != 2:
        raise ValueError("expected exactly the newly built wheel and sdist")
    output.mkdir(parents=True, exist_ok=True)
    receipt["artifacts"] = {}
    for path in artifacts:
        shutil.copyfile(path, output / path.name)
        receipt["artifacts"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "candidate-build.json").write_text(json.dumps(receipt, indent=2) + "\n")


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
        generated = staged / "candidate-dist"
        subprocess.run(
            [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(generated), str(staged)],
            check=True,
        )
        copy_new_artifacts(generated, output, receipt)


if __name__ == "__main__":
    main()
