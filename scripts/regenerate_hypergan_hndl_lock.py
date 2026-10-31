"""Regenerate the active modern lock from the one reviewed immutable core tree."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import tomllib
import zipfile
from pathlib import Path

TARGET = "a794e3b4a8813b012f2c4faa3da82b78d4a90214"
PREFIX = "packages/particle-sliders-core/"


def regenerate(repository, wheel, destination):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repository)

    if git("rev-parse", "HEAD").decode().strip() != TARGET:
        raise ValueError("the core checkout must be at the exact reviewed commit")
    if git("diff", "--name-only", TARGET, "--", PREFIX).strip():
        raise ValueError("the core package must have no tracked changes before lock generation")
    document = git("show", TARGET + ":" + PREFIX + "pyproject.toml")
    metadata = tomllib.loads(document.decode())
    entries = git("ls-tree", "-r", "--name-only", TARGET, PREFIX + "src").decode().splitlines()
    sources = {}
    for path in entries:
        relative = path.removeprefix(PREFIX + "src/")
        if relative.split("/", 1)[0] not in {"particle_sliders", "concept_slider_core"}:
            raise ValueError("unreviewed core runtime package root: " + relative)
        sources[relative] = hashlib.sha256(git("show", TARGET + ":" + path)).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.split("/", 1)[0] in {"particle_sliders", "concept_slider_core"}
        }
    if actual != sources:
        raise ValueError("the built core wheel differs from the recursive Git runtime closure")
    previous = json.loads(destination.read_text())
    record = {
        **previous,
        "schema": "qwen-hypergan-hndl-candidate-v2",
        "profile": "qwen-image21-hypergan-hndl-v3",
        "product_version": "0.3.0a2",
        "core_commit": TARGET,
        "core_version": metadata["project"]["version"],
        "core_roots": ["concept_slider_core", "particle_sliders"],
        "core_src_tree": git("rev-parse", TARGET + ":" + PREFIX + "src").decode().strip(),
        "core_source_sha256": dict(sorted(sources.items())),
        "core_source_closure_sha256": hashlib.sha256(
            json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "core_pyproject_sha256": hashlib.sha256(document).hexdigest(),
        "core_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "core_wheel_build": {
            "source_date_epoch": int(git("show", "-s", "--format=%ct", TARGET)),
            "setuptools": importlib.metadata.version("setuptools"),
            "purpose": "Reproducible local qualification artifact; not a published release.",
        },
        "qualification": "CPU software candidate; native model and GPU qualification remain product-specific.",
        "previous_preview": previous["previous_preview"]
        if previous.get("core_commit") == TARGET
        else {
            "profile": previous["profile"],
            "core_commit": previous["core_commit"],
            "product_version": "0.3.0a1",
            "product_head": "ecb50267562afa0eb374000d9761b57b0e25fde7",
            "status": "Use the original environment. No automatic packet migration is supported.",
        },
    }
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-checkout", type=Path, required=True)
    parser.add_argument("--core-wheel", type=Path, required=True)
    arguments = parser.parse_args()
    destination = Path(__file__).resolve().parents[1] / "src/qwen_image_sliders/hypergan-hndl.lock.json"
    result = regenerate(arguments.core_checkout, arguments.core_wheel, destination)
    print(
        json.dumps(
            {
                "core_commit": result["core_commit"],
                "runtime_files": len(result["core_source_sha256"]),
                "closure_sha256": result["core_source_closure_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
