"""Admit the canonical installed core/PG source cohort before native allocation."""

from importlib.metadata import distribution
from pathlib import Path

from .contracts import file_digest, lock


def require_current_core():
    pins = lock()
    for package, key, roots in (
        ("particle-sliders-core", "core", ("particle_sliders", "concept_slider_core")),
        ("particlegan", "particlegan", ("particlegan",)),
    ):
        installed = distribution(package)
        profile = pins[key]
        if installed.version != profile["version"]:
            raise RuntimeError("Qwen canonical cohort requires " + package + "==" + profile["version"])
        actual = {}
        for name in roots:
            root = Path(installed.locate_file(name))
            actual.update(
                {
                    path.relative_to(root.parent).as_posix(): path
                    for path in root.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                }
            )
        if actual.keys() != profile["source_sha256"].keys():
            raise RuntimeError("Qwen canonical " + package + " source inventory differs")
        for relative, expected in profile["source_sha256"].items():
            if file_digest(actual[relative]) != expected:
                raise RuntimeError("Qwen canonical shared source differs: " + relative)
