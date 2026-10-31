"""Admit only the distinct prerelease distribution and its actual import paths."""

import hashlib
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

DIST = "qwen-image-particle-sliders-modern-candidate"
VERSION = "0.3.0a1"


def module_origin(owner, name, relative):
    expected = Path(distribution(owner).locate_file(relative)).resolve()
    module = sys.modules.get(name)
    try:
        spec = importlib.util.find_spec(name)
    except (ValueError, ImportError) as error:
        raise RuntimeError("cannot verify imported module origin: " + name) from error
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError("imported module origin differs from its distribution: " + name)
    actual = getattr(module, "__file__", None) if module is not None else None
    if module is not None and (actual is None or Path(actual).resolve() != expected):
        raise RuntimeError("loaded module origin differs from its distribution: " + name)
    return expected


def isolated_runtime():
    try:
        distribution("qwen-image-particle-sliders")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError("modern Qwen requires a separate environment without the stable distribution")
    try:
        product = distribution(DIST)
    except PackageNotFoundError as error:
        raise RuntimeError(
            "install the distinct modern candidate distribution in an isolated environment"
        ) from error
    if product.version != VERSION:
        raise RuntimeError("modern Qwen candidate distribution version differs")
    records = tuple(product.files or ())
    if not {"qwen_image_sliders/__init__.py", "qwen_image_sliders/hypergan_hndl_runtime.py"}.issubset(
        map(str, records)
    ):
        raise RuntimeError("modern Qwen requires its built wheel and complete file manifest")
    sources = {}
    for path in records:
        relative = str(path)
        if relative.startswith("qwen_image_sliders/") and relative.endswith(".py"):
            name = relative[:-3].replace("/", ".").removesuffix(".__init__")
            actual = module_origin(DIST, name, relative)
            sources[relative] = hashlib.sha256(actual.read_bytes()).hexdigest()
    for owner, name in (
        ("particle-sliders-core", "particle_sliders"),
        ("particle-sliders-core", "concept_slider_core"),
        ("particlegan", "particlegan"),
        ("hndl", "hndl"),
        ("hypergan", "hypergan"),
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
    ):
        module_origin(owner, name, name.replace(".", "/") + "/__init__.py")
    return sources


def loaded_dependency_origins():
    """Catch already imported submodules redirected outside their owning wheel."""
    for owner, prefix in (
        ("particle-sliders-core", "particle_sliders"),
        ("particle-sliders-core", "concept_slider_core"),
        ("particlegan", "particlegan"),
        ("hndl", "hndl"),
        ("hypergan", "hypergan"),
        ("diffusers", "diffusers"),
    ):
        package = distribution(owner)
        files = {Path(package.locate_file(path)).resolve() for path in package.files or ()}
        for name, module in tuple(sys.modules.items()):
            if name == prefix or name.startswith(prefix + "."):
                path = getattr(module, "__file__", None)
                if path is not None and Path(path).resolve() not in files:
                    raise RuntimeError("loaded dependency module is outside its distribution: " + name)
