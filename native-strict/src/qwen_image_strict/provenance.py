"""Exact new candidate/base/core/runtime bytes, checked before native allocation."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

from . import BASE, CORE
from .assets import digest

VERSIONS = {"torch": "2.9.1", "torchvision": "0.24.1", "diffusers": "0.41.0.dev0",
    "transformers": "5.17.0", "accelerate": "1.15.0", "safetensors": "0.8.0",
    "hypergan": "2.0.0", "hndl": "0.10.0", "Pillow": "13.0.0", "numpy": "2.5.4", "huggingface-hub": "1.33.0", "tokenizers": "0.23.3"}
PACKAGES = {"Pillow": "PIL", "huggingface-hub": "huggingface_hub"}


def index(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def require_origin(name, url, commit, subdirectory=None):
    dist = metadata.distribution(name)
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    if (origin.get("url") != url or origin.get("vcs_info", {}).get("commit_id") != commit
            or origin.get("subdirectory") != subdirectory):
        raise ValueError("strict native exact public Git origin differs: " + name)
    return dist


def verify_sources():
    import qwen_image_sliders
    from qwen_image_sliders.core_profile import require_current_core
    require_current_core()
    core = {"commit": CORE, "profile": "official-PG0.9.4-auto"}
    require_origin("particle-sliders-core", "https://github.com/HyperGAN/particle-sliders.git", CORE,
                   "packages/particle-sliders-core")
    base = require_origin("qwen-image-particle-sliders", "https://github.com/HyperGAN/qwen-image-particle-sliders.git", BASE)
    if Path(qwen_image_sliders.__file__).resolve() != Path(base.locate_file("qwen_image_sliders/__init__.py")).resolve():
        raise ValueError("strict native base import is shadowed")
    lock_path = Path(__file__).with_name("source-lock.json")
    lock = json.loads(lock_path.read_text())
    actual_base = index(Path(qwen_image_sliders.__file__).parent)
    own = index(Path(__file__).parent)
    own.pop("source-lock.json")
    if actual_base != lock["base_sources"] or own != lock["product_sources"]:
        raise ValueError("strict native base/product source bytes differ")
    return dict(core=core, base_commit=BASE,
        product_index_sha256=hashlib.sha256(json.dumps(own, sort_keys=True).encode()).hexdigest(),
        source_lock_sha256=digest(lock_path))


def selected_files(dist, package):
    if package == "torch":
        # CPU conditioning executes libtorch_cpu and the Python dispatcher too.
        # Bind all Torch Python/native implementation files, including versioned
        # OpenMP libraries, rather than reusing a CUDA-only library inventory.
        return [f for f in dist.files or () if str(f).startswith('torch/') and
                (str(f).endswith('.py') or str(f).startswith('torch/lib/') or str(f).endswith('.so'))]
    return [f for f in dist.files or () if str(f).startswith((package + "/", package + ".libs/"))
            and (str(f).endswith(".py") or str(f).endswith(".so"))]


def runtime_sources():
    wanted = json.loads(Path(__file__).with_name("runtime-lock.json").read_text())
    result = {}
    for name, version in VERSIONS.items():
        dist = metadata.distribution(name)
        if dist.version.split("+")[0] != version:
            raise ValueError("strict native dependency version differs: " + name)
        package = PACKAGES.get(name, name.replace("-", "_"))
        imported = sys.modules.get(package)
        if imported is not None and Path(imported.__file__).resolve() != Path(dist.locate_file(package + "/__init__.py")).resolve():
            raise ValueError("strict native dependency import is shadowed: " + name)
        files = {str(f): digest(dist.locate_file(f)) for f in selected_files(dist, package)}
        if not files or files != wanted[name]["sources"]:
            raise ValueError("strict native implementation bytes differ: " + name)
        result[name] = dict(version=dist.version, sources_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    require_origin("diffusers", "https://github.com/huggingface/diffusers.git", "1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458")
    require_origin("hypergan", "https://github.com/HyperGAN/HyperGAN.git", "5f368a3770cf5bdae0448995c9f76e29af8049f7")
    import torch
    if torch.__version__ != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise ValueError("strict native execution requires actual Torch2.9.1+cu128")
    return result


def execution_identity(*, conditioning_precision, verify_runtime=True):
    from .execution import policy
    identity = dict(numerical=policy(conditioning_precision=conditioning_precision), sources=verify_sources())
    if verify_runtime:
        identity["runtime"] = runtime_sources()
    return identity
