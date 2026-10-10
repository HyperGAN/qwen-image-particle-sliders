"""Offline-first, verified official snapshot loading and read-only resource checks."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from .contracts import file_digest, lock, validate_shape

GIB = 2**30
# Conservative policy for this initial unquantized loader; not a measured fit guarantee.
MIN_HOST_BYTES = 40 * GIB
MIN_GPU_BYTES = 38 * GIB
MIN_DOWNLOAD_BYTES = 45 * GIB


def preflight(cache_dir=None, device="cpu"):
    info = lock()["model"]
    result = {
        "model_id": info["repository"],
        "revision": info["revision"],
        "weight_bytes": info["weight_bytes"],
        "cuda_visible": torch.cuda.is_available(),
        "policy_host_bytes": MIN_HOST_BYTES if str(device).startswith("cuda") else 80 * GIB,
        "policy_gpu_bytes": MIN_GPU_BYTES,
        "policy_download_free_bytes": MIN_DOWNLOAD_BYTES,
        "policy_is_unmeasured": True,
    }
    try:
        import psutil

        result["host_available_bytes"] = psutil.virtual_memory().available
    except ImportError:
        available = next(
            line
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        result["host_available_bytes"] = int(available.split()[1]) * 1024
    if cache_dir:
        existing = Path(cache_dir).resolve()
        while not existing.exists():
            existing = existing.parent
        result["disk_free_bytes"] = shutil.disk_usage(existing).free
    if str(device).startswith("cuda") and result["cuda_visible"]:
        with torch.cuda.device(device):
            result["gpu_free_bytes"], result["gpu_total_bytes"] = torch.cuda.mem_get_info()
    return result


def require_resources(report, *, download=False, device="cpu"):
    required_host = MIN_HOST_BYTES if str(device).startswith("cuda") else 80 * GIB
    if report["host_available_bytes"] < required_host:
        raise RuntimeError(
            f"full-model load blocked: need {required_host // GIB} GiB available host RAM by the unmeasured "
            "initial loader policy; use --dummy or stage resources first"
        )
    if str(device).startswith("cuda"):
        if not report["cuda_visible"]:
            raise RuntimeError("CUDA is not visible in this execution environment")
        if report.get("gpu_free_bytes", 0) < MIN_GPU_BYTES:
            raise RuntimeError("full-model run blocked: need 38 GiB free on the selected GPU by policy")
    if download and report.get("disk_free_bytes", 0) < MIN_DOWNLOAD_BYTES:
        raise RuntimeError("snapshot download blocked: need 45 GiB free storage by policy")


def verify_snapshot(directory):
    directory = Path(directory)
    manifest = lock()["model"]["files"]
    for name, spec in manifest.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            raise ValueError(f"missing or wrong-size official snapshot file: {name}")
        if file_digest(path) != spec["sha256"]:
            raise ValueError(f"official snapshot hash mismatch: {name}")
    # Reject extra loadable files: from_pretrained must not select an unverified variant.
    for path in directory.rglob("*"):
        name = path.relative_to(directory).as_posix()
        if (
            path.is_file()
            and path.suffix in {".safetensors", ".bin", ".pt", ".json"}
            and name not in manifest
            and not name.startswith(".cache/")
        ):
            raise ValueError(f"unverified loadable file in snapshot: {name}")
    return directory


class Runtime:
    """One-device unquantized pipeline; no automatic downloads or shared-GPU selection."""

    validation_kind = "official-weight-runtime"

    def __init__(self, model_dir=None, *, allow_hub=False, cache_dir=None, device="cpu"):
        if not model_dir and not (allow_hub and cache_dir):
            raise ValueError(
                "provide --model-dir, or explicitly permit a pinned download with --allow-hub --cache-dir"
            )
        report = preflight(cache_dir, device)
        require_resources(report, download=not model_dir, device=device)
        spec = lock()["model"]
        if not model_dir:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                spec["repository"],
                revision=spec["revision"],
                cache_dir=str(cache_dir),
                allow_patterns=list(spec["files"]),
            )
        directory = verify_snapshot(model_dir)
        from diffusers import QwenImage21Pipeline

        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.pipe = QwenImage21Pipeline.from_pretrained(
            str(directory),
            torch_dtype=dtype,
            local_files_only=True,
        ).to(self.device)
        if self.pipe.config.sample_sigmas is not None:
            raise ValueError("this backend targets the base 2.1 checkpoint, not a Turbo sampling schedule")
        self.pipe.set_progress_bar_config(disable=True)
        for component in (self.pipe.transformer, self.pipe.text_encoder, self.pipe.vae):
            component.eval().requires_grad_(False)
        self.base_identity = {
            "model_id": spec["repository"],
            "revision": spec["revision"],
            "manifest_sha256": file_digest(Path(__file__).with_name("backend.lock.json")),
        }

    @property
    def model(self):
        return self.pipe.transformer

    @torch.no_grad()
    def encode(self, prompt):
        embeds, mask, image_mask = self.pipe.encode_prompt(prompt=prompt, device=self.device)
        if image_mask.any():
            raise ValueError("only text-to-image prompt encoding is supported")
        return {"prompt_embeds": embeds, "prompt_embeds_mask": mask, "image_pad_mask": image_mask}

    def sample(self, encoding, *, seed, height, width, steps, output_type="latent"):
        validate_shape(height, width)
        return self.pipe(
            prompt_embeds=encoding["prompt_embeds"],
            prompt_embeds_mask=encoding["prompt_embeds_mask"],
            height=height,
            width=width,
            num_inference_steps=steps,
            true_cfg_scale=1.0,
            use_kv_cache=False,
            output_type=output_type,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.cache_dir, args.device), indent=2))
