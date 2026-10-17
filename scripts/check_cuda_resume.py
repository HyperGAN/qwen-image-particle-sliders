"""Offline native CPU PG0.9.1 resume proof (historical command name retained)."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwen_image_sliders.contracts import file_digest
from qwen_image_sliders.teachers import load_prompts
from qwen_image_sliders.tiny import TinyRuntime
from qwen_image_sliders.train import train


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())

    visit(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", required=True, help="CPU native fixture only")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--preset", choices=("bispf", "bispf_optimized"), default="bispf_optimized")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    full, split = args.outdir / "full", args.outdir / "split"
    if full.exists() or split.exists():
        parser.error("use a fresh output directory")
    options = {"steps": 4, "preset": args.preset, "gradient_checkpointing": True}
    rows = load_prompts()
    train(TinyRuntime(), rows, full, **options)
    train(TinyRuntime(), rows, split, stop_after=2, **options)
    train(TinyRuntime(), rows, split, resume=split / "training.pt", **options)
    hashes = {
        name: fingerprint(torch.load(path / "training.pt", map_location="cpu", weights_only=True))
        for name, path in (("full", full), ("resumed", split))
    }
    adapter_hashes = {
        name: file_digest(path / "adapter.safetensors") for name, path in (("full", full), ("resumed", split))
    }
    assert len(set(hashes.values())) == len(set(adapter_hashes.values())) == 1
    assert not torch.cuda.is_initialized()
    report = {
        "schema": "qwen-native-pg091-cpu-resume-v2",
        "status": "passed",
        "preset": args.preset,
        "steps": 4,
        "split_at": 2,
        "checkpoint_state_sha256": hashes,
        "adapter_sha256": adapter_hashes,
        "cuda_initialized": False,
        "quality_validated": False,
    }
    (args.outdir / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
