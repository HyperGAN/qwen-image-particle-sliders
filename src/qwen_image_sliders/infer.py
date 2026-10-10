"""Load a strict EMA particle export and sample with the native QwenImage21Pipeline."""

import argparse
from pathlib import Path

import torch

from .adapters import AdapterBank
from .checkpoint import load_adapter, metadata, read_metadata
from .contracts import PROTOCOL, formulation, validate_shape, validate_strength


def infer(
    runtime, adapter, prompt, *, strength=1.0, seed=42, height=512, width=512, steps=40, output_type="pil"
):
    validate_strength(strength)
    validate_shape(height, width)
    if steps < 1:
        raise ValueError("steps must be positive")
    meta = read_metadata(adapter)
    if meta["protocol"] != PROTOCOL:
        raise ValueError("adapter protocol differs from this backend")
    stamp, declared = formulation(declared=meta["formulation"]["declared"])
    bank = AdapterBank(runtime.model, stamp).to(runtime.device)
    particles = torch.empty(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]), device=runtime.device)
    try:
        load_adapter(adapter, bank, particles, metadata(runtime, bank, stamp, declared))
        encoding = runtime.encode(prompt)
        with bank.active(particles, strength), torch.no_grad():
            return runtime.sample(
                encoding, seed=seed, height=height, width=width, steps=steps, output_type=output_type
            )
    finally:
        bank.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dummy", action="store_true", help="verify synthetic latent inference, saving a .pt file"
    )
    args = parser.parse_args(argv)
    validate_strength(args.strength)
    if args.steps < 1:
        parser.error("steps must be positive")
    if not args.dummy:
        validate_shape(args.height, args.width)
        if args.out.suffix.lower() != ".png":
            parser.error("use .png to preserve the native RGBA output")
    # Read the protocol before allocating a model or permitting a download.
    meta = read_metadata(args.adapter)
    if meta["protocol"] != PROTOCOL:
        parser.error("adapter protocol differs from the supported text-to-image protocol")
    if args.dummy:
        if args.model_dir or args.allow_hub or args.device != "cpu" or args.out.suffix != ".pt":
            parser.error("dummy inference uses CPU only and requires an output .pt file")
        from .tiny import TinyRuntime

        runtime = TinyRuntime(meta["base"]["fixture_seed"])
        args.height = args.width = 32
        args.steps = min(args.steps, 4)
    else:
        if meta["validation_kind"] != "official-weight-runtime":
            parser.error("synthetic adapters cannot be loaded into the real model")
        from .runtime import Runtime

        runtime = Runtime(
            args.model_dir, allow_hub=args.allow_hub, cache_dir=args.cache_dir, device=args.device
        )
    images = infer(
        runtime,
        args.adapter,
        args.prompt,
        strength=args.strength,
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=args.steps,
        output_type="latent" if args.dummy else "pil",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.dummy:
        torch.save({"latents": images, "validation_kind": runtime.validation_kind}, args.out)
    else:
        images[0].save(args.out)
    print(str(args.out))


if __name__ == "__main__":
    main()
