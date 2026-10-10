"""Train routed Qwen-Image 2.1 adapters with the pinned shared FormulationGame."""

import argparse
import json
from pathlib import Path

import torch
from particle_sliders import FormulationGame
from torch import nn

from .adapters import AdapterBank
from .checkpoint import metadata, save_adapter
from .contracts import canonical, formulation, validate_shape
from .projection import QwenErrorBridge, VelocityProjection, isotropic_calibration_bank
from .teachers import batch_records, load_prompts, paired_records


def build_game(runtime, records, stamp, declared, seed):
    torch.manual_seed(seed)
    rank = int(stamp.spec["adapter_rank"])
    edits = []
    for row in records:
        delta = row["target"] - row["neutral"]
        if not torch.isfinite(delta).all():
            raise ValueError("non-finite paired teacher velocity")
        rms = delta.double().square().mean().sqrt()
        if not row["control"] and float(rms) > 1e-8:
            edits.append(rms)
    targets = isotropic_calibration_bank(torch.stack(edits) if edits else [], rank).to(runtime.device)
    bank = AdapterBank(runtime.model, stamp).to(runtime.device)
    try:
        bridge = QwenErrorBridge(runtime.model, bank, VelocityProjection(rank))
        particles = nn.Parameter(
            torch.randn(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]), device=runtime.device)
            * 0.02
        )
        game = FormulationGame(
            stamp,
            declared,
            seed=seed,
            device=runtime.device,
            bridge=bridge,
            particles=particles,
            critic_targets=targets,
            critic_neutrals=torch.zeros_like(targets),
        )
    except Exception:
        bank.close()
        raise
    game.normalization_records = len(edits)
    return game


def step_context(game, records, index, batch):
    kwargs, target = batch_records(records, index, batch, game.device)
    game.bridge.set_context(kwargs, target, game.critic.target_std.detach(), index)
    features = torch.zeros(batch, game.rank, device=game.device)
    # Keep the hooks active through backward, including native checkpoint recomputation.
    with game.bridge.bank.active(game.particles):
        return game.step(index, features)


def train(
    runtime,
    rows,
    outdir,
    *,
    steps=4,
    seed=7,
    batch=1,
    height=32,
    width=32,
    trajectory_steps=4,
    records_per_pair=2,
    g_lr=None,
    resume=None,
    gradient_checkpointing=False,
):
    if steps < 1:
        raise ValueError("steps must be positive")
    stamp, declared = formulation(batch=batch, g_lr=g_lr)
    records = paired_records(
        runtime,
        rows,
        seed=seed,
        height=height,
        width=width,
        trajectory_steps=trajectory_steps,
        records_per_pair=records_per_pair,
    )
    if gradient_checkpointing:
        runtime.model.enable_gradient_checkpointing()
    game = build_game(runtime, records, stamp, declared, seed)
    bank = game.bridge.bank
    try:
        meta = metadata(runtime, bank, stamp, declared)
        run_config = json.loads(
            canonical(
                {
                    "rows": rows,
                    "seed": seed,
                    "batch": batch,
                    "height": height,
                    "width": width,
                    "trajectory_steps": trajectory_steps,
                    "records_per_pair": records_per_pair,
                    "gradient_checkpointing": gradient_checkpointing,
                    "device_type": runtime.device.type,
                    "base_dtype": str(runtime.model.dtype),
                }
            )
        )
        ema = {k: v.detach().clone() for k, v in bank.state_dict().items()}
        ema_particles = game.particles.detach().clone()
        start = 1
        if resume:
            state = torch.load(resume, map_location=game.device, weights_only=True)
            if state["metadata"] != meta or state["run_config"] != run_config:
                raise ValueError("resume base, formulation, prompts, or trajectory configuration differs")
            game.bridge.load_state_dict(state["bridge"], strict=True)
            game.critic.load_state_dict(state["critic"], strict=True)
            with torch.no_grad():
                game.particles.copy_(state["particles"])
            game.opt_g.load_state_dict(state["opt_g"])
            game.opt_d.load_state_dict(state["opt_d"])
            torch.set_rng_state(state["rng"].cpu())
            if game.device.type == "cuda":
                if state["cuda_rng"] is None:
                    raise ValueError("exact CUDA resume requires a CUDA checkpoint")
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=game.device)
            ema, ema_particles = state["ema"], state["ema_particles"]
            start = int(state["step"]) + 1
            if steps < start:
                raise ValueError("resume steps must exceed the completed checkpoint step")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        decay = float(stamp.spec["ema"])
        for index in range(start, steps + 1):
            stats = step_context(game, records, index, batch)
            with torch.no_grad():
                for k, value in bank.state_dict().items():
                    ema[k].lerp_(value, 1 - decay)
                ema_particles.lerp_(game.particles, 1 - decay)
            with (outdir / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(stats) + "\n")
            print(json.dumps(stats), flush=True)
        torch.save(
            {
                "step": steps,
                "metadata": meta,
                "run_config": run_config,
                "bridge": game.bridge.state_dict(),
                "critic": game.critic.state_dict(),
                "particles": game.particles.detach(),
                "opt_g": game.opt_g.state_dict(),
                "opt_d": game.opt_d.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(game.device) if game.device.type == "cuda" else None,
                "ema": ema,
                "ema_particles": ema_particles,
            },
            outdir / "training.pt",
        )
        exported = {
            **meta,
            "training": {
                **run_config,
                "steps": steps,
                "export": "EMA",
                "ema": decay,
                "normalization_records": game.normalization_records,
                "controls_excluded_from_normalization": True,
            },
        }
        save_adapter(outdir / "adapter.safetensors", ema, ema_particles, exported)
        (outdir / "run.json").write_text(json.dumps({"metadata": exported, "last": stats}, indent=2) + "\n")
        return exported
    finally:
        bank.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dummy", action="store_true", help="tiny native random model, synthetic text, CPU only"
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--allow-hub", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--g-lr", type=float)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--trajectory-steps", type=int, default=40)
    parser.add_argument("--records-per-pair", type=int, default=2)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/warm-light"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    rows = load_prompts(args.prompts)
    if args.steps < 1:
        parser.error("steps must be positive")
    formulation(batch=args.batch, g_lr=args.g_lr)
    if not args.dummy:
        validate_shape(args.height, args.width)
    trajectory_steps = 4 if args.dummy else args.trajectory_steps
    if not 1 <= args.records_per_pair <= trajectory_steps or len(rows) * args.records_per_pair > 128:
        parser.error("invalid teacher record budget")
    if args.dummy:
        if args.model_dir or args.allow_hub or args.device != "cpu":
            parser.error("--dummy uses only CPU and accepts no model download options")
        from .tiny import TinyRuntime

        runtime = TinyRuntime(args.seed)
        args.steps = min(args.steps, 4)
        args.height = args.width = 32
        args.trajectory_steps = 4
    else:
        from .runtime import Runtime

        runtime = Runtime(
            args.model_dir, allow_hub=args.allow_hub, cache_dir=args.cache_dir, device=args.device
        )
    train(
        runtime,
        rows,
        args.outdir,
        steps=args.steps,
        seed=args.seed,
        batch=args.batch,
        g_lr=args.g_lr,
        height=args.height,
        width=args.width,
        trajectory_steps=args.trajectory_steps,
        records_per_pair=args.records_per_pair,
        resume=args.resume,
        gradient_checkpointing=args.gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
