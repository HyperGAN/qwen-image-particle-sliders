"""Train native Qwen-Image 2.1 responses with the installed ParticleGAN 0.9 API."""

import argparse
import json
from pathlib import Path

import torch
from particle_sliders.lifecycle import advance, EMA, atomic_save
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09_images import QwenImage21ResponseAdapter, image_condition_features
from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic

from .adapters import AdapterBank
from .checkpoint import metadata, save_adapter
from .contracts import canonical, formulation, validate_shape
from .teachers import load_prompts, paired_records


def _condition(encoding):
    return {
        "embeds": encoding["prompt_embeds"],
        "mask": encoding["prompt_embeds_mask"],
        "image_pad_mask": encoding["image_pad_mask"],
    }


def response_rows(runtime, records):
    """Recover native spatial fields and exact raw times from pipeline captures."""
    from diffusers import QwenImage21Pipeline

    result = []
    for record in records:
        kwargs = record["kwargs"]
        frames, height, width = kwargs["img_shapes"][0][-1]
        if frames != 1:
            raise ValueError("only native text-to-image spatial fields are supported")
        latent = QwenImage21Pipeline._unpack_latents(kwargs["hidden_states"], height * 8, width * 8, 8)
        neutral = _condition(record["neutral_condition"])
        positive = _condition(record["positive_condition"])
        result.append(
            ResponseRow(
                neutral,
                positive,
                neutral,
                latent.to(runtime.device),
                timestep=record["raw_timestep"].to(runtime.device),
            )
        )
    return result


def build_game(
    runtime,
    records,
    stamp,
    declared,
    seed,
    *,
    steps=4,
    preset="bispf_optimized",
    g_lr=None,
    critic_kind="readout",
):
    """Bind the real hooked bank once; losses, sampling, optimizers and state are shared."""
    torch.manual_seed(seed)
    edits = []
    for record in records:
        delta = record["target"] - record["neutral"]
        if not torch.isfinite(delta).all():
            raise ValueError("non-finite paired teacher velocity")
        rms = float(delta.double().square().mean().sqrt())
        if not record["control"] and rms > 1e-8:
            edits.append(rms)
    if len(edits) < 2:
        raise ValueError("calibration requires at least two nondegenerate noncontrol edits")
    response_scale = float(torch.tensor(edits, dtype=torch.float64).median())
    bank = AdapterBank(runtime.model, stamp).to(runtime.device)
    try:
        particles = torch.nn.Parameter(
            torch.randn(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]), device=runtime.device)
            * 0.02
        )
        adapter = QwenImage21ResponseAdapter(runtime, bank, particles, model_identity=runtime.base_identity)
        prepared = [adapter.prepare(row) for row in response_rows(runtime, records)]
        conditions = image_condition_features(prepared)
        if critic_kind == "hndl":
            from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic

            critic_class = HNDLConditionalReadoutCritic
        elif critic_kind == "readout":
            critic_class = ConditionalReadoutCritic
        else:
            raise ValueError("unsupported critic architecture")
        critic = critic_class(prepared[0].target.numel(), conditions.shape[1]).to(runtime.device)
        game = ConditionalPG09(
            adapter,
            prepared,
            critic,
            conditions=conditions,
            particles=particles,
            response_domain="continuous_field",
            response_scale=response_scale,
            preset=preset,
            total_steps=steps,
            batch_size=int(declared["adv_batch"]),
            seed=seed,
            recipe_overrides={"lr": g_lr} if g_lr is not None else None,
            sampling="balanced",
        )
        game.normalization_records = len(edits)
        return game
    except Exception:
        bank.close()
        raise


def step_context(game):
    """The common game owns both phase samplers and each native response forward."""
    return advance(game)


def _validate_ema(state, bank, particles):
    expected = bank.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("resume EMA keys differ")
    for key, value in {**state, "particles": particles}.items():
        wanted = expected.get(key)
        if key == "particles":
            wanted = bank.layers[0].up.weight.new_empty(bank.particle_shape)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != wanted.shape
            or value.dtype != wanted.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("resume EMA tensor schema or values differ")


def train(
    runtime,
    rows,
    outdir,
    *,
    steps=4,
    seed=7,
    batch=2,
    height=32,
    width=32,
    trajectory_steps=4,
    records_per_pair=2,
    g_lr=None,
    resume=None,
    gradient_checkpointing=False,
    preset="bispf_optimized",
    stop_after=None,
    critic_kind="readout",
):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be positive")
    if type(batch) is not int or not 2 <= batch <= 512:
        raise ValueError("batch must be an integer in [2,512]")
    if stop_after is None:
        stop_after = steps
    if type(stop_after) is not int or not 1 <= stop_after <= steps:
        raise ValueError("stop-after must be between one and the fixed steps budget")
    if preset not in ("bispf", "bispf_optimized"):
        raise ValueError("unsupported PG preset")
    if critic_kind not in ("readout", "hndl"):
        raise ValueError("unsupported critic architecture")
    if type(seed) is not int or not 0 <= seed < 2**63 - 100:
        raise ValueError("invalid seed")
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
    checkpoint_attributes = ("gradient_checkpointing", "_gradient_checkpointing_func")
    checkpoint_state = [
        (module, {key: getattr(module, key) for key in checkpoint_attributes if hasattr(module, key)})
        for module in runtime.model.modules()
    ]
    if gradient_checkpointing:
        runtime.model.enable_gradient_checkpointing()
    game = None
    try:
        game = build_game(
            runtime,
            records,
            stamp,
            declared,
            seed,
            steps=steps,
            preset=preset,
            g_lr=g_lr,
            critic_kind=critic_kind,
        )
        bank = game.backend.adapter
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
                    "steps": steps,
                    "preset": preset,
                    "critic_kind": critic_kind,
                }
            )
        )
        decay = float(stamp.spec["ema"])
        cloud = torch.nn.ParameterDict({'particles': game.backend.particles})
        bank_ema, cloud_ema = EMA(bank, decay), EMA(cloud, decay)
        ema, ema_particles = bank_ema.values, cloud_ema.values['particles']
        if resume:
            state = torch.load(resume, map_location=game.device, weights_only=True)
            if (
                state.get("schema") != "qwen-native-pg09-training-v2"
                or state.get("metadata") != meta
                or state.get("run_config") != run_config
            ):
                raise ValueError("resume base, objective, source, prompts or fixed training budget differs")
            _validate_ema(state["ema"], bank, state["ema_particles"])
            if state["game"]["completed_steps"] >= stop_after:
                raise ValueError("resume stop-after must exceed the completed checkpoint step")
            game.load_state_dict(state["game"])
            bank_ema.load_state_dict(bank, dict(decay=decay, values=state['ema']))
            cloud_ema.load_state_dict(cloud, dict(decay=decay, values={'particles': state['ema_particles']}))
            ema, ema_particles = bank_ema.values, cloud_ema.values['particles']
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stats = None
        while game.completed_steps < stop_after:
            stats = step_context(game)
            bank_ema.update(bank)
            cloud_ema.update(cloud)
            with (outdir / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(stats) + "\n")
            print(json.dumps(stats), flush=True)
        atomic_save(outdir / "training.pt",
            {
                "schema": "qwen-native-pg09-training-v2",
                "metadata": meta,
                "run_config": run_config,
                "game": game.state_dict(),
                "ema": ema,
                "ema_particles": ema_particles,
            },
        )
        exported = {
            **meta,
            "training": {
                **run_config,
                "completed_steps": game.completed_steps,
                "export": "EMA",
                "ema": decay,
                "normalization_records": game.normalization_records,
                "controls_excluded_from_normalization": True,
                "response_scale": game.response_scale,
                "recipe": game.state_dict()["config"]["recipe"],
            },
        }
        save_adapter(outdir / "adapter.safetensors", ema, ema_particles, exported)
        (outdir / "run.json").write_text(json.dumps({"metadata": exported, "last": stats}, indent=2) + "\n")
        return exported
    finally:
        if game is not None:
            game.backend.adapter.close()
        for module, previous in checkpoint_state:
            for key in checkpoint_attributes:
                if key in previous:
                    setattr(module, key, previous[key])
                elif hasattr(module, key):
                    delattr(module, key)


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
    parser.add_argument("--steps", type=int, default=1600, help="fixed PG recipe budget, retained on resume")
    parser.add_argument("--stop-after", type=int, help="checkpoint after this many total updates")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--preset", choices=("bispf", "bispf_optimized"), default="bispf_optimized")
    parser.add_argument("--critic", choices=("readout", "hndl"), default="readout")
    parser.add_argument(
        "--g-lr", type=float, help="PG base learning rate, including derived critic/prior rates"
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--trajectory-steps", type=int, default=40)
    parser.add_argument("--records-per-pair", type=int, default=2)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/warm-light"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args(argv)
    rows = load_prompts(args.prompts)
    if args.steps < 1 or not 2 <= args.batch <= 512:
        parser.error("steps must be positive and batch must be in [2,512]")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("stop-after must be between one and the fixed steps budget")
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
        if args.stop_after is not None:
            args.stop_after = min(args.stop_after, args.steps)
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
        preset=args.preset,
        stop_after=args.stop_after,
        critic_kind=args.critic,
    )


if __name__ == "__main__":
    main()
