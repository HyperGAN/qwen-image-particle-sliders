"""Bounded cap/resume correctness check; no downloads or convergence training."""

import argparse
import gc
import hashlib
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import psutil
import torch

import qwen_image_sliders.train as trainer
from qwen_image_sliders.contracts import file_digest
from qwen_image_sliders.runtime import GIB, Runtime

ROWS = [
    {
        "neutral": "A ceramic cup on a wooden table.",
        "positive": "A ceramic cup on a wooden table, lit by warm golden light.",
    },
    {"neutral": "A quiet reading room.", "positive": "A quiet reading room, lit by warm golden light."},
    {"neutral": "A pencil drawing of a tree.", "positive": "A pencil drawing of a tree.", "control": True},
]


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(type(item).__name__).encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())

    visit(value)
    return digest.hexdigest()


def differences(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            yield path
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            yield path + ".keys"
        else:
            for key in left:
                yield from differences(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            yield path + ".length"
        else:
            for i, (a, b) in enumerate(zip(left, right)):
                yield from differences(a, b, f"{path}[{i}]")
    elif left != right:
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true", help="native random CPU fixture, not real CUDA")
    args = parser.parse_args()
    if not args.dummy and (not args.model_dir or not args.device.startswith("cuda:")):
        parser.error("real validation requires an offline --model-dir and explicit --device cuda:N")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if any((args.outdir / name).exists() for name in ("full", "split")):
        parser.error("use a fresh outdir to avoid mixing previous validation artifacts")
    start = time.monotonic()
    report = {
        "status": "starting",
        "real_model": not args.dummy,
        "quality_validated": False,
        "download_bytes": 0,
        "jobs_modified": False,
        "height": 32 if args.dummy else 128,
        "width": 32 if args.dummy else 128,
        "training_updates_total": 8,
        "trajectory_steps": 2,
        "records_per_pair": 1,
        "batch": 1,
        "seed": 7,
        "gradient_checkpointing": True,
        "traces": {},
        "cap_events": [],
        "supervisor_timeout_seconds": 600,
        "process_rss_cap_bytes": 18 * GIB,
        "host_available_floor_bytes": 4 * GIB,
        "process_gpu_cap_bytes": 34 * GIB,
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    peak_rss = 0
    stop = threading.Event()
    process = psutil.Process()

    def write():
        report["elapsed_seconds"] = time.monotonic() - start
        report["process_peak_rss_bytes"] = peak_rss
        (args.outdir / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")

    def monitor():
        nonlocal peak_rss
        while not stop.wait(1):
            peak_rss = max(peak_rss, process.memory_info().rss)
            if (
                peak_rss > 18 * GIB
                or psutil.virtual_memory().available < 4 * GIB
                or time.monotonic() - start > 600
            ):
                report["status"] = "resource-or-time-self-abort"
                write()
                os._exit(75)

    threading.Thread(target=monitor, daemon=True).start()
    phase = "loading"
    original_build, original_step = trainer.build_game, trainer.step_context

    def observed_build(*positional, **kwargs):
        game = original_build(*positional, **kwargs)
        reg = game.regularizer
        assert reg.arm == "b_cap" and reg.coeff == 1 and reg.kappa == 1
        assert reg.lazy_k == 4 and reg.method == "autograd" and reg.target_anneal == "none"
        original_penalty = reg.penalty
        original_norm = reg._grad_norm
        norms = []

        def observed_norm(*args, **kwargs):
            result = original_norm(*args, **kwargs)
            norms.append(result.detach().cpu().tolist())
            return result

        reg._grad_norm = observed_norm

        def observed_penalty(critic, real, fake, *, step, collect_stats):
            norms.clear()
            penalty, stats = original_penalty(critic, real, fake, step=step, collect_stats=True)
            if stats["applied"]:
                event = {
                    "phase": phase,
                    "step": step,
                    **stats,
                    "coefficient": reg.coeff,
                    "effective_lazy_coefficient": reg.coeff * reg.lazy_k,
                    "requires_grad": penalty.requires_grad,
                    "fake_detached": not fake.requires_grad,
                    "input_shape": list(real.shape),
                    "gradient_norms": list(norms),
                    "backward_seen": False,
                }
                assert torch.isfinite(penalty) and penalty.requires_grad

                def seen(gradient):
                    event["backward_seen"] = True
                    event["incoming_gradient"] = float(gradient)
                    return gradient

                penalty.register_hook(seen)
                report["cap_events"].append(event)
            return penalty, stats

        reg.penalty = observed_penalty
        return game

    def observed_step(game, records, index, batch):
        chosen, target = trainer.batch_records(records, index, batch, game.device)
        rng = {"cpu": torch.get_rng_state()}
        if game.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(game.device)
        trace = {
            "step": index,
            "inputs_sha256": fingerprint((chosen, target)),
            "rng_before_sha256": fingerprint(rng),
        }
        trace["stats"] = original_step(game, records, index, batch)
        assert all(torch.isfinite(torch.tensor(v)) for v in trace["stats"].values())
        report["traces"].setdefault(phase, []).append(trace)
        write()
        return trace["stats"]

    trainer.build_game, trainer.step_context = observed_build, observed_step
    try:
        if args.dummy:
            from qwen_image_sliders.tiny import TinyRuntime

            runtime = TinyRuntime()
        else:
            torch.cuda.set_device(args.device)
            free, total = torch.cuda.mem_get_info(args.device)
            report["gpu_free_before_bytes"] = free
            report["host_available_before_bytes"] = psutil.virtual_memory().available
            torch.cuda.set_per_process_memory_fraction(34 * GIB / total, device=args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            runtime = Runtime(args.model_dir, device=args.device)
        options = {
            "seed": 7,
            "height": report["height"],
            "width": report["width"],
            "trajectory_steps": 2,
            "records_per_pair": 1,
            "gradient_checkpointing": True,
        }
        full, split = args.outdir / "full", args.outdir / "split"
        phase = "full"
        trainer.train(runtime, ROWS, full, steps=4, **options)
        gc.collect()
        phase = "split"
        trainer.train(runtime, ROWS, split, steps=2, **options)
        gc.collect()
        phase = "resume"
        trainer.train(runtime, ROWS, split, steps=4, resume=split / "training.pt", **options)
        gc.collect()
        a = torch.load(full / "training.pt", map_location="cpu", weights_only=True)
        b = torch.load(split / "training.pt", map_location="cpu", weights_only=True)
        report["state_mismatches"] = list(differences(a, b))
        report["all_checkpoint_fields_exact"] = not report["state_mismatches"]
        report["adapter_sha256"] = {
            "full": file_digest(full / "adapter.safetensors"),
            "resumed": file_digest(split / "adapter.safetensors"),
        }
        report["ema_safetensors_bytes_exact"] = len(set(report["adapter_sha256"].values())) == 1
        report["matched_rng_inputs_and_stats"] = (
            report["traces"]["full"] == report["traces"]["split"] + report["traces"]["resume"]
        )
        report["native_cap_backward_passed"] = len(report["cap_events"]) == 2 and all(
            e["step"] == 4 and e["backward_seen"] and e["incoming_gradient"] == 1
            for e in report["cap_events"]
        )
        assert report["all_checkpoint_fields_exact"], report["state_mismatches"][:20]
        assert report["ema_safetensors_bytes_exact"] and report["matched_rng_inputs_and_stats"]
        assert report["native_cap_backward_passed"]
        report["status"] = "success"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        trainer.build_game, trainer.step_context = original_build, original_step
        stop.set()
        if not args.dummy and torch.cuda.is_initialized():
            report["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        write()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
