"""Bounded paired velocities on native QwenImage21Pipeline neutral trajectories."""

import json
from importlib.resources import files
from pathlib import Path

import torch
from particle_sliders.qwen_native import with_encoding

from .contracts import validate_shape
from .projection import predict


def load_prompts(path=None):
    source = Path(path).read_text() if path else files(__package__).joinpath("prompts.json").read_text()
    data = json.loads(source)
    rows = data.get("rows", [])
    if len(rows) < 2:
        raise ValueError("at least two paired prompt rows are required")
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("neutral", "positive")):
            raise ValueError("every row needs nonempty neutral and positive text")
    controls = data.get("controls", [])
    if not all(isinstance(p, str) and p.strip() for p in controls):
        raise ValueError("controls must be nonempty prompt strings")
    return rows + [{"neutral": p, "positive": p, "control": True} for p in controls]


def cpu_kwargs(kwargs):
    return {
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
        for k, v in kwargs.items()
        if k not in {"return_dict", "kv_cache", "kv_cache_mode", "attention_kwargs"}
    }


@torch.no_grad()
def paired_records(runtime, rows, *, seed, height, width, trajectory_steps=40, records_per_pair=2):
    validate_shape(height, width)
    if not 1 <= records_per_pair <= trajectory_steps:
        raise ValueError("records-per-pair must be between one and trajectory-steps")
    if len(rows) * records_per_pair > 128:
        raise ValueError("initial product limits the teacher cache to 128 records; reduce the prompt budget")
    selected = set(torch.linspace(0, trajectory_steps - 1, records_per_pair).round().int().tolist())
    records = []
    for row in rows:
        neutral, positive = runtime.encode(row["neutral"]), runtime.encode(row["positive"])
        pending = []
        count = 0

        def capture(module, args, kwargs, output, _pending=pending):
            nonlocal count
            if kwargs.get("kv_cache") is not None or kwargs.get("kv_cache_mode") is not None:
                raise ValueError("paired training requires uncached native forwards")
            if count in selected:
                _pending.append(
                    (
                        cpu_kwargs(kwargs),
                        output[0][:, -kwargs["hidden_states"].shape[1] :].float().cpu().clone(),
                        runtime.pipe.scheduler.timesteps[count].detach().float().cpu().clone(),
                    )
                )
            count += 1

        handle = runtime.model.register_forward_hook(capture, with_kwargs=True)
        try:
            # The actual pipeline owns packing, dynamic timestep shift, t/1000, and Euler stepping.
            runtime.sample(neutral, seed=seed, height=height, width=width, steps=trajectory_steps)
        finally:
            handle.remove()
        if count != trajectory_steps or len(pending) != records_per_pair:
            raise RuntimeError("native pipeline trajectory contract changed")
        for kwargs_cpu, v_neutral, raw_timestep in pending:
            kwargs = {
                k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs_cpu.items()
            }
            v_positive = predict(runtime.model, with_encoding(kwargs, positive))
            records.append(
                {
                    "kwargs": kwargs_cpu,
                    "neutral": v_neutral,
                    "target": v_positive.cpu(),
                    "control": bool(row.get("control", False)),
                    "raw_timestep": raw_timestep,
                    "neutral_condition": cpu_kwargs(neutral),
                    "positive_condition": cpu_kwargs(positive),
                }
            )
    return records


def batch_records(records, index, batch, device):
    chosen = [records[((index - 1) * batch + i) % len(records)] for i in range(batch)]
    # Adding text padding changes image RoPE positions even if the keys are masked.
    # Preserve the exact captured prefix length for each native model forward.
    result = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in row["kwargs"].items()}
        for row in chosen
    ]
    return result, torch.cat([row["target"] for row in chosen]).to(device)
