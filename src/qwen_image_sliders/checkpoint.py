"""Strict safe inference exports and independently reconstructed compatibility metadata."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .contracts import FORMAT, PROTOCOL, canonical, digest, formulation_record, lock


def metadata(runtime, bank, stamp, declared):
    pins = lock()
    # Diffusers injects local _name_or_path and bookkeeping fields after loading.
    # Only semantic configuration belongs in a portable architecture identity.
    model_config = {k: v for k, v in runtime.model.config.items() if not k.startswith("_")}
    scheduler_config = {k: v for k, v in runtime.pipe.scheduler.config.items() if not k.startswith("_")}
    return {
        "format": FORMAT,
        "base": runtime.base_identity,
        "targets": bank.names,
        "model_config_sha256": digest(model_config),
        "scheduler_config_sha256": digest(scheduler_config),
        "core_commit": pins["core"]["commit"],
        "diffusers_commit": pins["diffusers"]["commit"],
        "formulation": formulation_record(stamp, declared),
        "protocol": PROTOCOL,
        "validation_kind": runtime.validation_kind,
        "quality_validated": False,
        "attribution": "Built with Qwen",
    }


def read_metadata(path):
    with safe_open(str(path), framework="pt") as handle:
        raw = (handle.metadata() or {}).get("qwen_image_sliders")
    if not raw:
        raise ValueError("missing particle slider metadata")
    meta = json.loads(raw)
    if meta.get("format") != FORMAT:
        raise ValueError("unsupported particle slider format")
    return meta


def save_adapter(path, bank_state, particles, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"bank." + k: v.detach().float().cpu().contiguous() for k, v in bank_state.items()}
    tensors["particles"] = particles.detach().float().cpu().contiguous()
    save_file(tensors, str(path), metadata={"qwen_image_sliders": canonical(meta)})
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def load_adapter(path, bank, particles, expected):
    meta = read_metadata(path)
    for key, value in expected.items():
        if meta.get(key) != json.loads(canonical(value)):
            raise ValueError(f"adapter {key} mismatch")
    state = load_file(str(path))
    wanted = {"bank." + k: v for k, v in bank.state_dict().items()}
    wanted["particles"] = particles
    if state.keys() != wanted.keys():
        raise ValueError("adapter tensor keys differ")
    for key, tensor in state.items():
        if (
            tensor.shape != wanted[key].shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid adapter tensor {key}")
    bank.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("bank.")}, strict=True)
    with torch.no_grad():
        particles.copy_(state["particles"].to(particles.device))
    return meta
