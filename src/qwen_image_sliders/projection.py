"""All-coordinate latent error sketches, adapted from the Apache-2.0 Iris product.

Modified for packed Qwen-Image 2.1 latents. The game remains in the shared core.
"""

import math

import torch
from torch import nn


def isotropic_calibration_bank(edit_rms, rank):
    values = torch.as_tensor(edit_rms, dtype=torch.float64)
    if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError("calibration requires at least two finite positive edit RMS values")
    axes = torch.eye(rank, dtype=torch.float64, device=values.device) * math.sqrt(rank)
    bank = (values[:, None, None] * torch.cat([axes, -axes])[None]).flatten(0, 1).float()
    if not torch.isfinite(bank).all():
        raise ValueError("calibration moments overflow FP32")
    return bank


class VelocityProjection(nn.Module):
    def __init__(self, rank, seed=31415):
        super().__init__()
        self.rank = int(rank)
        self.register_buffer("base_seed", torch.tensor(seed, dtype=torch.int64))

    def seed_for_step(self, index):
        if index < 1:
            raise ValueError("steps start at one")
        return int(self.base_seed.item()) + 1000003 * int(index)

    def forward(self, velocity, *, seed):
        flat = velocity.float().flatten(1)
        generator = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (flat.shape[1], self.rank), dtype=torch.int8, generator=generator)
        matrix = signs.to(device=flat.device, dtype=torch.float32).mul_(2).sub_(1)
        return flat @ matrix / math.sqrt(flat.shape[1])


def predict(model, kwargs):
    if isinstance(kwargs, list):
        # 2.1 RoPE ties image positions to prefix length. Per-row forwards preserve native layouts.
        return torch.cat([predict(model, row) for row in kwargs])
    # The native model emits the joint text/image sequence; the pipeline takes its target suffix.
    tokens = math.prod(kwargs["img_shapes"][0][-1])
    return model(**kwargs, return_dict=False)[0][:, -tokens:].float()


class QwenErrorBridge(nn.Module):
    def __init__(self, model, bank, projector):
        super().__init__()
        object.__setattr__(self, "model", model)
        self.bank = bank
        self.projector = projector
        self.context = None

    def set_context(self, kwargs, target, scale, index):
        self.context = kwargs, target, scale, self.projector.seed_for_step(index)

    def forward(self, features, particles):
        if self.context is None:
            raise RuntimeError("paired Qwen context has not been set")
        kwargs, target, scale, seed = self.context
        if features.shape[0] != target.shape[0]:
            raise ValueError("batch does not match paired context")
        with self.bank.active(particles):
            student = predict(self.model, kwargs)
        return self.projector(student - target, seed=seed) / scale
