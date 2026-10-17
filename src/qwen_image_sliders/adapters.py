"""2.1 attention-output hooks. Adapted from HyperGAN Iris's Apache-2.0 product hooks.

Modified for QwenImage21Transformer2DModel's single-stream projection contract.
The nonlinear routed map is supplied by particle-sliders-core.
"""

from contextlib import contextmanager

import torch
from torch import nn

from .contracts import validate_strength


def target_modules(model):
    from diffusers import QwenImage21Transformer2DModel

    if not isinstance(model, QwenImage21Transformer2DModel):
        raise TypeError("requires QwenImage21Transformer2DModel, not an older Qwen transformer")
    modules = dict(model.named_modules())
    names = [f"transformer_blocks.{i}.attn.to_out.0" for i in range(model.config.num_layers)]
    if not names or any(not isinstance(modules.get(name), nn.Linear) for name in names):
        raise ValueError("Qwen-Image 2.1 attention output projection contract changed")
    return [(name, modules[name]) for name in names]


class RoutedProjection(nn.Module):
    def __init__(self, base, stamp):
        super().__init__()
        rank = int(stamp.spec["adapter_rank"])
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.route = stamp.bridge()
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, particles):
        return self.up(self.route(self.down(x.to(self.down.weight.dtype)), particles))


class AdapterBank(nn.Module):
    def __init__(self, model, stamp):
        super().__init__()
        targets = target_modules(model)
        model.requires_grad_(False)
        self.names = [name for name, _ in targets]
        self.layers = nn.ModuleList([RoutedProjection(base, stamp) for _, base in targets])
        self.strength = 0.0
        self.particle_shape = (int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]))
        object.__setattr__(self, "_particles", None)
        self.handles = [base.register_forward_hook(self._hook(i)) for i, (_, base) in enumerate(targets)]

    def _hook(self, index):
        def hook(module, inputs, output):
            if self.strength == 0:
                return output
            if self._particles is None:
                raise RuntimeError("active adapter requires a particle bank")
            residual = self.layers[index](inputs[0], self._particles)
            return output + self.strength * residual.to(output.dtype)

        return hook

    @contextmanager
    def active(self, particles, strength=1.0):
        validate_strength(strength)
        if strength:
            if not isinstance(particles, torch.Tensor) or tuple(particles.shape) != self.particle_shape:
                raise ValueError("particle shape differs from the shared stamp")
            if particles.device != self.layers[0].up.weight.device:
                raise ValueError("particles and adapter must use the same device")
        previous = self._particles, self.strength
        # Keep the context unregistered: the shared game network owns the particle parameter.
        object.__setattr__(self, "_particles", particles)
        self.strength = float(strength)
        try:
            yield self
        finally:
            object.__setattr__(self, "_particles", previous[0])
            self.strength = previous[1]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
