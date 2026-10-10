"""Actual tiny 2.1 transformer/pipeline/scheduler with random weights and synthetic text.

The processor stub only satisfies pipeline initialization; no tokenization is tested.
No pretrained files, encoder weights, or Hub downloads are involved.
"""

import hashlib

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline, QwenImage21Transformer2DModel

from .contracts import lock
from .runtime import Runtime


class SyntheticProcessor:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, *args, **kwargs):
        return [[0]]

    def encode(self, text):
        return [0]


class TinyRuntime(Runtime):
    validation_kind = "tiny-native-random-weights-synthetic-text"

    def __init__(self, seed=7):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        transformer = (
            QwenImage21Transformer2DModel(
                in_channels=8,
                out_channels=8,
                num_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                context_in_dim=16,
                mlp_ratio=2,
                axes_dims_rope=(4, 6, 6),
            )
            .eval()
            .requires_grad_(False)
        )
        # Use exactly the official dynamic-shift scheduler configuration.
        self.pipe = QwenImage21Pipeline(
            transformer=transformer,
            text_encoder=None,
            vae=None,
            processor=SyntheticProcessor(),
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(lock()["scheduler_config"]),
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.base_identity = {"fixture_seed": seed, "kind": self.validation_kind}

    def encode(self, prompt):
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "little")
        generator = torch.Generator().manual_seed(seed)
        # Include right padding to exercise the native attention mask.
        return {
            "prompt_embeds": torch.randn(1, 5, 16, generator=generator).to(self.model.dtype),
            "prompt_embeds_mask": torch.tensor([[True, True, True, True, False]]),
            "image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
        }
