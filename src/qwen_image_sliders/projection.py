"""Native joint text/image output cropping shared with the image response API."""

import math

import torch


def predict(model, kwargs):
    if isinstance(kwargs, list):
        # 2.1 RoPE ties image positions to prefix length. Per-row forwards preserve native layouts.
        return torch.cat([predict(model, row) for row in kwargs])
    # The native model emits the joint text/image sequence; the pipeline takes its target suffix.
    tokens = math.prod(kwargs["img_shapes"][0][-1])
    return model(**kwargs, return_dict=False)[0][:, -tokens:].float()
