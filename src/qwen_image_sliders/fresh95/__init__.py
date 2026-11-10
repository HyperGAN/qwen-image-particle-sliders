"""Opt-in cached Qwen-Image2.1 fresh95 APIs; old registrations stay unchanged."""

PROFILE = "qwen-image21-explicit-cached-fresh95-v1"


def open_training(*args, **kwargs):
    from .training import Training

    return Training(*args, **kwargs)


def open_inference(*args, **kwargs):
    from .inference import Inference

    return Inference(*args, **kwargs)


def save_prepared_cache(*args, **kwargs):
    from .cache import save_prepared_cache as save

    return save(*args, **kwargs)


def configuration(*args, **kwargs):
    from .configuration import configuration as make

    return make(*args, **kwargs)
