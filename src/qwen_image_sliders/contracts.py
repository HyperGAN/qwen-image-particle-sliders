"""Pinned identities and formulation contracts for model-specific artifacts."""

import hashlib
import json
import math
from importlib.resources import files

from particle_sliders import winning_formulation

FORMAT = "qwen-image21-routed-particle-v1"
PROTOCOL = {
    "id": "qwen-image21-t2i-paired-velocity-sketch-v1",
    "true_cfg_scale": 1.0,
    "use_kv_cache": False,
    "prediction": "native-flow-velocity",
    "projection": "all-latent-coordinate-step-seeded-rademacher",
    "projection_seed": 31415,
    "projection_stride": 1000003,
    "calibration": "exact-isotropic-edit-rms-moments",
    "trained_strength": 1.0,
    "strength_range": [0.0, 1.0],
    "image_conditioning": False,
}


def lock():
    return json.loads(files(__package__).joinpath("backend.lock.json").read_text())


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def digest(data):
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def formulation(*, batch=1, g_lr=None, declared=None):
    stamp = winning_formulation()
    config = stamp.as_dict()
    if declared is not None:
        # Recover tuple fields serialized in JSON; require() rejects all other drift.
        config = {k: tuple(declared[k]) if isinstance(v, tuple) else declared[k] for k, v in config.items()}
    else:
        if batch < 1:
            raise ValueError("batch must be positive")
        config["adv_batch"] = int(batch)
        if g_lr is not None:
            if not math.isfinite(g_lr) or g_lr <= 0:
                raise ValueError("g-lr must be finite and positive")
            config["g_lr"] = float(g_lr)
    stamp.require(config)
    return stamp, config


def formulation_record(stamp, declared):
    return json.loads(
        canonical(
            {
                "architecture_id": stamp.architecture_id,
                "formulation_id": stamp.formulation_id,
                "provisional": bool(stamp.formulation_provisional),
                "declared": declared,
            }
        )
    )


def validate_shape(height, width):
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise ValueError("Qwen-Image 2.1 height and width must be positive multiples of 32")


def validate_strength(strength):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("the shared unipolar formulation supports strength in [0, 1]")
