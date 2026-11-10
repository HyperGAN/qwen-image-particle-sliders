"""Only already-completed PreparedResponse banks; never prepare or recover packed records."""

import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass, fields

import torch
from particle_sliders.pg09_backends import PreparedResponse, ResponseRow
from particle_sliders.pg09_images import image_condition_features

from . import ownership
from .util import Checkpoint, canonical, equal, read_packet, save_packet, tree

SCHEMA = "QwenImage21-completed-native-PreparedResponse-cache-fresh95-v1"
PROVENANCE = "QwenImage21-completed-PreparedResponse-provenance-v1"
TEACHER_KEYS = ("rows", "seed", "height", "width", "trajectory_steps", "records_per_pair")


@dataclass(frozen=True)
class CacheBinding:
    """External retained whole-file identity AND separately retained provenance.

    Construct this from the actual completed cache's authenticated metadata.
    A serializer cannot establish that arbitrary caller-supplied claims are true.
    The owner-admission callback receives this entire provenance before attach.
    """

    checkpoint: Checkpoint
    provenance: dict


def teacher_configuration(config):
    return {k: deepcopy(config[k]) for k in TEACHER_KEYS}


def _finite(value):
    if isinstance(value, torch.Tensor):
        if (
            value.requires_grad
            or value.is_complex()
            or value.layout != torch.strided
            or not torch.isfinite(value).all()
        ):
            raise ValueError("completed frozen finite dense real tensors required")
    elif isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise ValueError("string condition keys required")
        for v in value.values():
            _finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _finite(v)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("unsupported cache object")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("finite cache scalar required")


def validate_row(row, config, runtime):
    if type(row) is not ResponseRow:
        raise ValueError("actual native ResponseRow required")
    expected = (1, int(runtime.model.config.in_channels), 1, config["height"] // 8, config["width"] // 8)
    if (
        row.schedule is not None
        or not isinstance(row.timestep, torch.Tensor)
        or row.timestep.numel() != 1
        or not 0 <= float(row.timestep) <= 1000
        or row.scale != 1
        or row.strength != 1
        or row.guidance != 1
    ):
        raise ValueError("original single Qwen native velocity/true_cfg1 row required")
    if (
        not isinstance(row.latent, torch.Tensor)
        or tuple(row.latent.shape) != expected
        or row.latent.dtype != torch.float32
    ):
        raise ValueError("complete native FP32 VAE field [1,C,1,H/8,W/8] required")
    for condition in (row.neutral, row.positive, row.negative):
        if not isinstance(condition, dict) or set(condition) != {"embeds", "mask", "image_pad_mask"}:
            raise ValueError("all original native Qwen embeddings/masks required")
        e, m, p = condition["embeds"], condition["mask"], condition["image_pad_mask"]
        if (
            not all(isinstance(x, torch.Tensor) for x in (e, p))
            or e.ndim != 3
            or e.shape[0] != 1
            or e.shape[2] != int(runtime.model.config.context_in_dim)
            or p.shape != e.shape[:2]
            or not e.is_floating_point()
            or p.dtype != torch.bool
            or bool(p.any())
            or (
                m is not None
                and (
                    not isinstance(m, torch.Tensor)
                    or m.shape != e.shape[:2]
                    or m.dtype not in (torch.bool, torch.int32, torch.int64)
                    or not bool(m.any())
                    or not bool(((m == 0) | (m == 1)).all())
                )
            )
        ):
            raise ValueError("native text-only condition geometry/dtypes differ")
    if not equal(row.negative, row.neutral):
        raise ValueError("original Qwen negative context is the neutral context")
    # Original encode_prompt may return None for an all-valid mask, or retain
    # its original bool/int padding mask. No mask casting/normalization here.
    for value in (row.neutral, row.positive, row.negative, row.latent, row.timestep):
        _finite(value)
    return expected


def validate_prepared(prepared, config, runtime):
    if len(prepared) != len(config["rows"]) * config["records_per_pair"] or len(prepared) < 2:
        raise ValueError("complete original fixed native row bank required")
    for item in prepared:
        if type(item) is not PreparedResponse:
            raise ValueError("actual PreparedResponse required; packed teacher dictionaries are not accepted")
        expected = validate_row(item.row, config, runtime)
        for value in (item.target, item.neutral):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected
                or value.dtype != torch.float32
            ):
                raise ValueError("complete native FP32 target/neutral fields required")
            _finite(value)
    return prepared


def _dump(item):
    return {
        "row": {f.name: tree(getattr(item.row, f.name), "cpu") for f in fields(ResponseRow)},
        "target": tree(item.target, "cpu"),
        "neutral": tree(item.neutral, "cpu"),
    }


def _restore(item, device):
    if (
        not isinstance(item, dict)
        or set(item) != {"row", "target", "neutral"}
        or set(item["row"]) != {f.name for f in fields(ResponseRow)}
    ):
        raise ValueError("complete serialized PreparedResponse fields required")
    return PreparedResponse(
        ResponseRow(**tree(item["row"], device)), tree(item["target"], device), tree(item["neutral"], device)
    )


def prepared_digest(item):
    from .policy import tensor_sha

    def digest(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_sha(value)}
        if isinstance(value, dict):
            return {k: digest(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return {"container": type(value).__name__, "values": [digest(v) for v in value]}
        return value

    return hashlib.sha256(canonical(digest(_dump(item))).encode()).hexdigest()


def _origin_check(origins, prepared, runtime, config, sources):
    if not isinstance(origins, list) or len(origins) != len(prepared):
        raise ValueError("every completed row origin required")
    synthetic = runtime.validation_kind == "tiny-native-random-weights-synthetic-text"
    kind = "NEW-synthetic-software-PreparedResponse" if synthetic else "completed-native-PreparedResponse"
    for i, (origin, item) in enumerate(zip(origins, prepared)):
        if (
            not isinstance(origin, dict)
            or set(origin)
            != {
                "kind",
                "completed",
                "row_index",
                "pair_index",
                "record_index",
                "control",
                "model_identity",
                "original_sources",
                "prepared_sha256",
                "completed_output_receipt",
            }
            or origin["kind"] != kind
            or origin["completed"] is not True
            or origin["row_index"] != i
            or origin["pair_index"] != i // config["records_per_pair"]
            or origin["record_index"] != i % config["records_per_pair"]
            or origin["control"] is not config["rows"][i // config["records_per_pair"]].get("control", False)
            or origin["model_identity"] != runtime.base_identity
            or origin["original_sources"] != sources
            or origin["prepared_sha256"] != prepared_digest(item)
        ):
            raise ValueError("retained PreparedResponse provenance/native/Source/row binding differs")
        receipt = origin["completed_output_receipt"]
        if synthetic:
            if receipt != {
                "scope": "NEW synthetic software data; no native teacher output claimed",
                "row": i,
            }:
                raise ValueError("synthetic rows cannot claim pretrained/native teacher provenance")
        elif (
            not isinstance(receipt, dict)
            or set(receipt) != {"path", "bytes", "sha256"}
            or type(receipt["path"]) is not str
            or not receipt["path"]
            or type(receipt["bytes"]) is not int
            or receipt["bytes"] < 1
            or type(receipt["sha256"]) is not str
            or len(receipt["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in receipt["sha256"])
        ):
            raise ValueError("actual retained native completed-output receipt FileRecord required")
    canonical(origins)


def calibration(prepared, origins):
    edits = []
    for item, origin in zip(prepared, origins):
        rms = float((item.target - item.neutral).double().square().mean().sqrt())
        if not origin["control"] and rms > 1e-8:
            edits.append(rms)
    if len(edits) < 2:
        raise ValueError("original median calibration needs two nondegenerate noncontrol edits")
    return {
        "response_scale": float(torch.tensor(edits, dtype=torch.float64).median()),
        "normalization_records": len(edits),
        "method": "original median noncontrol positive-edit RMS over complete native PreparedResponse fields",
    }


def save_prepared_cache(path, *, runtime, config, prepared, teacher_origins, cache_id):
    """Serialize retained prepared data, zero encode/model/teacher/prepare calls.

    Native callers must supply actual completed-output receipt metadata for each
    row. The returned binding records input provenance; it is not qualification
    or an assertion that this function itself observed any native output.
    """
    from .configuration import configuration

    ownership.require()
    if configuration(**config) != config:
        raise ValueError("exact original configuration required")
    if type(cache_id) is not str or not cache_id.strip():
        raise ValueError("retained cache ID required")
    sources = ownership.sources()["original"]
    validate_prepared(prepared, config, runtime)
    _origin_check(teacher_origins, prepared, runtime, config, sources)
    provenance = {
        "schema": PROVENANCE,
        "cache_id": cache_id,
        "teacher_configuration": teacher_configuration(config),
        "model_identity": deepcopy(runtime.base_identity),
        "original_sources": sources,
        "teacher_origins": deepcopy(teacher_origins),
    }
    conditions = image_condition_features(prepared).detach().cpu().clone()
    packet = {
        "schema": SCHEMA,
        "provenance": provenance,
        "rows": [_dump(x) for x in prepared],
        "conditions": conditions,
        "calibration": calibration(prepared, teacher_origins),
        "completed": True,
        "model_teacher_or_prepare_calls_by_serializer": 0,
    }
    return CacheBinding(save_packet(path, packet), deepcopy(provenance))


def load_cache(binding, *, runtime, config, device="cpu"):
    if type(binding) is not CacheBinding or not isinstance(binding.provenance, dict):
        raise ValueError("authentic separately retained whole-file CacheBinding/provenance required")
    packet = read_packet(binding.checkpoint)
    keys = {
        "schema",
        "provenance",
        "rows",
        "conditions",
        "calibration",
        "completed",
        "model_teacher_or_prepare_calls_by_serializer",
    }
    expected = dict(binding.provenance)
    if (
        not isinstance(packet, dict)
        or set(packet) != keys
        or packet["schema"] != SCHEMA
        or packet["provenance"] != expected
        or set(expected)
        != {
            "schema",
            "cache_id",
            "teacher_configuration",
            "model_identity",
            "original_sources",
            "teacher_origins",
        }
        or expected["schema"] != PROVENANCE
        or type(expected["cache_id"]) is not str
        or not expected["cache_id"].strip()
        or packet["completed"] is not True
        or packet["model_teacher_or_prepare_calls_by_serializer"] != 0
        or expected["teacher_configuration"] != teacher_configuration(config)
        or expected["model_identity"] != runtime.base_identity
        or expected["original_sources"] != ownership.sources()["original"]
    ):
        raise ValueError("retained completed PreparedResponse cache/provenance/native/Source differs")
    prepared = [_restore(x, "cpu") for x in packet["rows"]]
    validate_prepared(prepared, config, runtime)
    _origin_check(expected["teacher_origins"], prepared, runtime, config, expected["original_sources"])
    conditions = packet["conditions"]
    _finite(conditions)
    if (
        not isinstance(conditions, torch.Tensor)
        or conditions.dtype != torch.float32
        or conditions.ndim != 2
        or conditions.shape[0] != len(prepared)
        or not torch.equal(image_condition_features(prepared).cpu(), conditions)
        or packet["calibration"] != calibration(prepared, expected["teacher_origins"])
    ):
        raise ValueError("persisted exact conditions/calibration differ")
    pin = binding.checkpoint
    identity = {
        "schema": SCHEMA,
        "bytes": pin.bytes,
        "sha256": pin.sha256,
        "provenance": deepcopy(expected),
        "calibration": deepcopy(packet["calibration"]),
        "row_count": len(prepared),
    }
    return [_restore(x, device) for x in packet["rows"]], conditions.to(device), identity


def to_device(prepared, device):
    return [_restore(_dump(x), device) for x in prepared]
