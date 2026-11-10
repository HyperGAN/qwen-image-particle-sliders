"""Explicit Qwen-Image2.1 fresh-components policy; authored Source, not executed.

Use the already admitted public PG0.9.5 API, not a runtime/core upgrade.
Only down matrices receive host-scale QR; declared banks receive R2Normal.
Every other owned parameter is explicitly KEEP.  Original model modules,
Parameter objects, forward methods and optimizer ownership remain untouched.
The product factory must supply truthful construction intent before optimizers;
this helper cannot detect arbitrary earlier load_state_dict calls.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import torch
from particlegan import init as pg_init
from torch import nn

INIT_SHA256 = "3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b"
SCHEMA = "product-owned-PG095-downQR-bankR2-KEEP-fresh-initialization-v1"
MARKER = "_fresh_PG095_initialization_receipt_sha256"
POLICIES = {"qwen": "qwen-image21-fresh-downQR-sqrt5-bankR2std002-router-readoutKEEP-v1"}


@dataclass(frozen=True)
class Construction:
    mode: str
    cohort_id: str
    options_sha256: str
    restored_from: str | None = None
    completed_steps: int = 0


class _DeclaredOwnedParameters(nn.Module):
    """Private declarations only; host classes/global host declarations stay exact."""

    def __init__(self, rows):
        super().__init__()
        self.specs = {}
        for index, row in enumerate(rows):
            name = f"p{index}"
            self.register_parameter(name, row["parameter"])
            self.specs[name] = row["distribution"]


pg_init.register(_DeclaredOwnedParameters, lambda module: module.specs)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_record(path):
    p = Path(path).absolute()
    raw = p.read_bytes()
    return {"file": p.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def tensor_sha(value):
    # A fresh flat allocation avoids singleton zero-stride Float->Byte views.
    flat = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
    flat.copy_(value.detach().reshape(-1).cpu())
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def parameter_seed(policy, seed, name):
    return int.from_bytes(hashlib.sha256(f"{policy}:{seed}:{name}".encode()).digest()[:8], "big") % 2**63


def _bias_free_linear(leaf):
    if type(leaf) is not nn.Linear:
        raise ValueError("canonical Qwen keeps exact ordinary nn.Linear leaves")
    if (
        leaf.bias is not None
        or any(leaf.children())
        or list(leaf.buffers())
        or nn.utils.parametrize.is_parametrized(leaf)
        or set(leaf._parameters) != {"weight", "bias"}
        or type(leaf.weight) is not nn.Parameter
        or not leaf.weight.requires_grad
        or tuple(leaf.weight.shape) != (leaf.out_features, leaf.in_features)
    ):
        raise ValueError("fresh ordinary bias-free down/up Parameter required")
    return leaf.weight


def policies(product, adapter, critic, *, particles=None, qwen_profile=None):
    """All actual added adapter/critic/bank trainables; never the native model."""
    if product != "qwen" or qwen_profile != "canonical-readout-v1":
        raise ValueError("separate canonical Qwen profile required; no HNDL/strict migration")
    from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic

    from qwen_image_sliders.adapters import AdapterBank

    if type(adapter) is not AdapterBank or type(critic) is not ConditionalReadoutCritic:
        raise ValueError("original canonical Qwen AdapterBank and readout critic required")
    rows, by_id, storages = [], {}, set()
    for prefix, root in (("adapter", adapter), ("critic", critic)):
        for name, parameter in root.named_parameters(remove_duplicate=False):
            if not parameter.requires_grad or id(parameter) in by_id:
                raise ValueError("unique actual added trainables required")
            row = {
                "parameter": parameter,
                "name": prefix + "." + name,
                "distribution": pg_init.KEEP,
                "role": "constructor-law-KEEP",
            }
            rows.append(row)
            by_id[id(parameter)] = row
    if (
        type(particles) is not nn.Parameter
        or not particles.requires_grad
        or tuple(particles.shape) != (128, 4)
        or id(particles) in by_id
    ):
        raise ValueError("original separately owned Qwen-Image2.1128x4 bank required")
    row = {
        "parameter": particles,
        "name": "bank.particles",
        "distribution": pg_init.R2Normal(0.0, 0.02),
        "role": "declared-bank-R2std002",
    }
    rows.append(row)
    by_id[id(particles)] = row
    leaves = [(layer.down, layer.up) for layer in adapter.layers]
    if not leaves:
        raise ValueError("empty Qwen-Image2.1 adapter")
    if any(type(x) is not nn.Linear for pair in leaves for x in pair):
        raise ValueError("canonical Qwen leaves must be original ordinary Linear")
    for down, up in leaves:
        weight, zero = _bias_free_linear(down), _bias_free_linear(up)
        if id(weight) not in by_id or id(zero) not in by_id or not bool(torch.all(zero == 0)):
            raise ValueError("original owned down and protected zero-UP required")
        bound = 1 / math.sqrt(down.in_features)
        by_id[id(weight)].update(
            distribution=pg_init.Uniform(-bound, bound), role="declared-host-scale-down-QR"
        )
        by_id[id(zero)].update(distribution=pg_init.KEEP, role="protected-zero-UP")
    for row in rows:
        p = row["parameter"]
        if (
            type(p) is not nn.Parameter
            or p.grad is not None
            or not p.is_contiguous()
            or not p.is_floating_point()
            or p.is_complex()
            or p.device.type == "meta"
            or not bool(torch.isfinite(p).all())
            or hasattr(p, MARKER)
        ):
            raise ValueError("fresh materialized owned real Parameter required")
        storage = (str(p.device), p.untyped_storage().data_ptr())
        if p.numel() and storage in storages:
            raise ValueError("owned parameters share storage")
        if p.numel():
            storages.add(storage)
    return rows


def initialize(
    product,
    adapter,
    critic,
    *,
    construction,
    options,
    seed,
    source_identity,
    particles=None,
    qwen_profile=None,
    optimizers=(),
):
    """One fresh opt-in call before optimizers; restore paths must never call it."""
    if (
        not isinstance(construction, Construction)
        or construction.mode != "fresh"
        or type(construction.completed_steps) is not int
        or construction.completed_steps != 0
        or construction.restored_from is not None
        or not construction.cohort_id
        or construction.options_sha256 != hashlib.sha256(canonical(options).encode()).hexdigest()
        or not isinstance(optimizers, (tuple, list))
        or len(optimizers) != 0
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or not isinstance(source_identity, dict)
        or not source_identity
    ):
        raise ValueError(
            "truthful fresh construction before optimizers required; resume/inference never initialize"
        )
    if metadata.version("particlegan") != "0.9.5" or source_record(pg_init.__file__)["sha256"] != INIT_SHA256:
        raise ValueError("exact already admitted PG0.9.5 initializer Source required")
    rows = policies(product, adapter, critic, particles=particles, qwen_profile=qwen_profile)
    policy = POLICIES[product]
    before = [tensor_sha(row["parameter"]) for row in rows]
    ids = [id(row["parameter"]) for row in rows]
    holder = _DeclaredOwnedParameters(rows)
    seeds = {f"p{i}": parameter_seed(policy, seed, row["name"]) for i, row in enumerate(rows)}
    # Strict complete private declarations; no patch to host/global layer declarations.
    pg_init.deterministic_orthogonal_(holder, strict=True, parameter_seeds=seeds)
    if ids != [id(row["parameter"]) for row in rows]:
        raise RuntimeError("initializer replaced an owned Parameter")
    detail = []
    for i, row in enumerate(rows):
        after = tensor_sha(row["parameter"])
        if row["distribution"] is pg_init.KEEP and before[i] != after:
            raise RuntimeError("explicit constructor/zero-UP KEEP changed")
        detail.append(
            {
                "name": row["name"],
                "role": row["role"],
                "shape": list(row["parameter"].shape),
                "dtype": str(row["parameter"].dtype),
                "seed": seeds[f"p{i}"],
                "distribution": repr(row["distribution"]),
                "before_sha256": before[i],
                "after_sha256": after,
            }
        )
    receipt = {
        "schema": SCHEMA,
        "policy": policy,
        "product": product,
        "cohort_id": construction.cohort_id,
        "completed_steps": 0,
        "restored_from": None,
        "seed": seed,
        "source_identity": source_identity,
        "initializer": source_record(pg_init.__file__),
        "helper": source_record(__file__),
        "parameter_identity_preserved": True,
        "parameter_policies": detail,
        "selection": "raw",
        "ema_decay": None,
    }
    digest = hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for row in rows:
        setattr(row["parameter"], MARKER, digest)
    return receipt
