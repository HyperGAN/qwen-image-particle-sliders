"""Explicit cached fresh/FULL owners using original canonical Qwen components."""

import hashlib
from copy import deepcopy
from pathlib import Path

import torch
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_images import QwenImage21ResponseAdapter
from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic

from qwen_image_sliders.adapters import AdapterBank
from qwen_image_sliders.contracts import formulation
from qwen_image_sliders.lifecycle import _validate_runtime

from . import PROFILE, ownership, policy
from .cache import load_cache, to_device
from ._configuration import configuration
from .ownership import Owner
from .util import equal, new_directory, read_packet, save_packet, write_json

FULL_SCHEMA = "QwenImage21-explicit-cached-fresh95-v1-FULL"
RAW_SCHEMA = "QwenImage21-explicit-cached-fresh95-v1-RAW"
ARCHITECTURE = "canonical-readout-v1"


class ReceiptBackend:
    def __init__(self, backend, initialization, sources, cache_identity):
        self.backend = backend
        self.network, self.particles = backend.network, backend.particles
        self._receipt, self._sources, self._cache = (
            deepcopy(initialization),
            deepcopy(sources),
            deepcopy(cache_identity),
        )

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def metadata(self):
        return dict(
            self.backend.metadata(),
            fresh_initialization={
                "profile": PROFILE,
                "receipt": deepcopy(self._receipt),
                "sources": deepcopy(self._sources),
            },
            completed_native_cache=deepcopy(self._cache),
        )


def receipt_valid(receipt, cohort, config, sources):
    return (
        isinstance(receipt, dict)
        and receipt.get("schema") == policy.SCHEMA
        and receipt.get("policy") == policy.POLICIES["qwen"]
        and receipt.get("product") == "qwen"
        and receipt.get("cohort_id") == cohort
        and receipt.get("seed") == config["seed"]
        and receipt.get("source_identity") == dict(sources, network_profile=ARCHITECTURE)
        and receipt.get("completed_steps") == 0
        and receipt.get("restored_from") is None
        and receipt.get("parameter_identity_preserved") is True
        and receipt.get("selection") == "raw"
        and receipt.get("ema_decay") is None
    )


def attach(owner, config):
    stamp, declared = formulation(batch=config["batch"], lr=config.get("lr"))
    # Retain partial-constructor handles on failure, while executing the exact
    # original AdapterBank constructor rather than replacing its layer classes.
    bank = AdapterBank.__new__(AdapterBank)
    owner.bank = bank
    try:
        AdapterBank.__init__(bank, owner.runtime.model, stamp)
    finally:
        owner.capture_handles()
    bank.to(owner.runtime.device)
    particles = torch.nn.Parameter(
        torch.randn(int(stamp.spec["parts"]), int(stamp.spec["particle_dim"]), device=owner.runtime.device)
        * 0.02
    )
    if config["gradient_checkpointing"]:
        owner.runtime.model.enable_gradient_checkpointing()
    native = QwenImage21ResponseAdapter(
        owner.runtime, bank, particles, model_identity=owner.runtime.base_identity
    )
    owner.seal()
    return particles, native, declared


def admit(runtime, config, request, owner_admission):
    from qwen_image_sliders.adapters import target_modules
    from qwen_image_sliders.runtime import Runtime
    from qwen_image_sliders.tiny import TinyRuntime

    if type(runtime) not in (Runtime, TinyRuntime) or runtime.pipe.transformer is not runtime.model:
        raise ValueError("original canonical Runtime/TinyRuntime and actual transformer owner required")
    target_modules(runtime.model)  # Exact original model/attention-output shape gate, zero forwards.
    _validate_runtime(runtime, config)  # Original frozen/native-kind/device/precision contract.
    if not callable(owner_admission) or owner_admission(deepcopy(request)) is not True:
        raise ValueError("external owner must admit this exact cached operation before attachment")


class Training(Owner):
    def __init__(
        self,
        runtime,
        cache,
        directory,
        *,
        mode,
        cohort_id,
        config,
        resume=None,
        ema_decay=None,
        owner_admission=None,
        release=None,
    ):
        if mode not in ("fresh", "resume") or type(cohort_id) is not str or not cohort_id.strip():
            raise ValueError("explicit fresh|resume and nonempty new cohort required")
        if (mode == "fresh" and resume is not None) or (mode == "resume" and resume is None):
            raise ValueError("fresh refuses saved state; resume requires exact new-profile FULL")
        if ema_decay is not None:
            raise ValueError("no weight EMA; current RAW only")
        directory = Path(directory).absolute()
        if directory.exists():
            raise FileExistsError("new owned output directory required")
        ownership.require()
        if configuration(**config) != config:
            raise ValueError("exact declared original-option configuration required")
        config = deepcopy(config)
        sources = ownership.sources()
        packet = read_packet(resume) if mode == "resume" else None
        if packet is not None:
            keys = {
                "schema",
                "profile",
                "cohort_id",
                "config",
                "sources",
                "initialization",
                "cache_identity",
                "game",
                "ema_decay",
            }
            if (
                not isinstance(packet, dict)
                or set(packet) != keys
                or packet["schema"] != FULL_SCHEMA
                or packet["profile"] != PROFILE
                or packet["cohort_id"] != cohort_id
                or packet["config"] != config
                or packet["sources"] != sources
                or packet["ema_decay"] is not None
                or not receipt_valid(packet["initialization"], cohort_id, config, sources)
            ):
                raise ValueError("exact new-profile FULL/receipt required; no old migration")
        # Every cache/native/Source/provenance assertion precedes added components.
        prepared, conditions, cache_identity = load_cache(cache, runtime=runtime, config=config, device="cpu")
        if packet is not None and packet["cache_identity"] != cache_identity:
            raise ValueError("FULL retained cache differs")
        admit(
            runtime,
            config,
            {
                "operation": "construct-cached-train",
                "mode": mode,
                "profile": PROFILE,
                "cohort_id": cohort_id,
                "config": config,
                "model_identity": runtime.base_identity,
                "completed_prepared_cache": cache_identity,
                "selection": "raw",
                "ema_decay": None,
            },
            owner_admission,
        )
        super().__init__(runtime, release)
        self.directory, self.config, self.cohort_id, self.sources = directory, config, cohort_id, sources
        self.cache_identity = cache_identity
        devices = [runtime.device.index or 0] if runtime.device.type == "cuda" else []
        try:
            prepared, conditions = to_device(prepared, runtime.device), conditions.to(runtime.device)
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
                if devices:
                    torch.cuda.set_rng_state(
                        torch.Generator(device=runtime.device).manual_seed(config["seed"]).get_state(),
                        runtime.device,
                    )
                particles, native, declared = attach(self, config)
                critic = ConditionalReadoutCritic(prepared[0].target.numel(), conditions.shape[1]).to(
                    runtime.device
                )
            if mode == "fresh":
                self.initialization = policy.initialize(
                    "qwen",
                    self.bank,
                    critic,
                    construction=policy.Construction(
                        "fresh", cohort_id, hashlib.sha256(policy.canonical(config).encode()).hexdigest()
                    ),
                    options=config,
                    seed=config["seed"],
                    source_identity=dict(sources, network_profile=ARCHITECTURE),
                    particles=particles,
                    qwen_profile=ARCHITECTURE,
                    optimizers=(),
                )
            else:
                self.initialization = deepcopy(packet["initialization"])
            self.backend = ReceiptBackend(native, self.initialization, sources, cache_identity)
            calibrated = cache_identity["calibration"]
            self.game = ConditionalPG09(
                self.backend,
                prepared,
                critic,
                conditions=conditions,
                particles=particles,
                response_domain="continuous_field",
                response_scale=calibrated["response_scale"],
                preset=config["preset"],
                total_steps=config["steps"],
                batch_size=int(declared["adv_batch"]),
                seed=config["seed"],
                sampling="balanced",
                recipe_overrides={k: config[k] for k in ("lr", "d_lr_mult", "prior_lr_mult") if k in config},
            )
            self.game.normalization_records = calibrated["normalization_records"]
            if mode == "resume":
                self.game.load_state_dict(packet["game"])
                if not equal(self.game.state_dict(), packet["game"]):
                    raise RuntimeError("entire typed game/optimizer/RNG restore differs")
            new_directory(directory)
            write_json(directory / "initialization.json", self.initialization)
            self.entry_FULL = self.save()
            self.entry_RAW = self.export(directory / f"RAW{self.step:06d}.pt")
        except BaseException as primary:
            self.observe_native()
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain interruption-safe pending cleanup
                failure = BaseExceptionGroup("Qwen cached entry/cleanup failed", [primary, cleanup])
                failure.cleanup_owner = self
                raise failure
            raise

    @property
    def step(self):
        return self.game.completed_steps

    def update(self):
        self.require_open()
        _validate_runtime(self.runtime, self.config)
        try:
            return self.game.step()
        finally:
            self.observe_native()

    def save(self):
        self.require_open()
        return save_packet(
            self.directory / f"FULL{self.step:06d}.pt",
            {
                "schema": FULL_SCHEMA,
                "profile": PROFILE,
                "cohort_id": self.cohort_id,
                "config": deepcopy(self.config),
                "sources": deepcopy(self.sources),
                "initialization": deepcopy(self.initialization),
                "cache_identity": deepcopy(self.cache_identity),
                "game": self.game.state_dict(),
                "ema_decay": None,
            },
        )

    def export(self, path):
        self.require_open()
        state = self.game.export_state_dict()
        return save_packet(
            path,
            {
                "schema": RAW_SCHEMA,
                "profile": PROFILE,
                "cohort_id": self.cohort_id,
                "config": deepcopy(self.config),
                "sources": deepcopy(self.sources),
                "initialization": deepcopy(self.initialization),
                "cache_identity": deepcopy(self.cache_identity),
                "model_identity": deepcopy(self.runtime.base_identity),
                "backend_metadata": self.backend.metadata(),
                "network": state["network"],
                "completed_steps": self.step,
                "selection": "raw",
                "ema_decay": None,
                "rendering": "native VAE-field response; pretrained/new-profile full rendering unqualified",
            },
        )
