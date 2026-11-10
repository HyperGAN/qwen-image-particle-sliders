"""New-profile strict RAW inference; no cache, critic, game or fresh initialization."""

import math

import torch
from particle_sliders.lifecycle import _network_observation
from particle_sliders.pg09_backends import ResponseRow

from qwen_image_sliders.lifecycle import _observation, _render_native, _validate_render

from . import PROFILE, ownership
from .cache import validate_row
from .configuration import configuration
from .training import RAW_SCHEMA, Owner, ReceiptBackend, admit, attach, receipt_valid
from .util import canonical, equal, read_packet


class Inference(Owner):
    def __init__(self, runtime, *, raw, owner_admission=None, release=None):
        ownership.require()
        sources = ownership.sources()
        packet = read_packet(raw)
        keys = {
            "schema",
            "profile",
            "cohort_id",
            "config",
            "sources",
            "initialization",
            "cache_identity",
            "model_identity",
            "backend_metadata",
            "network",
            "completed_steps",
            "selection",
            "ema_decay",
            "rendering",
        }
        if (
            not isinstance(packet, dict)
            or set(packet) != keys
            or packet["schema"] != RAW_SCHEMA
            or packet["profile"] != PROFILE
            or packet["sources"] != sources
            or packet["selection"] != "raw"
            or packet["ema_decay"] is not None
            or packet["model_identity"] != runtime.base_identity
            or type(packet["completed_steps"]) is not int
            or packet["completed_steps"] < 0
        ):
            raise ValueError("exact new-profile RAW metadata required; no old migration")
        config = packet["config"]
        if (
            configuration(**config) != config
            or packet["completed_steps"] > config["steps"]
            or not receipt_valid(packet["initialization"], packet["cohort_id"], config, sources)
        ):
            raise ValueError("RAW exact config/receipt/horizon differs")
        admit(
            runtime,
            config,
            {
                "operation": "strict-RAW-infer",
                "profile": PROFILE,
                "cohort_id": packet["cohort_id"],
                "model_identity": runtime.base_identity,
                "completed_steps": packet["completed_steps"],
                "selection": "raw",
                "ema_decay": None,
            },
            owner_admission,
        )
        super().__init__(runtime, release)
        self.packet, self.config = packet, config
        devices = [runtime.device.index or 0] if runtime.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
                if devices:
                    torch.cuda.set_rng_state(
                        torch.Generator(device=runtime.device).manual_seed(config["seed"]).get_state(),
                        runtime.device,
                    )
                _particles, self.native, _ = attach(self, config)
            self.backend = ReceiptBackend(
                self.native, packet["initialization"], sources, packet["cache_identity"]
            )
            if canonical(self.backend.metadata()) != canonical(packet["backend_metadata"]):
                raise ValueError("RAW backend identity differs")
            expected = self.native.network.state_dict()
            state = packet["network"]
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError("RAW tensor keys differ")
            for k, v in state.items():
                if (
                    not isinstance(v, torch.Tensor)
                    or v.shape != expected[k].shape
                    or v.dtype != expected[k].dtype
                    or not torch.isfinite(v).all()
                ):
                    raise ValueError("RAW tensor schema/finite state differs")
            self.native.network.load_state_dict(state, strict=True)
            if not equal(self.native.network.state_dict(), state):
                raise RuntimeError("RAW tensor equality failed")
        except BaseException as primary:
            self.observe_native()
            try:
                self.close()
            except BaseException as cleanup:  # noqa: BLE001 - retain interruption-safe pending cleanup
                failure = BaseExceptionGroup("Qwen RAW entry/cleanup failed", [primary, cleanup])
                failure.cleanup_owner = self
                raise failure
            raise

    @torch.no_grad()
    def response(self, row, *, strength=1.0, seed=0):
        self.require_open()
        if type(row) is not ResponseRow:
            raise ValueError("original native ResponseRow required")
        validate_row(row, self.config, self.runtime)
        if type(seed) is not int or not 0 <= seed < 2**63 - 100:
            raise ValueError("original seed range required")
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("native scale [0,1] required")
        device = self.runtime.device
        devices = [device.index or 0] if device.type == "cuda" else []
        try:
            with _network_observation(
                self.native.network, seed, devices=devices, observation=lambda: _observation(self.runtime)
            ):
                return self.native.cfg(row, row.latent, row.timestep, scale=float(strength)).detach().clone()
        finally:
            self.observe_native()

    @torch.no_grad()
    def render(self, *, prompt, strength, seed, **native):
        self.require_open()
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("native scale [0,1] required")
        _validate_render(dict(native, prompt=prompt), dummy=self.config["dummy"])
        defaults = {
            "height": self.config["height"],
            "width": self.config["width"],
            "steps": self.config["trajectory_steps"],
            "output_type": "latent",
        }
        devices = [self.runtime.device.index or 0] if self.runtime.device.type == "cuda" else []
        try:
            with _network_observation(
                self.native.network, seed, devices=devices, observation=lambda: _observation(self.runtime)
            ):
                return _render_native(
                    self.runtime,
                    self.native.network,
                    defaults,
                    prompt=prompt,
                    strength=float(strength),
                    seed=seed,
                    **native,
                )
        finally:
            self.observe_native()
