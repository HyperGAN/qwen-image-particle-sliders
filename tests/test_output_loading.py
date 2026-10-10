from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLVideoProcessor

from qwen_image_sliders.runtime import GIB, Runtime
from qwen_image_sliders.tiny import TinyRuntime


def test_native_qwen_video_processor_dependency_without_model_files():
    processor = Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    assert processor.patch_size == 16 and processor.merge_size == 2
    assert processor.temporal_patch_size == 2


def test_actual_native_vae_pipeline_rgba_png_roundtrip(tmp_path):
    runtime = TinyRuntime(with_vae=True)
    image = runtime.sample(
        runtime.encode("An RGBA image with transparency."),
        seed=42,
        height=32,
        width=32,
        steps=2,
        output_type="pil",
    )[0]
    assert image.mode == "RGBA" and image.size == (32, 32)
    path = tmp_path / "native.png"
    image.save(path)
    with Image.open(path) as reopened:
        assert reopened.mode == "RGBA" and reopened.size == image.size
        assert np.array_equal(np.asarray(reopened), np.asarray(image))
        assert reopened.getchannel("A").getextrema()[0] < 255


def test_direct_cuda_loading_options_are_mocked_unit_contract(monkeypatch, tmp_path):
    calls = {}
    report = {"host_available_bytes": 20 * GIB, "cuda_visible": True, "gpu_free_bytes": 40 * GIB}
    monkeypatch.setattr("qwen_image_sliders.runtime.preflight", lambda *args: report)
    monkeypatch.setattr("qwen_image_sliders.runtime.verify_snapshot", lambda path: path)

    def device_context(device):
        calls["selected_device"] = str(device)
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "device", device_context)

    class Pipe:
        config = SimpleNamespace(sample_sigmas=None)
        transformer = torch.nn.Linear(1, 1)
        text_encoder = torch.nn.Linear(1, 1)
        vae = torch.nn.Linear(1, 1)

        def set_progress_bar_config(self, **kwargs):
            pass

        def to(self, *args):
            raise AssertionError("direct CUDA loading must avoid CPU-first staging and pipeline.to")

    def from_pretrained(path, **kwargs):
        calls["load_options"] = kwargs
        return Pipe()

    monkeypatch.setattr("diffusers.QwenImage21Pipeline.from_pretrained", from_pretrained)
    Runtime(tmp_path, device="cuda:1")
    assert calls["selected_device"] == "cuda:1"
    assert calls["load_options"] == {
        "device_map": "cuda",
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
