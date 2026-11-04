"""Native CPU Qwen3-VL conditioning with explicit, verified precision.

The FP32 route casts original BF16 parameters exactly, and rounds native hidden
states to BF16 for transport. It does not claim BF16-encoder output equivalence.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path

from .execution import initialization, prompts, require_host, require_policy


def tensor_digest(value):
    import torch
    if value is None:
        return None
    value = value.detach().to('cpu').contiguous()
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite conditioning/base tensor')
    head = json.dumps([list(value.shape), str(value.dtype)], separators=(',', ':')).encode()
    # Keep one tensor resident; hashing must not create two full embedding-sized
    # bytes copies on top of the FP32 encoder and its source-cast check.
    result=hashlib.sha256(head)
    result.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return result.hexdigest()


def encoding_identity(encoding):
    return {key: None if value is None else dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=tensor_digest(value)) for key, value in encoding.items()}


def verify_parameters(model, directory, *, source_dtype, loaded_dtype, device='cpu'):
    """Compare every actual loaded parameter against one source tensor at a time."""
    import torch
    from safetensors import safe_open
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text())['weight_map']
    else:
        names = []
        for path in sorted(directory.glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu', backend='pread') as f:
                names.extend((key, path.name) for key in f.keys())
        mapping = dict(names)
        if len(mapping) != len(names):
            raise ValueError('duplicate checkpoint tensor')
    actual = model.state_dict()
    if set(actual) != set(mapping):
        raise ValueError('native loaded state keys differ from exact source checkpoint')
    checks = {}
    for filename in sorted(set(mapping.values())):
        with safe_open(directory / filename, framework='pt', device='cpu', backend='pread') as f:
            for key in sorted(k for k, name in mapping.items() if name == filename):
                source = f.get_tensor(key)
                value = actual[key]
                if (source.dtype != source_dtype or value.dtype != loaded_dtype
                        or str(value.device) != device or source.shape != value.shape
                        or not torch.isfinite(value).all()
                        or not torch.equal(source.to(loaded_dtype), value.detach().cpu())):
                    raise ValueError('actual CPU encoder parameter differs: ' + key)
                checks[key] = tensor_digest(value)
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('conditioning model must be frozen')
    return dict(tensors=len(checks), parameters=sum(p.numel() for p in model.parameters()),
        exact_source_cast=True, source_dtype=str(source_dtype), loaded_dtype=str(loaded_dtype),
        state_sha256=hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest())


def frozen_state(model):
    checks = {key: tensor_digest(value) for key, value in model.state_dict().items()}
    return hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest()


def buffer_identity(model):
    """Native rotary buffers are not necessarily stored in the checkpoint."""
    import torch
    registered={name:dict(device=str(value.device),dtype=str(value.dtype),shape=list(value.shape),
            sha256=tensor_digest(value)) for name,value in model.named_buffers()}
    plain={};rotary={}
    for name,module in model.named_modules():
        for attr,value in vars(module).items():
            if isinstance(value,torch.Tensor):
                plain[name+'.'+attr]=dict(device=str(value.device),dtype=str(value.dtype),shape=list(value.shape),sha256=tensor_digest(value))
        if type(module).__name__.endswith('RotaryEmbedding'):
            attrs={key:getattr(module,key) for key in ('rope_type','attention_scaling','mrope_section','max_seq_len_cached','original_max_seq_len') if hasattr(module,key)}
            attrs['rope_parameters']=getattr(getattr(module,'config',None),'rope_parameters',None)
            rotary[name]=json.loads(json.dumps(attrs,sort_keys=True))
    return dict(registered=registered,plain_tensor_attributes=plain,native_rotary_attributes=rotary)


def config_precision(model):
    return dict(dtype=str(getattr(model.config,'dtype',None)),
        text_dtype=str(getattr(model.config.text_config,'dtype',None)),
        vision_dtype=str(getattr(model.config.vision_config,'dtype',None)))


class CPUConditioner:
    """Own only the CPU encoder; no transformer, VAE or CUDA allocation."""
    def __init__(self, model, processor, scheduler, *, numerical, weights, guard=None):
        import torch
        from diffusers import QwenImage21Pipeline
        from transformers import Qwen3VLForConditionalGeneration
        if type(model) is not Qwen3VLForConditionalGeneration:
            raise ValueError('exact native Qwen3VLForConditionalGeneration required')
        dtype = getattr(torch, numerical['conditioning_precision'])
        if any(p.device.type != 'cpu' or p.dtype != dtype or p.requires_grad for p in model.parameters()):
            raise ValueError('native CPU encoder device/precision/frozen state differs')
        require_policy(numerical)
        self.numerical, self.guard = numerical, guard
        self.buffers=buffer_identity(model)
        self.config_precision=config_precision(model)
        self.weights=dict(weights,native_buffers=self.buffers,native_config_precision=self.config_precision)
        self.pipe = QwenImage21Pipeline(vae=None, text_encoder=model, processor=processor,
            transformer=None, scheduler=scheduler)
        self.pipe.set_progress_bar_config(disable=True)
        self.handle = model.register_forward_pre_hook(lambda *_: self.check())

    @classmethod
    def load_local(cls, snapshot, *, numerical, guard=None, cpu_only=True):
        """Caller must first obtain its separately reviewed host-memory lease."""
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from .encoder_loader import load_encoder
        from .assets import inspect_assets
        from .provenance import execution_identity
        if cpu_only and (torch.cuda.is_initialized() or __import__('os').environ.get('CUDA_VISIBLE_DEVICES') != ''):
            raise ValueError('standalone CPU conditioner requires CUDA_VISIBLE_DEVICES empty and no initialized CUDA')
        if execution_identity(conditioning_precision=numerical['conditioning_precision'])['numerical'] != numerical:
            raise ValueError('conditioning execution identity differs')
        assets = inspect_assets(snapshot, full_hashes=True)
        require_host(72)
        if guard is None or not callable(guard):
            raise ValueError('full CPU conditioner requires an active reviewed resource guardian')
        guard()
        directory = Path(snapshot)
        model = None
        try:
            with initialization():
                model = load_encoder(directory / 'text_encoder',dtype=getattr(torch,numerical['conditioning_precision']),guard=guard)
                model.eval().requires_grad_(False)
                guard()
                weights = verify_parameters(model, directory / 'text_encoder',
                    source_dtype=torch.bfloat16, loaded_dtype=getattr(torch, numerical['conditioning_precision']))
                processor = AutoProcessor.from_pretrained(directory / 'processor', local_files_only=True)
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(directory / 'scheduler', local_files_only=True)
                result = cls(model, processor, scheduler, numerical=numerical, weights=weights, guard=guard)
                result.assets = assets
                return result
        except BaseException:
            del model
            gc.collect()
            raise

    def check(self):
        require_policy(self.numerical)
        if self.guard is not None:
            self.guard()
        if buffer_identity(self.pipe.text_encoder)!=self.buffers:
            raise ValueError('actual native encoder rotary/buffer identity changed')
        if config_precision(self.pipe.text_encoder)!=self.config_precision:
            raise ValueError('actual native encoder config precision changed')
        config = self.pipe.text_encoder.config
        # Native text+vision must both remain eager; no flash/SDPA override.
        if any(getattr(c, '_attn_implementation', None) != 'eager'
               for c in (config, config.text_config, config.vision_config)):
            raise ValueError('actual CPU encoder attention implementation changed')

    def token_ids(self, prompt):
        prompts([prompt])
        pipe = self.pipe
        text = pipe.prompt_template_t2i.format(prompt)
        tokens = pipe.processor(text=[text], padding=True, padding_side='left', return_tensors='pt')
        ids = tokens['input_ids']
        if ids.shape[0] != 1 or ids.shape[1] > 512:
            raise ValueError('native full-prefix encoder token budget exceeds512')
        return ids

    def encode(self, prompt):
        import torch
        self.check()
        ids = self.token_ids(prompt)
        with torch.no_grad(), initialization():
            embeds, mask, image_mask = self.pipe.encode_prompt(prompt=prompt, device='cpu')
        if image_mask.any() or embeds.device.type != 'cpu' or embeds.shape[0] != 1:
            raise ValueError('only native single-row text conditioning is supported')
        if not torch.isfinite(embeds).all():
            raise ValueError('nonfinite native CPU conditioning')
        original = {'prompt_embeds': embeds.detach().clone(),
                    'prompt_embeds_mask': None if mask is None else mask.detach().clone(),
                    'image_pad_mask': image_mask.detach().clone()}
        transport = {k: v.to(torch.bfloat16) if v is not None and v.is_floating_point() else v
                     for k, v in original.items()}
        self.check()
        return dict(prompt=prompt, token_ids=ids, original=original, transport=transport,
                    original_identity=encoding_identity(original), transport_identity=encoding_identity(transport),
                    weights=self.weights, numerical=self.numerical)

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.pipe.text_encoder = None
        gc.collect()
