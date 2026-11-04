"""Explicit CPU encoder followed by owned CUDA BF16 DiT+VAE.

No automatic fallback, download, offload, quantization or attention replacement.
Full pretrained execution and the proposed memory cap remain unqualified.
"""
from contextlib import contextmanager
import gc
from pathlib import Path

from .assets import inspect_assets
from .conditioning import CPUConditioner, verify_parameters, encoding_identity, frozen_state
from .execution import admission, initialization, pixels, prompts, require_host, require_policy
from .provenance import execution_identity


def require_live(runtime):
    import torch
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor
    if runtime.closed:
        raise ValueError('strict Qwen runtime is closed')
    require_policy(runtime.strict_execution['numerical'])
    if runtime.guard is not None:
        runtime.guard()
    for root in (runtime.model, runtime.pipe.vae):
        if any(p.device != runtime.device or p.dtype != torch.bfloat16 or p.requires_grad for p in root.parameters()):
            raise ValueError('actual native CUDA BF16/frozen base changed')
    if any(type(module.processor) is not QwenImage21AttnProcessor for module in runtime.model.modules()
           if hasattr(module, 'processor')):
        raise ValueError('actual native attention processor changed')
    if runtime.pipe._execution_device != runtime.device:
        raise ValueError('native pipeline execution device differs')


@contextmanager
def observation(runtime):
    from qwen_image_sliders.lifecycle import _observation
    require_live(runtime)
    with _observation(runtime):
        yield
    require_live(runtime)


class StrictRuntime:
    validation_kind = 'canonical-native-staged-strict-candidate'

    def __init__(self, snapshot, *, conditioning_precision, declared_prompts, expected_gpu_uuid, guard):
        import torch
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21
        if guard is None or not callable(guard):
            raise ValueError('strict native construction needs an active resource/lifetime guardian')
        self.guard, self.closed, self.handles, self.pipe = guard, False, [], None
        self.device = torch.device('cuda:0')
        declared_prompts = prompts(declared_prompts)
        # Exact source, full selected asset hashes and CPU policy precede both
        # encoder and pretrained CUDA allocation. The shared session may already
        # own a small CUDA RNG context; CPU initialization never reseeds it.
        self.strict_execution = execution_identity(conditioning_precision=conditioning_precision)
        self.strict_assets = inspect_assets(snapshot, full_hashes=True)
        numerical = self.strict_execution['numerical']
        encoder = None
        try:
            encoder = CPUConditioner.load_local(snapshot, numerical=numerical, guard=guard, cpu_only=False)
            self.conditioning = {text: encoder.encode(text) for text in declared_prompts}
            self.conditioning_identity = {text: dict(token_ids=row['token_ids'].tolist(),
                original=row['original_identity'], transport=row['transport_identity'])
                for text, row in self.conditioning.items()}
            self.encoder_weights = encoder.weights
            processor, scheduler = encoder.pipe.processor, encoder.pipe.scheduler
            encoder.close()
            del encoder
            encoder = None
            gc.collect()
            require_host(24)
            guard()
            admission(expected_gpu_uuid)
            directory = Path(snapshot)
            # Own only CUDA0 RNG; restore caller CPU/selected CUDA state on error.
            with torch.cuda.device(0), initialization(device='cuda:0'):
                model = QwenImage21Transformer2DModel.from_pretrained(directory/'transformer',
                    torch_dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True, low_cpu_mem_usage=True)
                model.eval().requires_grad_(False)
                self._partial_model = model
                self.transformer_weights = verify_parameters(model, directory/'transformer', source_dtype=torch.bfloat16,
                    loaded_dtype=torch.bfloat16, device='cuda:0')
                vae = AutoencoderKLQwenImage21.from_pretrained(directory/'vae',torch_dtype=torch.bfloat16,
                    device_map='cuda:0',local_files_only=True,low_cpu_mem_usage=True)
                vae.eval().requires_grad_(False)
                self._partial_vae = vae
                self.vae_weights = verify_parameters(vae, directory/'vae',source_dtype=torch.float32,
                    loaded_dtype=torch.bfloat16,device='cuda:0')
                self.pipe = QwenImage21Pipeline(vae=vae, text_encoder=None, processor=processor,
                    transformer=model, scheduler=scheduler)
                self.pipe.set_progress_bar_config(disable=True)
                if self.pipe.config.sample_sigmas is not None:
                    raise ValueError('base native scheduler required')
            self.base_identity = dict(execution=self.strict_execution, assets=self.strict_assets,
                conditioning=self.conditioning_identity, encoder_weights=self.encoder_weights,
                transformer_weights=self.transformer_weights, vae_weights=self.vae_weights,
                native_geometry=dict(latent_channels=64,spatial_compression=16,maximum_pixels=512,
                                     conditioning_width=4096,one_native_prefix_per_row=True))
            require_live(self)
            for root in (self.model, self.pipe.vae, *self.model.transformer_blocks):
                self.handles.append(root.register_forward_pre_hook(lambda *_: require_live(self)))
        except BaseException:
            if encoder is not None:
                encoder.close()
            self.strict_close()
            raise

    @property
    def model(self):
        return self.pipe.transformer

    def encode(self, prompt):
        require_live(self)
        if prompt not in self.conditioning:
            raise ValueError('prompt was not independently encoded in the declared native cohort')
        row=self.conditioning[prompt]
        if encoding_identity(row['transport']) != row['transport_identity']:
            raise ValueError('actual conditioning transport changed')
        return {key:None if value is None else value.to(self.device) for key,value in row['transport'].items()}

    def sample(self, encoding, *, seed, height, width, steps, output_type='latent'):
        import torch
        require_live(self)
        pixels(height); pixels(width)
        if type(seed) is not int or not 0<=seed<2**63-100 or type(steps) is not int or not 1<=steps<=40:
            raise ValueError('bounded native sampler seed/steps required')
        if output_type not in ('latent','pil'):
            raise ValueError('native latent or RGBA PIL output only')
        with observation(self):
            return self.pipe(prompt_embeds=encoding['prompt_embeds'],prompt_embeds_mask=encoding['prompt_embeds_mask'],
                height=height,width=width,num_inference_steps=steps,true_cfg_scale=1.0,use_kv_cache=False,
                output_type=output_type,generator=torch.Generator('cpu').manual_seed(seed)).images

    def frozen_identity(self):
        require_live(self)
        return dict(transformer=frozen_state(self.model),vae=frozen_state(self.pipe.vae))

    def strict_close(self, *, game=None, bank=None, network=None):
        import torch
        if self.closed:
            return
        self.closed=True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if game is not None:
            bank=game.backend.adapter
        if bank is not None:
            bank.close()
        roots=[getattr(self,'_partial_model',None),getattr(self,'_partial_vae',None)]
        if self.pipe is not None:
            roots.extend((self.pipe.transformer,self.pipe.vae))
        if game is not None:
            roots.extend((game.network,game.critic))
        roots.extend((bank,network))
        with torch.no_grad():
            for root in {id(x):x for x in roots if isinstance(x,torch.nn.Module)}.values():
                root.to('cpu')
                for parameter in root.parameters():
                    parameter.grad=None
            if game is not None:
                # Optimizer states retain CUDA tensors even after modules move.
                for value in vars(game).values():
                    if isinstance(value,torch.optim.Optimizer):
                        for state in value.state.values():
                            for key,tensor in state.items():
                                if isinstance(tensor,torch.Tensor):state[key]=tensor.cpu()
            for root in roots:
                rope=getattr(root,'pos_embed',None)
                if rope is not None and hasattr(rope,'freqs'):
                    rope.freqs=[value.cpu() if isinstance(value,torch.Tensor) else value for value in rope.freqs]
        self.pipe=None
        self._partial_model=self._partial_vae=None
        gc.collect()
        # No empty_cache/set_device: unrelated caller-owned CUDA state is untouched.
