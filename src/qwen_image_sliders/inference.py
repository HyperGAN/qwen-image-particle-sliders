"""Native-only Qwen inference: no teachers, rows, critic, optimizer or game."""

from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Capabilities

from .lifecycle import (
    _base_precision, _observation, _render_native, _runtime_cleanup, _validate_render, _validate_runtime,
)

PROTOCOL = 'qwen-image21-native-t2i-v1'
_LOCAL = {'model_dir', 'cache_dir', 'device', 'base_precision'}


def capabilities(configuration):
    return Capabilities('routed_particle_gan', render=True,
        base_precisions=(configuration.get('base_precision', 'float32'),),
        render_outputs=('latent',) if configuration.get('dummy') else ('pil', 'latent'),
        operations=('render',))


def _options(options, spec):
    profile = spec.configuration
    expected = {'dummy', 'base_precision'} | ({'fixture_seed'} if profile.get('dummy') else set())
    if set(profile) != expected or type(profile.get('dummy')) is not bool:
        raise ValueError('invalid Qwen native inference profile')
    if profile['base_precision'] not in ('float32', 'bfloat16'):
        raise ValueError('unsupported Qwen native precision profile')
    if profile['dummy'] and (type(profile['fixture_seed']) is not int or
                             not 0 <= profile['fixture_seed'] < 2**63 - 100):
        raise ValueError('invalid Qwen fixture identity')
    return {**options, 'dummy': profile['dummy'], 'base_precision': profile['base_precision']}


def validate(options, spec):
    import re
    from pathlib import Path
    if set(options) - _LOCAL:
        raise ValueError('native inference accepts local runtime options only; no training rows or cache')
    local = _options(options, spec)
    device = local.get('device', 'cpu')
    if not isinstance(device, str) or re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device) is None:
        raise ValueError('Qwen supports cpu or an explicitly selected cuda device')
    if options.get('base_precision', local['base_precision']) != local['base_precision']:
        raise ValueError('local precision differs from exported Qwen native profile')
    if _base_precision(local) != local['base_precision']:
        raise ValueError('select a local device matching the exported Qwen base precision')
    if local['dummy']:
        if device != 'cpu' or local['base_precision'] != 'float32' or {'model_dir', 'cache_dir'} & options.keys():
            raise ValueError('Qwen fixture inference uses CPU FP32 and no model paths')
    elif not isinstance(local.get('model_dir'), str) or not local['model_dir'].strip():
        raise ValueError('Qwen native inference requires a staged local model_dir')
    elif not Path(local['model_dir']).is_dir():
        raise ValueError('Qwen native model_dir is not a staged local directory')
    for key in ('model_dir', 'cache_dir'):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(key + ' must be a nonempty local path')


def preflight(options, request, spec):
    _validate_render(request['native'], dummy=spec.configuration['dummy'])


def specification(native, runtime, *, dummy):
    profile = dict(dummy=dummy, base_precision=str(runtime.model.dtype).removeprefix('torch.'))
    if dummy:
        profile['fixture_seed'] = runtime.base_identity['fixture_seed']
    return INFERENCE_BACKEND.specification(native, model_identity=_model_identity(runtime),
        protocol=PROTOCOL, configuration=profile)


def _model_identity(runtime):
    import inspect
    from .contracts import file_digest
    components = [runtime.pipe, runtime.pipe.scheduler,
                  getattr(runtime.pipe, 'text_encoder', None), getattr(runtime.pipe, 'vae', None)]
    return {**runtime.base_identity,
            'native_helpers_sha256': file_digest(inspect.getsourcefile(_render_native)),
            'native_sampler_sources': {
                type(component).__module__: file_digest(inspect.getsourcefile(type(component)))
                for component in components if component is not None}}


def prepare(options, spec):
    import torch
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter
    from .adapters import AdapterBank
    from .contracts import formulation

    validate(options, spec)
    local = _options(options, spec)
    if local['dummy']:
        from .tiny import TinyRuntime
        runtime = TinyRuntime(spec.configuration['fixture_seed'])
    else:
        from .runtime import Runtime
        runtime = Runtime(allow_hub=False,
            **{key: value for key, value in local.items() if key in {'model_dir', 'cache_dir', 'device'}})
    _validate_runtime(runtime, local)
    restore = _runtime_cleanup(runtime)
    bank = None
    try:
        stamp, _ = formulation()
        bank = AdapterBank(runtime.model, stamp)
        bank.to(runtime.device)
        # Values are validated and replaced atomically by open_inference.
        particles = torch.nn.Parameter(torch.zeros(*bank.particle_shape, device=runtime.device))
        native = QwenImage21ResponseAdapter(runtime, bank, particles, model_identity=runtime.base_identity)
        identity = native_identity(native, model_identity=_model_identity(runtime), protocol=PROTOCOL)
    except BaseException:
        try:
            if bank is not None:
                bank.close()
        finally:
            restore()
        raise

    def close():
        try:
            bank.close()
        finally:
            restore()

    def render(*, prompt, strength, seed, images=None, **request):
        defaults = dict(height=32 if local['dummy'] else 512, width=32 if local['dummy'] else 512,
                        steps=4 if local['dummy'] else 40,
                        output_type='latent' if local['dummy'] else 'pil')
        return _render_native(runtime, native.network, defaults,
            prompt=prompt, strength=strength, seed=seed, images=images, **request)

    return InferenceBinding(native.network, identity, close, render=render,
                            observation=lambda: _observation(runtime))


INFERENCE_BACKEND = InferenceBackend('qwen', capabilities({}), prepare, validate,
    response=False, configure_capabilities=capabilities, preflight=preflight)
