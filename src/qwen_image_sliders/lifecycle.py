"""Qwen factory; native 2.1 packing, conditioning and sampling stay here."""
from copy import deepcopy
from contextlib import contextmanager
import re

from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup

_TRAIN = {'rows', 'steps', 'seed', 'batch', 'height', 'width', 'trajectory_steps',
          'records_per_pair', 'g_lr', 'preset', 'critic_kind', 'gradient_checkpointing'}
_RUNTIME = {'model_dir', 'cache_dir', 'device'}


@contextmanager
def _sampling(runtime):
    """Preserve the pinned native pipeline's transient sampling state."""
    pipe = runtime.pipe
    scheduler = pipe.scheduler
    missing = object()
    # Assigned by QwenImage21Pipeline.__call__ at the retained Diffusers pin.
    keys = ('_attention_kwargs', '_current_timestep', '_interrupt', '_num_timesteps')
    previous = {key: vars(pipe).get(key, missing) for key in keys}
    try:
        pipe.scheduler = deepcopy(scheduler)
        yield
    finally:
        pipe.scheduler = scheduler
        for key, value in previous.items():
            if value is missing:
                vars(pipe).pop(key, None)
            else:
                setattr(pipe, key, value)


def _base_precision(options):
    # Runtime loads FP32 on CPU and BF16 on CUDA; this is not a caller cast.
    device = options.get('device', 'cpu')
    return 'bfloat16' if isinstance(device, str) and device.startswith('cuda') else 'float32'


def validate(options):
    from .contracts import formulation, validate_shape
    if set(options) - (_TRAIN | _RUNTIME | {'dummy', 'base_precision', 'ema_decay'}):
        raise ValueError('unknown Qwen session option')
    if not isinstance(options.get('rows'), list) or len(options['rows']) < 2:
        raise ValueError('Qwen requires at least two paired prompt rows')
    if any(not isinstance(row, dict) or not all(isinstance(row.get(key), str) and row[key].strip()
                   for key in ('neutral', 'positive')) for row in options['rows']):
        raise ValueError('every Qwen pair requires nonempty neutral and positive text')
    for key in ('dummy', 'gradient_checkpointing'):
        if type(options.get(key, False)) is not bool:
            raise ValueError(key + ' must be a boolean')
    if any(type(row.get('control', False)) is not bool for row in options['rows']):
        raise ValueError('Qwen control flags must be boolean')
    device = options.get('device', 'cpu')
    if not isinstance(device, str) or re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device) is None:
        raise ValueError('Qwen supports cpu or an explicitly selected cuda device')
    for key in ('model_dir', 'cache_dir'):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(key + ' must be a nonempty local path')
    trajectory, records = options.get('trajectory_steps', 4), options.get('records_per_pair', 2)
    if (type(trajectory) is not int or type(records) is not int or not 1 <= records <= trajectory
            or len(options['rows']) * records > 128):
        raise ValueError('invalid Qwen teacher record budget')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported Qwen PG preset')
    if any(type(options.get(key, 32)) is not int for key in ('height', 'width')):
        raise ValueError('Qwen height and width must be integers')
    validate_shape(options.get('height', 32), options.get('width', 32))
    formulation(batch=options.get('batch', 2), g_lr=options.get('g_lr'))
    if options.get('critic_kind', 'readout') not in ('readout', 'hndl'):
        raise ValueError('unsupported Qwen critic')
    if options.get('base_precision', _base_precision(options)) != _base_precision(options):
        raise ValueError('Qwen base precision is determined by the selected runtime device')
    if options.get('dummy') and (device != 'cpu' or any(key in options for key in ('model_dir', 'cache_dir'))):
        raise ValueError('Qwen dummy is CPU float32 only and accepts no model paths')


def prepare(options, *, runtime=None):
    from .contracts import formulation
    from .teachers import paired_records
    from .train import build_game
    validate(options)
    seed = options.get('seed', 7)
    if runtime is None:
        if options.get('dummy'):
            from .tiny import TinyRuntime
            runtime = TinyRuntime(seed)
        else:
            from .runtime import Runtime
            runtime = Runtime(allow_hub=False, **{key: value for key, value in options.items() if key in _RUNTIME})
    import torch
    requested_device = torch.device(options.get('device', 'cpu'))
    is_dummy = runtime.validation_kind == 'tiny-native-random-weights-synthetic-text'
    if is_dummy != options.get('dummy', False):
        raise ValueError('staged Qwen runtime kind differs from the declared dummy/native contract')
    if (runtime.device.type != requested_device.type or
            requested_device.index is not None and runtime.device.index != requested_device.index or
            str(runtime.model.dtype) != 'torch.' + _base_precision(options)):
        raise ValueError('staged Qwen runtime device or base precision differs from configuration')
    if any(parameter.requires_grad for parameter in runtime.model.parameters()):
        raise ValueError('staged Qwen backbone must already be frozen')
    restore = checkpoint_cleanup(runtime.model)
    try:
        # Teacher capture runs the native pipeline, which mutates scheduler state.
        # A staged caller retains its original scheduler even on capture failure.
        with _sampling(runtime):
            records = paired_records(runtime, options['rows'], seed=seed,
                **{key: options.get(key, default) for key, default in
                   [('height', 32), ('width', 32), ('trajectory_steps', 4), ('records_per_pair', 2)]})
        stamp, declared = formulation(batch=options.get('batch', 2), g_lr=options.get('g_lr'))
        if options.get('gradient_checkpointing', False):
            runtime.model.enable_gradient_checkpointing()
        game = build_game(runtime, records, stamp, declared, seed,
            **{key: value for key, value in options.items() if key in {'steps', 'preset', 'g_lr', 'critic_kind'}})
    except BaseException:
        restore()
        raise
    def close():
        try:
            game.backend.adapter.close()
        finally:
            restore()

    def render(*, prompt, strength, seed, images, **native):
        from .contracts import validate_shape
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Qwen rendering requires a nonempty prompt')
        if images is not None:
            raise ValueError('Qwen reference-image conditioning is unsupported')
        if set(native) - {'height', 'width', 'steps', 'output_type'}:
            raise ValueError('unknown Qwen native render option')
        native = {'height': options.get('height', 32), 'width': options.get('width', 32),
                  'steps': options.get('trajectory_steps', 4),
                  'output_type': 'latent' if options.get('dummy') else 'pil', **native}
        if any(type(native[key]) is not int for key in ('height', 'width', 'steps')) or native['steps'] < 1:
            raise ValueError('Qwen render geometry and steps require positive integers')
        validate_shape(native['height'], native['width'])
        if type(seed) is not int or not 0 <= seed < 2**63 - 100:
            raise ValueError('invalid Qwen render seed')
        with _sampling(runtime), game.network.scaled(strength):
            return runtime.sample(runtime.encode(prompt), seed=seed, **native)

    return Binding(game, close, render)


def capabilities(options):
    return Capabilities('routed_particle_gan', render=True,
        base_precisions=(_base_precision(options),),
        render_outputs=('latent',) if options.get('dummy') else ('pil', 'latent'))


BACKEND = Backend('qwen', capabilities({}), prepare, validate, capabilities)
