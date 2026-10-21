"""Qwen factory; native 2.1 packing, conditioning and sampling stay here."""
from copy import deepcopy
from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup

_TRAIN = {'rows', 'steps', 'seed', 'batch', 'height', 'width', 'trajectory_steps',
          'records_per_pair', 'g_lr', 'preset', 'critic_kind'}
_RUNTIME = {'model_dir', 'cache_dir', 'device'}


def validate(options):
    from .contracts import formulation, validate_shape
    if set(options) - (_TRAIN | _RUNTIME | {'dummy', 'base_precision', 'ema_decay'}):
        raise ValueError('unknown Qwen session option')
    if not isinstance(options.get('rows'), list) or len(options['rows']) < 2:
        raise ValueError('Qwen requires at least two paired prompt rows')
    if any(not all(isinstance(row.get(key), str) and row[key].strip()
                   for key in ('neutral', 'positive')) for row in options['rows']):
        raise ValueError('every Qwen pair requires nonempty neutral and positive text')
    trajectory, records = options.get('trajectory_steps', 4), options.get('records_per_pair', 2)
    if (type(trajectory) is not int or type(records) is not int or not 1 <= records <= trajectory
            or len(options['rows']) * records > 128):
        raise ValueError('invalid Qwen teacher record budget')
    if options.get('preset', 'bispf_optimized') not in ('bispf', 'bispf_optimized'):
        raise ValueError('unsupported Qwen PG preset')
    validate_shape(options.get('height', 32), options.get('width', 32))
    formulation(batch=options.get('batch', 2), g_lr=options.get('g_lr'))
    if options.get('critic_kind', 'readout') not in ('readout', 'hndl'):
        raise ValueError('unsupported Qwen critic')
    if options.get('dummy') and (options.get('device', 'cpu') != 'cpu' or
                                 options.get('base_precision', 'float32') != 'float32'):
        raise ValueError('Qwen dummy is CPU float32 only')


def prepare(options, *, runtime=None):
    from .contracts import formulation
    from .teachers import paired_records
    from .train import build_game
    seed = options.get('seed', 7)
    if runtime is None:
        if options.get('dummy'):
            from .tiny import TinyRuntime
            runtime = TinyRuntime(seed)
        else:
            from .runtime import Runtime
            runtime = Runtime(allow_hub=False, **{key: value for key, value in options.items() if key in _RUNTIME})
    records = paired_records(runtime, options['rows'], seed=seed,
        **{key: options.get(key, default) for key, default in
           [('height', 32), ('width', 32), ('trajectory_steps', 4), ('records_per_pair', 2)]})
    stamp, declared = formulation(batch=options.get('batch', 2), g_lr=options.get('g_lr'))
    restore = checkpoint_cleanup(runtime.model)
    try:
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
        scheduler = runtime.pipe.scheduler
        try:
            runtime.pipe.scheduler = deepcopy(scheduler)
            with game.network.scaled(strength):
                return runtime.sample(runtime.encode(prompt), seed=seed, **native)
        finally:
            runtime.pipe.scheduler = scheduler

    return Binding(game, close, render)


def capabilities(options):
    return Capabilities('routed_particle_gan', render=True,
        render_outputs=('latent',) if options.get('dummy') else ('pil', 'latent'))


BACKEND = Backend('qwen', capabilities({}), prepare, validate, capabilities)
