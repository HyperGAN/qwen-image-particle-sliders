"""Opt-in strict source cohort; default release and old packets stay unchanged."""
from pathlib import Path
from particle_sliders.lifecycle import Backend,Binding,Capabilities,SessionConfig
from . import BACKEND_NAME,COHORT,PROFILES
from .execution import canonical_uuid,pixels,prompts,require_policy

ALLOWED={'rows','declared_prompts','conditioning_precision','model_dir','expected_gpu_uuid','cache_file',
    'steps','seed','batch','height','width','trajectory_steps','records_per_pair','lr','d_lr_mult','prior_lr_mult',
    'preset','critic_kind','gradient_checkpointing','base_precision','ema_decay','device'}

def config(*,conditioning_precision,cohort=None,**options):
    if conditioning_precision not in PROFILES:raise ValueError('explicit conditioning precision required')
    defaults=dict(steps=4,seed=7,batch=2,height=512,width=512,trajectory_steps=8,records_per_pair=2,
        preset='bispf_optimized',critic_kind='readout',gradient_checkpointing=True,device='cuda:0',
        base_precision='bfloat16',ema_decay=None,conditioning_precision=conditioning_precision)
    defaults.update(options)
    return SessionConfig(BACKEND_NAME,cohort or PROFILES[conditioning_precision],**defaults)

def validate(options):
    from qwen_image_sliders.lifecycle import validate as base_validate
    from .assets import inspect_assets
    from .provenance import execution_identity
    if set(options)-ALLOWED or options.get('device','cuda:0')!='cuda:0' or options.get('base_precision','bfloat16')!='bfloat16':
        raise ValueError('unsupported strict Qwen option/device/precision')
    precision=options.get('conditioning_precision')
    if precision not in PROFILES:raise ValueError('explicit native CPU conditioning precision required')
    local={key:value for key,value in options.items() if key not in {
        'declared_prompts','conditioning_precision','expected_gpu_uuid','cache_file'}}
    base_validate(local)
    pixels(options.get('height',512));pixels(options.get('width',512))
    if not 1<=options.get('trajectory_steps',8)<=40:raise ValueError('native trajectory budget through40 only')
    texts=prompts(options.get('declared_prompts'))
    if any(row[key] not in texts for row in options['rows'] for key in ('neutral','positive')):
        raise ValueError('all teacher prompts must be explicitly CPU encoded')
    canonical_uuid(options.get('expected_gpu_uuid'))
    inspect_assets(options.get('model_dir',''),full_hashes=False)
    if options.get('cache_file') is not None and not Path(options['cache_file']).is_file():
        raise ValueError('cache_file must be an existing current native receipt')
    execution_identity(conditioning_precision=precision)

def preflight(options,request,staged):
    if set(staged)-{'runtime','before_update','guardian'}:raise ValueError('unknown staged strict Qwen field')
    for name in ('before_update','guardian'):
        if staged.get(name) is not None and not callable(staged[name]):raise ValueError(name+' must be callable')
    from .inference import check_request
    check_request(options,request)

def prepare(options,*,runtime=None,before_update=None,guardian=None):
    from qwen_image_sliders.contracts import formulation
    from .cache import current_records
    from .game import build_game
    from .native import StrictRuntime,observation,require_live
    from .inference import specification,render,response
    validate(options)
    if runtime is None:
        runtime=StrictRuntime(options['model_dir'],conditioning_precision=options['conditioning_precision'],
            declared_prompts=options['declared_prompts'],expected_gpu_uuid=options['expected_gpu_uuid'],guard=guardian)
    game=None
    try:
        require_live(runtime)
        from .assets import inspect_assets
        from .provenance import execution_identity
        expected_assets=inspect_assets(options['model_dir'],full_hashes=False)
        expected_assets['full_weight_hashes_verified']=True
        if (runtime.strict_execution!=execution_identity(conditioning_precision=options['conditioning_precision'])
                or runtime.strict_assets!=expected_assets
                or set(runtime.conditioning)!=set(options['declared_prompts'])):
            raise ValueError('staged strict runtime source/precision/assets/declared conditioning differs')
        records,cache_identity=current_records(runtime,options)
        runtime.current_cache_identity=cache_identity
        stamp,declared=formulation(batch=options.get('batch',2),lr=options.get('lr'))
        if options.get('gradient_checkpointing',True):runtime.model.enable_gradient_checkpointing()
        game=build_game(runtime,records,stamp,declared,options.get('seed',7),steps=options.get('steps',4),
            **{key:options[key] for key in ('preset','lr','d_lr_mult','prior_lr_mult','critic_kind') if key in options})
        spec=specification(game.backend,runtime,options)
    except BaseException:
        runtime.strict_close(game=game)
        raise
    def before(context):
        require_live(runtime)
        return before_update(context) if before_update is not None else True
    return Binding(game,lambda:runtime.strict_close(game=game),render=lambda **kw:render(runtime,game.network,options,**kw),
        response=lambda **kw:response(runtime,game.backend,**kw),inference=spec,before_update=before,
        observation=lambda:observation(runtime))

CAPABILITIES=Capabilities('routed_particle_gan',render=True,base_precisions=('bfloat16',),render_outputs=('latent','pil'),
    qualification='Separate strict CPU-encoder/CUDA-DiT+VAE candidate; pretrained execution and LR unqualified')
BACKEND=Backend(BACKEND_NAME,CAPABILITIES,prepare,validate,preflight=preflight)
