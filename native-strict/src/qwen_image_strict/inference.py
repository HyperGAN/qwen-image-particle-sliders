"""Strict native inference constructs no rows, teachers, critic, optimizer or game."""
from particle_sliders.inference import InferenceBackend,InferenceBinding,native_identity
from particle_sliders.lifecycle import Capabilities
from . import BACKEND_NAME,PROFILES
from .execution import canonical_uuid,pixels,prompts,lifetime_guard
PROTOCOL='qwen-image21-cpu-encoder-cuda-dit-vae-strict-native-v1'
CAPABILITIES=Capabilities('routed_particle_gan',render=True,base_precisions=('bfloat16',),render_outputs=('latent','pil'),
    operations=('response','render'),qualification='Separate strict source candidate; full pretrained inference unqualified')

def check_request(options,request):
    import torch
    native=request['native']
    if request['operation']=='response':
        if set(native)-{'prompt','latent','timestep'}:raise ValueError('unknown strict Qwen numerical response field')
        if 'prompt' not in native or native['prompt'] not in options.get('declared_prompts',[]):raise ValueError('declare the encoded response prompt')
        value=native.get('latent')
        if (not isinstance(value,torch.Tensor) or value.ndim!=5 or tuple(value.shape[:3])!=(1,64,1)
                or not 2<=value.shape[-2]<=32 or not 2<=value.shape[-1]<=32
                or value.shape[-2]%2 or value.shape[-1]%2 or not torch.isfinite(value).all()):
            raise ValueError('strict native response requires finite [1,64,1,H,W], even latent axes2..32')
        time=native.get('timestep')
        if not isinstance(time,torch.Tensor) or time.numel()!=1 or not torch.isfinite(time).all() or not 0<=float(time)<=1000:
            raise ValueError('strict native response requires one raw scheduler timestep0..1000')
    elif request['operation']=='render':
        from qwen_image_sliders.lifecycle import _validate_render
        _validate_render(native)
        if native.get('prompt') is not None and native['prompt'] not in options.get('declared_prompts',[]):
            raise ValueError('render prompt not declared for CPU encoding')
        pixels(native.get('height',512));pixels(native.get('width',512))

def specification(adapter,runtime,options):
    return BACKEND.specification(adapter,model_identity=runtime.base_identity,protocol=PROTOCOL,
        configuration=dict(execution=runtime.strict_execution,assets=runtime.strict_assets,
                           declared_prompts=options['declared_prompts'],base_precision='bfloat16'))

def render(runtime,network,options,*,prompt,strength,seed,images=None,**native):
    from .native import observation
    if images is not None:raise ValueError('reference images are unsupported')
    defaults=dict(height=options.get('height',512),width=options.get('width',512),
                  steps=options.get('trajectory_steps',8),output_type='pil')
    defaults.update(native)
    check_request(dict(declared_prompts=runtime.conditioning),dict(operation='render',native=dict(defaults,prompt=prompt)))
    with observation(runtime),network.scaled(strength):
        return runtime.sample(runtime.encode(prompt),seed=seed,**defaults)

def response(runtime,adapter,*,strength,prompt,latent,timestep,seed=0,**unused):
    import torch
    from .native import observation
    from qwen_image_sliders.train import _condition
    if type(seed) is not int or not 0<=seed<2**63-100:raise ValueError('bounded native response seed required')
    if unused:raise ValueError('unknown numerical response argument')
    check_request(dict(declared_prompts=runtime.conditioning),dict(operation='response',native=dict(prompt=prompt,latent=latent,timestep=timestep)))
    with observation(runtime),torch.no_grad(),adapter.network.scaled(strength):
        return adapter._velocity(_condition(runtime.encode(prompt)),latent.to(runtime.device),timestep.to(runtime.device))

def validate(options,spec):
    from .assets import inspect_assets
    from .provenance import execution_identity
    if set(options)-{'model_dir','expected_gpu_uuid'}:raise ValueError('strict inference accepts only local assets and GPU UUID')
    profile=spec.configuration
    if set(profile)!={'execution','assets','declared_prompts','base_precision'} or profile['base_precision']!='bfloat16':
        raise ValueError('strict inference profile differs')
    precision=profile['execution']['numerical']['conditioning_precision']
    if precision not in PROFILES or profile['execution']!=execution_identity(conditioning_precision=precision):
        raise ValueError('strict inference executed source/numerical profile differs before allocation')
    prompts(profile['declared_prompts']);canonical_uuid(options.get('expected_gpu_uuid'))
    actual=inspect_assets(options.get('model_dir',''),full_hashes=False);actual['full_weight_hashes_verified']=True
    if profile['assets']!=actual or spec.identity['protocol']!=PROTOCOL:
        raise ValueError('strict inference original assets/protocol differs before allocation')
    if spec.identity['model'].get('execution')!=profile['execution'] or spec.identity['model'].get('assets')!=profile['assets']:
        raise ValueError('strict inference base identity differs before allocation')
    if set(spec.identity['model'].get('conditioning',{}))!=set(profile['declared_prompts']):
        raise ValueError('strict inference declared conditioning differs before allocation')

def preflight(options,request,spec):
    check_request(spec.configuration,request)

def prepare(options,spec):
    import torch
    from qwen_image_sliders.adapters import AdapterBank
    from qwen_image_sliders.contracts import formulation
    from particle_sliders.pg09_images import QwenImage21ResponseAdapter
    from .native import StrictRuntime,observation
    from .execution import initialization
    validate(options,spec)
    runtime=StrictRuntime(options['model_dir'],conditioning_precision=spec.configuration['execution']['numerical']['conditioning_precision'],
        declared_prompts=spec.configuration['declared_prompts'],expected_gpu_uuid=options['expected_gpu_uuid'],guard=lifetime_guard)
    bank=None
    try:
        with initialization(7,device=str(runtime.device)):
            stamp,_=formulation()
            bank=AdapterBank(runtime.model,stamp).to(runtime.device)
            particles=torch.nn.Parameter(torch.zeros(*bank.particle_shape,device=runtime.device))
            adapter=QwenImage21ResponseAdapter(runtime,bank,particles,model_identity=runtime.base_identity)
            identity=native_identity(adapter,model_identity=runtime.base_identity,protocol=PROTOCOL)
        return InferenceBinding(adapter.network,identity,lambda:runtime.strict_close(bank=bank,network=adapter.network),
            render=lambda **kw:render(runtime,adapter.network,{},**kw),response=lambda **kw:response(runtime,adapter,**kw),
            observation=lambda:observation(runtime))
    except BaseException:
        runtime.strict_close(bank=bank)
        raise

BACKEND=InferenceBackend(BACKEND_NAME,CAPABILITIES,prepare,validate,response=True,preflight=preflight)
