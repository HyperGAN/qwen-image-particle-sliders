"""Caller-owned strict math policy and CPU/owned-CUDA initialization only."""
from contextlib import contextmanager
import os,uuid
from . import PROFILES
LOADER_SEED=41

def policy(*,conditioning_precision):
    if conditioning_precision not in PROFILES:raise ValueError("explicit separate conditioning precision required")
    import torch
    from diffusers.models import attention_dispatch as dispatch
    active,fn=dispatch._AttentionBackendRegistry.get_active_backend()
    flags=dict(deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_precision=torch.get_float32_matmul_precision(),TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        fp16_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),math_low_precision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),CPU_threads=torch.get_num_threads(),diffusers_backend=active.value)
    wanted=dict(deterministic=True,warn_only=False,matmul_precision='highest',TF32=False,cudnn_TF32=False,cudnn_benchmark=False,cudnn_deterministic=True,
        fp16_reduction=False,bf16_reduction=False,flash_sdp=False,efficient_sdp=False,cudnn_sdp=False,math_sdp=True,math_low_precision=False,
        cublas=':4096:8',CPU_threads=4,diffusers_backend='native')
    if flags!=wanted or fn is not dispatch._native_attention:
        raise ValueError('Qwen strict numerical policy/actual native SDPA dispatch differs')
    env={k:os.environ.get(k) for k in ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVIDIA_TF32_OVERRIDE','DIFFUSERS_ATTN_BACKEND']}
    if env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] not in (None,'0') or env['NVIDIA_TF32_OVERRIDE'] not in (None,'0') or env['DIFFUSERS_ATTN_BACKEND'] not in (None,'native'):
        raise ValueError('Qwen strict numerical environment override differs')
    return dict(profile=PROFILES[conditioning_precision],conditioning_precision=conditioning_precision,flags=flags,environment=env,conditioning='CPU '+conditioning_precision+' native fullprefix, explicit BF16transport; no padding between rows',
        transformer='ownedCUDA0 BF16',decoder='ownedCUDA0 BF16 from original FP32 weights',CPU_capability=torch.backends.cpu.get_cpu_capability(),maximum_pixels=512,maximum_encoder_tokens=512)

def require_policy(expected):
    if policy(conditioning_precision=expected["conditioning_precision"])!=expected:raise ValueError('Qwen strict numerical identity changed')

@contextmanager
def initialization(seed=LOADER_SEED,*,device='cpu'):
    import torch
    if device not in ('cpu','cuda:0') or type(seed) is not int or not 0<=seed<2**63-100:raise ValueError('ownedCPU/visibleCUDA0 RNG only')
    with torch.random.fork_rng(devices=[0] if device=='cuda:0' else []):
        torch.random.default_generator.manual_seed(seed)
        if device=='cuda:0':torch.cuda.default_generators[0].manual_seed(seed)
        yield

def canonical_uuid(value):
    if not isinstance(value,str):raise ValueError('explicit GPU UUID required')
    try:return 'GPU-'+str(uuid.UUID(value.removeprefix('GPU-')))
    except ValueError as error:raise ValueError('invalid GPU UUID') from error

def admission(expected):
    import torch
    expected=canonical_uuid(expected)
    if os.environ.get('QWEN_STRICT_GUARDIAN')!=str(os.getppid()) or os.environ.get('QWEN_STRICT_PROCESS_CAP_GIB')!='24' or canonical_uuid(os.environ.get('CUDA_VISIBLE_DEVICES'))!=expected:
        raise ValueError('strict Qwen needs admitted24GiB guardian and exactvisibleUUID')
    if torch.cuda.device_count()!=1 or canonical_uuid(str(torch.cuda.get_device_properties(0).uuid))!=expected:raise ValueError('actual CUDA UUID differs')
    if torch.cuda.mem_get_info(0)[0]<29*2**30:raise MemoryError('Qwen proposed24GiB cap plus5GiB floor unavailable')

def require_host(floor_gib):
    from pathlib import Path
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available<floor_gib*2**30:raise MemoryError('Qwen CPU staging host reserve unavailable')

def lifetime_guard():
    """Refuse a missing/replaced guardian throughout native construction/forwards."""
    from pathlib import Path
    parent=os.getppid()
    try:
        ticks=Path('/proc',str(parent),'stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,IndexError):raise RuntimeError('Qwen guardian disappeared')
    if os.environ.get('QWEN_STRICT_GUARDIAN')!=str(parent) or os.environ.get('QWEN_STRICT_GUARDIAN_START')!=ticks:
        raise RuntimeError('Qwen guardian parent/start identity differs')
    require_host(24)

def prompts(values):
    if not isinstance(values,list) or not values or any(not isinstance(p,str) or not p.strip() for p in values) or len(set(values))!=len(values):raise ValueError('explicit unique nonempty declared prompts required')
    return values

def pixels(value):
    if type(value) is not int or not 32<=value<=512 or value%32:raise ValueError('strict Qwen pixels must be multiplesof32 through512')
