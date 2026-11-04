"""Native positioned-read loading; avoid retaining every source shard mapping."""
from contextlib import ExitStack
import json
from pathlib import Path

def load_encoder(directory, *, dtype, guard):
    from safetensors import safe_open
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from .execution import initialization
    directory=Path(directory)
    config=Qwen3VLConfig.from_pretrained(directory,local_files_only=True)
    config._name_or_path=str(directory)
    index=directory/'model.safetensors.index.json'
    files=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else [p.name for p in sorted(directory.glob('*.safetensors'))]
    if not files or any(Path(name).name!=name for name in files):raise ValueError('bounded local encoder checkpoint required')
    with ExitStack() as stack:
        state={}
        for name in files:
            handle=stack.enter_context(safe_open(directory/name,framework='pt',device='cpu',backend='pread'))
            for key in handle.keys():
                if key in state:raise ValueError('duplicate native encoder tensor')
                state[key]=handle.get_slice(key)
        guard()
        with initialization():
            # Exact native public loader, config and conversion machinery. The
            # values remain lazy PySafeSlice objects until the native loader
            # materializes them. No custom parameter assignment/math is used.
            model,info=Qwen3VLForConditionalGeneration.from_pretrained(None,config=config,state_dict=state,
                dtype=dtype,device_map='cpu',attn_implementation='eager',local_files_only=True,
                low_cpu_mem_usage=True,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs','conversion_errors')):
            raise ValueError('incomplete native CPU encoder load: '+str(info))
    return model
