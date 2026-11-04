"""Exact current native teacher identity; no external response-scale override."""
import hashlib
import json

def tree_identity(value):
    import torch
    from .conditioning import encoding_identity, tensor_digest
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_digest(value))
    if isinstance(value,dict):return {k:tree_identity(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tree_identity(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise ValueError('unsupported native cache value')

def identity(records):
    data=tree_identity(records)
    return dict(records=len(records),sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest())

def capture(runtime,options):
    from qwen_image_sliders.teachers import paired_records
    from .native import observation
    with observation(runtime):
        return paired_records(runtime,options['rows'],seed=options.get('seed',7),height=options.get('height',512),
            width=options.get('width',512),trajectory_steps=options.get('trajectory_steps',8),
            records_per_pair=options.get('records_per_pair',2))

def current_records(runtime,options):
    """Recapture current source and compare every byte before using supplied cache."""
    import torch
    records=capture(runtime,options)
    actual=identity(records)
    if options.get('cache_file') is not None:
        packet=torch.load(options['cache_file'],map_location='cpu',weights_only=True)
        if (set(packet)!={'schema','base_identity','options','records','identity'}
                or packet['schema']!='qwen-native-strict-current-cache-v1'
                or packet['base_identity']!=runtime.base_identity
                or packet['options']!=capture_options(options)
                or packet['identity']!=actual or identity(packet['records'])!=actual):
            raise ValueError('cache differs from actual current native trajectory/source/conditioning')
    return records,actual

def capture_options(options):
    return {key:options.get(key,default) for key,default in dict(rows=None,seed=7,height=512,width=512,
        trajectory_steps=8,records_per_pair=2).items()}

def packet(runtime,options,records):
    return dict(schema='qwen-native-strict-current-cache-v1',base_identity=runtime.base_identity,
                options=capture_options(options),records=records,identity=identity(records))
