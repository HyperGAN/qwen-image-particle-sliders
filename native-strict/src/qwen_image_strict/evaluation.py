"""Native observation splits are captured from declared texts, never injected tensors."""
from copy import deepcopy


def settings(options, evaluation_pairs):
    """Validate allocation-free inputs; all geometry and sampler budgets inherit."""
    if evaluation_pairs is None:
        return {}
    if not isinstance(evaluation_pairs, dict):
        raise ValueError('evaluation_pairs must map non-train split names to declared prompt pairs and seed')
    declared = options.get('declared_prompts', ())
    result = {}
    for name, value in evaluation_pairs.items():
        if not isinstance(name, str) or not name or name == 'train':
            raise ValueError('evaluation split needs a non-train name')
        if not isinstance(value, dict) or set(value) != {'rows', 'seed'}:
            raise ValueError('evaluation split accepts only rows and seed; no tensors, identity or budget override')
        if type(value['seed']) is not int or not 0 <= value['seed'] < 2**63-100:
            raise ValueError('evaluation seed must be a bounded integer')
        rows = value['rows']
        if not isinstance(rows, list) or not rows:
            raise ValueError('evaluation rows must be nonempty declared prompt pairs')
        if len(rows)*options.get('records_per_pair',2)>128:
            raise ValueError('evaluation cache exceeds128 native records')
        for row in rows:
            if (not isinstance(row, dict) or set(row)-{'neutral', 'positive', 'control'}
                    or not {'neutral', 'positive'} <= set(row)
                    or any(not isinstance(row[key], str) or row[key] not in declared for key in ('neutral', 'positive'))
                    or 'control' in row and type(row['control']) is not bool
                    or row.get('control', False) and row['neutral'] != row['positive']):
                raise ValueError('evaluation conditioning must come from this runtime own declared prompts')
        result[name] = dict(rows=deepcopy(rows), seed=value['seed'],
            **{key:options.get(key, default) for key, default in dict(
                height=512, width=512, trajectory_steps=8, records_per_pair=2).items()})
    return result


def prepare(runtime, adapter, options):
    """Prepare fresh native teachers and derive identity from actual tensor bytes."""
    from particle_sliders.lifecycle import EvaluationRows
    from particle_sliders.pg09 import _hash
    from qwen_image_sliders.train import response_rows
    from .cache import capture, capture_options, identity
    from .native import observation
    from .cleanup import release_tensors
    import torch
    from diffusers import QwenImage21Pipeline
    splits = {}
    active=[]
    try:
        with observation(runtime):
            for name, selected in options.items():
                records = capture(runtime, selected)
                active=[]
                for record,row in zip(records,response_rows(runtime, records)):
                    actual=adapter.prepare(row)
                    active.append(actual)
                    for key,value in (('neutral',actual.neutral),('target',actual.target)):
                        expected=QwenImage21Pipeline._unpack_latents(record[key],selected['height'],
                            selected['width'],runtime.pipe.vae_scale_factor)
                        if not torch.equal(value.detach().cpu(),expected):
                            raise ValueError('fresh native evaluation teacher/OFF path differs: '+key)
                prepared=tuple(active)
                splits[name] = EvaluationRows(prepared, dict(
                    protocol='qwen-native-fresh-declared-evaluation-v1', base_identity=runtime.base_identity,
                    capture_options=capture_options(selected), records=identity(records), prepared_sha256=_hash(prepared)))
    except BaseException:
        release_tensors((splits,active))
        raise
    return splits
