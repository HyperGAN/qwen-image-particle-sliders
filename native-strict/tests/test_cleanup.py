"""Tiny actual native/game ownership tests; no pretrained payload or CUDA."""
from types import SimpleNamespace
import pytest
import torch
from qwen_image_strict.cleanup import checkpoint_restore, release_tensors
from qwen_image_strict.execution import initialization
from qwen_image_strict.native import StrictRuntime


def owned(base):
    runtime=StrictRuntime.__new__(StrictRuntime)
    runtime.device=torch.device('cpu')
    runtime.closed=False
    runtime._cleanup_complete=False
    runtime.handles=[]
    runtime.pipe=base.pipe
    runtime._partial_model=base.model
    runtime._partial_vae=base.pipe.vae
    if base.pipe.vae is not None:
        base.pipe.vae.clear_cache()
    runtime._checkpoint_restore=checkpoint_restore(base.model)
    return runtime


def test_real_native_game_close_restores_checkpoint_and_releases_retained_states(monkeypatch):
    from qwen_image_sliders.tiny import TinyRuntime
    from qwen_image_sliders.contracts import formulation
    from qwen_image_sliders.teachers import paired_records
    from qwen_image_strict.game import build_game
    import qwen_image_strict.native as native
    monkeypatch.setattr(native,'require_live',lambda _:None)
    with initialization():base=TinyRuntime(41,with_vae=True)
    model,vae=base.model,base.pipe.vae
    original_checkpoint={id(m):{k:vars(m)[k] for k in ('gradient_checkpointing','_gradient_checkpointing_func') if k in vars(m)} for m in model.modules()}
    runtime=owned(base)
    foreign_handle=model.register_forward_hook(lambda *_:None)
    owned_handle=model.register_forward_pre_hook(lambda *_:None)
    runtime.handles.append(owned_handle)
    model.enable_gradient_checkpointing()
    records=paired_records(base,[dict(neutral='blue square',positive='red square'),
        dict(neutral='blue circle',positive='red circle'),dict(neutral='forest',positive='forest',control=True)],
        seed=7,height=32,width=32,trajectory_steps=2,records_per_pair=1)
    stamp,declared=formulation(batch=2,lr=.001)
    game=build_game(base,records,stamp,declared,7,steps=2,lr=.001)
    game.step()
    # Actual public scale-captured checkpoint wrapper replaced the native hook.
    assert model.is_gradient_checkpointing and any('_gradient_checkpointing_func' in vars(m) for m in model.modules())
    held=[game.rows[0].target,game.rows[0].neutral,game.rows[0].row.latent,
          game.rows[0].row.neutral['embeds'],game.targets,game.conditions]
    nested=torch.ones(2)
    game.opt_g.state[next(iter(game.network.parameters()))]['extra_nested_history']={'rows':[(nested,)]}
    held.append(nested)
    # Interrupted native decode leaves plain (non-buffer) tensors in these lists.
    cache=torch.ones(2)
    vae._feat_map[0]=cache;vae._enc_feat_map[0]=torch.ones(2)
    held.append(cache)
    original_ids=list(map(id,held))
    external=torch.nn.Linear(3,2);outside=external.weight.detach().clone()
    rng=torch.random.get_rng_state().clone()
    calls=[];original_to=torch.Tensor.to
    import qwen_image_strict.cleanup as cleanup
    visited=[];original_release=cleanup.release_tensors
    def released(value,seen=None):
        if isinstance(value,torch.Tensor):visited.append(id(value))
        return original_release(value,seen)
    monkeypatch.setattr(cleanup,'release_tensors',released)
    def tracked(tensor,*args,**kwargs):
        if args and args[0]=='cpu':calls.append(id(tensor))
        return original_to(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',tracked)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:pytest.fail('CPU cleanup touched CUDA'))
    runtime.strict_close(game=game)
    assert runtime.closed and runtime._cleanup_complete and runtime.pipe is None
    assert game._failed and not game.backend.adapter.handles
    assert all(id(value)==before for value,before in zip(held,original_ids))
    # detach() creates a new view for the transfer. The original aliases remain
    # usable, retain their bytes/dtypes, and every nested tensor is reached.
    assert nested.device.type=='cpu' and torch.equal(nested,torch.ones(2))
    assert len(calls)>len(held) and all(id(value) in visited for value in held)
    assert id(external.weight) not in calls and torch.equal(external.weight,outside)
    assert all(value is None for value in vae._feat_map+vae._enc_feat_map)
    for module in model.modules():
        assert {k:vars(module)[k] for k in ('gradient_checkpointing','_gradient_checkpointing_func') if k in vars(module)}==original_checkpoint[id(module)]
    assert foreign_handle.id in model._forward_hooks and owned_handle.id not in model._forward_pre_hooks
    assert torch.equal(rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized()
    runtime.strict_close(game=game)  # idempotent retained closed binding
    foreign_handle.remove()


def test_cleanup_failure_does_not_skip_other_owned_resources_and_can_retry(monkeypatch):
    from qwen_image_sliders.tiny import TinyRuntime
    with initialization():base=TinyRuntime(41,with_vae=True)
    runtime=owned(base);events=[]
    class BrokenHandle:
        def remove(self):events.append('hook');raise RuntimeError('hook removal failed')
    runtime.handles=[BrokenHandle()]
    original_clear=base.pipe.vae.clear_cache
    def clear():events.append('VAE');original_clear()
    monkeypatch.setattr(base.pipe.vae,'clear_cache',clear)
    bank=SimpleNamespace(close=lambda:events.append('bank'))
    with pytest.raises(BaseExceptionGroup,match='owned cleanup failed'):runtime.strict_close(bank=bank)
    assert runtime.closed and not runtime._cleanup_complete and runtime.pipe is not None
    assert events==['hook','bank','VAE']
    runtime.handles=[]
    runtime.strict_close(bank=bank)
    assert runtime._cleanup_complete and runtime.pipe is None


def test_partial_native_construction_closes_owned_model_without_pipe():
    from qwen_image_sliders.tiny import TinyRuntime
    with initialization():base=TinyRuntime(41,with_vae=True)
    runtime=owned(base);runtime.pipe=None
    base.pipe.vae._feat_map[0]=torch.ones(3)
    runtime.strict_close()
    assert runtime._cleanup_complete and runtime._partial_model is None and runtime._partial_vae is None
    assert all(value is None for value in base.pipe.vae._feat_map)


@pytest.mark.parametrize('failure',['model_eval','vae_loader'])
def test_actual_tiny_model_released_after_constructor_partial_failure(monkeypatch,failure):
    from contextlib import nullcontext
    import diffusers
    import qwen_image_strict.native as native
    from qwen_image_sliders.tiny import TinyRuntime
    with initialization():base=TinyRuntime(41)
    model=base.model
    closed=[]
    class Encoder:
        pipe=base.pipe
        weights={}
        def encode(self,text):
            return dict(token_ids=torch.ones(1,2,dtype=torch.long),original_identity={},transport_identity={},transport={})
        def close(self):closed.append('encoder')
    monkeypatch.setattr(native.CPUConditioner,'load_local',lambda *a,**k:Encoder())
    monkeypatch.setattr(native,'execution_identity',lambda **k:dict(numerical={}))
    monkeypatch.setattr(native,'inspect_assets',lambda *a,**k:{})
    monkeypatch.setattr(native,'require_host',lambda *a:None)
    monkeypatch.setattr(native,'admission',lambda *a:None)
    monkeypatch.setattr(native,'verify_parameters',lambda *a,**k:{})
    monkeypatch.setattr(native,'initialization',lambda **k:nullcontext())
    monkeypatch.setattr(torch.cuda,'device',lambda *a:nullcontext())
    monkeypatch.setattr(diffusers.QwenImage21Transformer2DModel,'from_pretrained',lambda *a,**k:model)
    def fail(*a,**k):raise RuntimeError('intentional '+failure)
    if failure=='model_eval':monkeypatch.setattr(model,'eval',fail)
    else:monkeypatch.setattr(diffusers.AutoencoderKLQwenImage21,'from_pretrained',fail)
    runtime=StrictRuntime.__new__(StrictRuntime)
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError,match='intentional '+failure):
        runtime.__init__('/unused',conditioning_precision='float32',declared_prompts=['red mug'],
            expected_gpu_uuid='unused',guard=lambda:None)
    assert runtime.closed and runtime._cleanup_complete and runtime._partial_model is None
    assert runtime.pipe is None and closed==['encoder']
    assert all(parameter.device.type=='cpu' for parameter in model.parameters())
    assert torch.equal(rng,torch.random.get_rng_state()) and not torch.cuda.is_initialized()


def test_original_checkpoint_function_and_absent_instance_attributes_restored():
    from qwen_image_sliders.tiny import TinyRuntime
    with initialization():base=TinyRuntime(41)
    model=base.model
    original=lambda *args,**kwargs:None
    model.enable_gradient_checkpointing(gradient_checkpointing_func=original)
    restore=checkpoint_restore(model)
    model.enable_gradient_checkpointing(gradient_checkpointing_func=lambda *a,**k:None)
    restore()
    assert all(module._gradient_checkpointing_func is original for module in model.modules() if getattr(module,'gradient_checkpointing',False))
    assert model.is_gradient_checkpointing


def test_recursive_owned_rows_preserve_aliases_and_do_not_traverse_foreign_objects():
    from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
    latent=torch.ones(1,2,1,2,2);value=torch.ones(2)
    row=PreparedResponse(ResponseRow({'embeds':value},{'embeds':value},None,latent),value,value)
    foreign=SimpleNamespace(value=torch.ones(4))
    values={'rows':(row,), 'nested':[{'v':(value,)}], 'foreign':foreign}
    release_tensors(values)
    assert row.target is value and row.row.neutral['embeds'] is value and values['nested'][0]['v'][0] is value
    assert foreign.value.shape==(4,) and torch.equal(latent,torch.ones_like(latent))
