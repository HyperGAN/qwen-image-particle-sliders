"""Actual public sessions/native Qwen rows; only CPU asset admission is stubbed."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from qwen_image_strict.execution import initialization,policy

TEXTS=['blue square','red square','blue circle','red circle','forest','blue triangle','red triangle']
TRAIN=[dict(neutral='blue square',positive='red square'),dict(neutral='blue circle',positive='red circle'),
       dict(neutral='forest',positive='forest',control=True)]
DEVELOPMENT={'development':dict(rows=[dict(neutral='blue triangle',positive='red triangle')],seed=202)}


@pytest.fixture
def public_factory(monkeypatch):
    from qwen_image_sliders.tiny import TinyRuntime
    from diffusers import QwenImage21Transformer2DModel
    from particle_sliders import lifecycle as core
    from qwen_image_strict import lifecycle as product,native,assets,provenance
    from qwen_image_strict.cleanup import checkpoint_restore
    # This fixture does not claim native asset/device admission. Every teacher,
    # adapter, game, session, evaluation transaction and native forward is real.
    monkeypatch.setattr(product,'validate',lambda _:None)
    monkeypatch.setattr(native,'require_live',lambda _:None)
    execution=dict(numerical=policy(conditioning_precision='float32'),sources={'CPU_native_fixture':True})
    monkeypatch.setattr(provenance,'execution_identity',lambda **_:deepcopy(execution))
    monkeypatch.setattr(assets,'inspect_assets',lambda *a,**k:dict(CPU_native_fixture=True))
    backend=replace(product.BACKEND,validate=lambda _:None)
    monkeypatch.setattr(core,'backend',lambda name:backend)
    runtimes=[]
    def runtime():
        with initialization():
            value=TinyRuntime(41)
            value.pipe.transformer=QwenImage21Transformer2DModel(in_channels=64,out_channels=64,
                num_layers=2,attention_head_dim=16,num_attention_heads=2,context_in_dim=16,
                mlp_ratio=2,axes_dims_rope=(4,6,6)).eval().requires_grad_(False)
        value.base_identity=dict(CPU_native_fixture=True,seed=41,channels=64)
        value.strict_execution=deepcopy(execution)
        value.strict_assets=dict(CPU_native_fixture=True,full_weight_hashes_verified=True)
        value.conditioning={text:None for text in TEXTS}
        owner=native.StrictRuntime.__new__(native.StrictRuntime)
        owner.device=torch.device('cpu');owner.closed=False;owner._cleanup_complete=False;owner.handles=[]
        owner.pipe=value.pipe;owner._partial_model=value.model;owner._partial_vae=None
        owner._checkpoint_restore=checkpoint_restore(value.model)
        value.strict_close=owner.strict_close
        runtimes.append(value)
        return value
    def open_session(pairs=None):
        value=runtime()
        cfg=product.config(conditioning_precision='float32',rows=TRAIN,declared_prompts=TEXTS,
            model_dir='/CPU-fixture-no-assets',expected_gpu_uuid='GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
            device='cpu',steps=2,height=32,width=32,trajectory_steps=2,records_per_pair=1,lr=.001)
        return value,core.open_session(cfg,runtime=value,evaluation_pairs=deepcopy(DEVELOPMENT if pairs is None else pairs))
    return open_session,runtime


def test_actual_public_train_dev_indices_strengths_state_RNG_and_adapter_equivalence(public_factory):
    from particle_sliders.pg09 import _hash
    from qwen_image_strict.native import observation
    make,_=public_factory
    runtime,session=make()
    try:
        assert session.binding.response is None and set(session.binding.evaluation_rows)=={'development'}
        session.step()
        before=_hash(session.checkpoint());rng=torch.random.get_rng_state().clone()
        for split,rows in [('train',session.game.rows),('development',session.binding.evaluation_rows['development'].rows)]:
            for index,row in enumerate(rows):
                for strength in (0.,.5,1.):
                    actual=session.evaluate(index=index,split=split,strength=strength,seed=19)
                    with torch.no_grad(),observation(runtime):
                        expected=session.game.backend.generated(replace(row,row=replace(row.row,scale=strength)))
                    assert torch.equal(actual,expected) and actual.shape==(1,64,1,2,2)
                    if strength==0:assert torch.equal(actual,row.neutral)
        assert _hash(session.checkpoint())==before and torch.equal(rng,torch.random.get_rng_state())
        assert session.manifest['native_response'] is None and 'development' in session.manifest['evaluation_splits']
    finally:session.close()


def test_actual_public_unknown_split_and_bad_indices_do_not_change_state(public_factory):
    from particle_sliders.lifecycle import UnsupportedOperation
    from particle_sliders.pg09 import _hash
    make,_=public_factory;_,session=make()
    try:
        before=_hash(session.checkpoint());rng=torch.random.get_rng_state().clone()
        for kwargs,error in [(dict(split='unknown'),UnsupportedOperation),(dict(index=-1),ValueError),
            (dict(split='development',index=1),ValueError),(dict(strength=2.),UnsupportedOperation)]:
            with pytest.raises(error):session.evaluate(**kwargs)
        assert _hash(session.checkpoint())==before and torch.equal(rng,torch.random.get_rng_state())
    finally:session.close()


def test_fresh_dev_seed_changes_resume_identity_and_cached_rows_snapshot(public_factory):
    from particle_sliders.lifecycle import CheckpointError
    from particle_sliders.pg09 import _hash
    make,_=public_factory
    _,first=make();other=None
    try:
        packet=first.checkpoint()
        _,other=make({'development':dict(rows=DEVELOPMENT['development']['rows'],seed=203)})
        before=_hash(other.checkpoint())
        with pytest.raises(CheckpointError):other.restore(packet)
        assert _hash(other.checkpoint())==before
        assert first.manifest['evaluation_splits']!=other.manifest['evaluation_splits']
    finally:
        first.close()
        if other is not None:other.close()


def test_observation_pairs_are_snapshotted_and_prepared_rows_released_on_close(public_factory,monkeypatch):
    from particle_sliders.pg09 import _hash
    import qwen_image_strict.cleanup as cleanup
    make,_=public_factory
    pairs=deepcopy(DEVELOPMENT);_,session=make(pairs)
    before=session.evaluate(0,split='development');manifest=deepcopy(session.manifest)
    pairs['development']['rows'][0]['neutral']='forest'
    assert torch.equal(before,session.evaluate(0,split='development')) and session.manifest==manifest
    held=session.binding.evaluation_rows['development'].rows[0].target
    reached=[];release=cleanup.release_tensors
    def track(value,seen=None):
        if isinstance(value,torch.Tensor):reached.append(id(value))
        return release(value,seen)
    monkeypatch.setattr(cleanup,'release_tensors',track)
    session.close()
    assert id(held) in reached


@pytest.mark.parametrize('bad',[{'development':{'rows':[{'neutral':{'embeds':torch.zeros(1)},'positive':'red triangle'}],'seed':202}},
    {'development':dict(DEVELOPMENT['development'],identity={'caller_claim':True})},
    {'development':dict(DEVELOPMENT['development'],height=128)},
    {'development':{'rows':[dict(neutral='undeclared',positive='red triangle')],'seed':202}}])
def test_no_tensor_conditioning_caller_identity_geometry_or_undeclared_prompt_injection(bad):
    from qwen_image_strict.evaluation import settings
    with pytest.raises(ValueError):settings(dict(declared_prompts=TEXTS),bad)


def test_actual_raw_inference_callback_keeps_native_request_contract(public_factory,monkeypatch):
    from qwen_image_strict import inference
    make,runtime_factory=public_factory
    runtime,session=make()
    expected=session.evaluate(0,strength=.5,seed=7)
    row=session.game.rows[0].row
    state=deepcopy(session.game.network.state_dict())
    latent=row.latent.detach().clone();timestep=row.timestep.detach().clone()
    session.close()
    fresh=runtime_factory()
    monkeypatch.setattr(inference,'validate',lambda *a:None)
    import qwen_image_strict.native as native
    monkeypatch.setattr(native,'StrictRuntime',lambda *a,**k:fresh)
    from types import SimpleNamespace
    spec=SimpleNamespace(configuration={'execution':{'numerical':{'conditioning_precision':'float32'}},'declared_prompts':TEXTS})
    bound=inference.prepare(dict(model_dir='/CPU-fixture',expected_gpu_uuid='unused'),spec)
    calls=[];handle=fresh.model.register_forward_hook(lambda *a:calls.append(True))
    try:
        bound.network.load_state_dict(state)
        actual=bound.response(strength=.5,prompt=TRAIN[0]['neutral'],latent=latent,timestep=timestep)
        assert calls and torch.equal(actual,expected)
    finally:handle.remove();bound.close()
