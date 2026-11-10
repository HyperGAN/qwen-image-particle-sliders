"""FOUR NEW AUTHORED UNRUN cases; real random tiny Qwen model, synthetic retained data.

No pretrained/model download, paired teacher capture or adapter.prepare. Exactly
ONE original ConditionalPG09 update across the complete file (third node only).
These fixtures establish no pretrained/native-render or LR qualification.
"""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
from particle_sliders.pg09_images import QwenImage21ResponseAdapter
from qwen_image_sliders.tiny import TinyRuntime
from qwen_image_sliders.fresh95 import open_training,open_inference,save_prepared_cache,configuration
from qwen_image_sliders.fresh95 import policy,training,ownership
from qwen_image_sliders.fresh95.cache import CacheBinding,prepared_digest
from qwen_image_sliders.fresh95.util import Checkpoint,equal,file_pin


def runtime_fixture():
    with torch.random.fork_rng():return TinyRuntime(seed=9)


def config_fixture(**kwargs):
    return configuration([dict(neutral='NEW software neutral one',positive='NEW software positive one'),
        dict(neutral='NEW software neutral two',positive='NEW software positive two')],steps=4,seed=7,batch=2,
        height=32,width=32,trajectory_steps=4,records_per_pair=1,device='cpu',dummy=True,**kwargs)


def prepared_fixture(config):
    result=[]
    for index in range(2):
        # Exercise both actual original mask forms without casting either.
        mask=torch.tensor([[True,True,True,True,False]])if index==0 else None
        embeds=torch.full((1,5,16),.2+index*.1)
        condition=dict(embeds=embeds,mask=mask,image_pad_mask=torch.zeros(1,5,dtype=torch.bool))
        positive=dict(embeds=embeds+.1,mask=None if mask is None else mask.clone(),image_pad_mask=torch.zeros(1,5,dtype=torch.bool))
        latent=torch.randn(1,8,1,4,4,generator=torch.Generator().manual_seed(index+11))
        row=ResponseRow(condition,positive,deepcopy(condition),latent,timestep=torch.tensor(500.-100*index))
        result.append(PreparedResponse(row,torch.full_like(latent,.1+.1*index),torch.zeros_like(latent)))
    return result


def cache_fixture(tmp_path,runtime,config,name='completed-cache.pt'):
    rows=prepared_fixture(config);sources=ownership.sources()['original']
    origins=[dict(kind='NEW-synthetic-software-PreparedResponse',completed=True,row_index=i,pair_index=i,
        record_index=0,control=False,model_identity=deepcopy(runtime.base_identity),original_sources=deepcopy(sources),
        prepared_sha256=prepared_digest(row),completed_output_receipt=dict(
            scope='NEW synthetic software data; no native teacher output claimed',row=i))for i,row in enumerate(rows)]
    return save_prepared_cache(tmp_path/name,runtime=runtime,config=config,prepared=rows,teacher_origins=origins,
        cache_id='NEW-Qwen-software-retained-PreparedResponse'),rows


def forbid_teachers(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('cached route attempted native teacher/encoding/preparation')
    import qwen_image_sliders.teachers as teachers
    monkeypatch.setattr(teachers,'paired_records',forbidden)
    monkeypatch.setattr(QwenImage21ResponseAdapter,'prepare',forbidden)
    monkeypatch.setattr(TinyRuntime,'encode',forbidden)


def admitted(request):
    assert request['ema_decay']is None and request['selection']=='raw'
    return True


def test_new_cached_fresh_initializer_before_optimizers_hostQR_zeroUP_router_criticKEEP_and_identity(tmp_path,monkeypatch):
    runtime,config=runtime_fixture(),config_fixture();cache,rows=cache_fixture(tmp_path,runtime,config)
    forbid_teachers(monkeypatch);initialize,game=policy.initialize,training.ConditionalPG09;seen={}
    frozen=[(p,p.detach().clone())for p in runtime.model.parameters()]
    def init_spy(*args,**kwargs):
        bank,critic=args[1:3];particles=kwargs['particles']
        ids=[id(p)for p in bank.parameters()]+[id(particles)]+[id(p)for p in critic.parameters()]
        critics=deepcopy(critic.state_dict());routes=[deepcopy(layer.route.state_dict())for layer in bank.layers]
        receipt=initialize(*args,**kwargs)
        assert ids==[id(p)for p in bank.parameters()]+[id(particles)]+[id(p)for p in critic.parameters()]
        assert equal(critics,critic.state_dict())
        assert all(equal(before,layer.route.state_dict())for before,layer in zip(routes,bank.layers))
        seen['receipt']=receipt;return receipt
    def game_spy(backend,prepared,critic,**kwargs):
        assert backend.metadata()['fresh_initialization']['receipt']==seen['receipt']
        assert backend.network.particles is backend.particles is kwargs['particles']
        seen['receipt_before_optimizers']=True
        return game(backend,prepared,critic,**kwargs)
    monkeypatch.setattr(policy,'initialize',init_spy);monkeypatch.setattr(training,'ConditionalPG09',game_spy)
    owner=open_training(runtime,cache,tmp_path/'fresh',mode='fresh',cohort_id='NEW-Qwen-software',config=config,owner_admission=admitted)
    try:
        assert seen['receipt_before_optimizers']and owner.step==0
        assert owner.entry_FULL.bytes>0 and owner.entry_RAW.bytes>0
        assert all(torch.count_nonzero(layer.up.weight).item()==0 for layer in owner.bank.layers)
        assert owner.game.prior.z is owner.backend.network.particles
        bankrow=next(r for r in owner.initialization['parameter_policies']if r['name']=='bank.particles')
        assert bankrow['role']=='declared-bank-R2std002'and '0.02'in bankrow['distribution']
        for row in owner.initialization['parameter_policies']:
            if row['role']in('constructor-law-KEEP','protected-zero-UP'):assert row['before_sha256']==row['after_sha256']
        assert all(p is actual and not p.requires_grad and torch.equal(p,before)
            for(p,before),actual in zip(frozen,runtime.model.parameters()))
    finally:owner.close()
    assert all(not module._forward_hooks for module in runtime.model.modules())


def test_new_named_policy_determinism_actual_sqrt5_scale_bank002_and_ambient_RNG(tmp_path):
    runtime,config=runtime_fixture(),config_fixture();cache,_=cache_fixture(tmp_path,runtime,config)
    before=torch.get_rng_state().clone()
    first=open_training(runtime,cache,tmp_path/'a',mode='fresh',cohort_id='fresh-a',config=config,owner_admission=admitted)
    second=None
    try:
        assert torch.equal(before,torch.get_rng_state())
        second=open_training(runtime_fixture(),cache,tmp_path/'b',mode='fresh',cohort_id='fresh-b',config=config,owner_admission=admitted)
        assert equal(first.backend.network.state_dict(),second.backend.network.state_dict())
        assert equal(first.game.critic.state_dict(),second.game.critic.state_dict())
        for layer in first.bank.layers:
            weight=layer.down.weight.detach();variance=1/(3*layer.down.in_features)
            assert torch.allclose(weight@weight.T,torch.eye(weight.shape[0])*max(weight.shape)*variance,atol=2e-6,rtol=2e-5)
        bank=first.backend.particles.detach()
        assert bank.shape==(128,4)and torch.isfinite(bank).all()
        assert abs(float(bank.std(unbiased=False))-.02)<.002
        declaration=next(r for r in first.initialization['parameter_policies']if r['name']=='bank.particles')
        assert declaration['before_sha256']!=declaration['after_sha256']
        assert declaration['role']=='declared-bank-R2std002'and 'R2Normal'in declaration['distribution']
        assert first.initialization['seed']==7 and first.initialization['parameter_identity_preserved']is True
    finally:
        first.close()
        if second is not None:second.close()


def test_new_one_original_update_strict_whole_FULL_resume_and_RAW_native_response_skip_init(tmp_path,monkeypatch):
    runtime,config=runtime_fixture(),config_fixture();cache,rows=cache_fixture(tmp_path,runtime,config)
    forbid_teachers(monkeypatch)
    first=open_training(runtime,cache,tmp_path/'first',mode='fresh',cohort_id='resume-case',config=config,owner_admission=admitted)
    try:
        first.update()  # The ONLY update in all four NEW nodes.
        full=first.save();raw=first.export(tmp_path/'RAW1.pt')
        expected,receipt=deepcopy(first.game.state_dict()),deepcopy(first.initialization)
        assert first.step==1
    finally:first.close()
    def forbidden(*args,**kwargs):raise AssertionError('resume/RAW invoked fresh initializer or teacher cache load')
    monkeypatch.setattr(policy,'initialize',forbidden)
    resumed=open_training(runtime_fixture(),cache,tmp_path/'resumed',mode='resume',cohort_id='resume-case',
        config=config,resume=full,owner_admission=admitted)
    try:
        assert resumed.step==1 and resumed.initialization==receipt
        assert equal(resumed.game.state_dict(),expected)
    finally:resumed.close()
    # RAW route never reads a prepared cache or constructs a critic/game.
    import qwen_image_sliders.fresh95.cache as cache_module
    monkeypatch.setattr(cache_module,'load_cache',forbidden)
    inference=open_inference(runtime_fixture(),raw=raw,owner_admission=admitted)
    try:
        assert not hasattr(inference,'game')and not hasattr(inference,'critic')
        response=inference.response(rows[0].row,strength=1.,seed=3)
        assert response.shape==(1,8,1,4,4)and torch.isfinite(response).all()
        assert inference.packet['initialization']==receipt
    finally:inference.close()


def test_new_provenance_intent_refusals_before_attach_retryable_cleanup_and_foreign_owner(tmp_path,monkeypatch):
    runtime,config=runtime_fixture(),config_fixture();cache,_=cache_fixture(tmp_path,runtime,config)
    def forbidden(*args,**kwargs):raise AssertionError('refusal reached added components')
    broken=tmp_path/'corrupt-cache.pt';broken.write_bytes(Path(cache.checkpoint.path).read_bytes()+b'corrupt')
    corrupt=CacheBinding(Checkpoint(str(broken),cache.checkpoint.bytes,cache.checkpoint.sha256),cache.provenance)
    wrong=deepcopy(cache.provenance);wrong['model_identity']={'not':'this native owner'}
    wrong=CacheBinding(cache.checkpoint,wrong)
    with monkeypatch.context()as scoped:
        scoped.setattr(training,'attach',forbidden)
        for patch in(dict(cache=None),dict(cache=corrupt),dict(cache=wrong),dict(mode='resume'),
                dict(mode='fresh',resume=cache.checkpoint),dict(ema_decay=.9),dict(owner_admission=None)):
            args=dict(runtime=runtime,cache=cache,directory=tmp_path/'refused',mode='fresh',cohort_id='refused',config=config,
                owner_admission=admitted);args.update(patch)
            with pytest.raises((ValueError,FileNotFoundError)):open_training(**args)
        legacy=tmp_path/'packed-teachers.pt';torch.save({'schema':'qwen-native-pg09-training-v3','records':[]},legacy)
        with pytest.raises(ValueError):
            open_training(runtime,CacheBinding(file_pin(legacy),cache.provenance),tmp_path/'refused',mode='fresh',
                cohort_id='refused',config=config,owner_admission=admitted)
    # Original native checkpoint settings are captured, including missing attrs.
    checkpoint_config=config_fixture(gradient_checkpointing=True)
    checkpoint_cache,_=cache_fixture(tmp_path,runtime,checkpoint_config,'checkpoint-cache.pt')
    target=runtime.model.transformer_blocks[0].attn.to_out[0]
    foreign=target.register_forward_hook(lambda module,args,output:output)
    releases=[]
    owner=open_training(runtime,checkpoint_cache,tmp_path/'cleanup',mode='fresh',cohort_id='cleanup',
        config=checkpoint_config,owner_admission=admitted,release=lambda:releases.append('release'))
    handle=owner.bank.handles[0];remove=handle.remove;calls=[]
    def fail_once():
        calls.append('remove')
        if len(calls)==1:raise RuntimeError('NEW synthetic hook-removal fault')
        remove()
    monkeypatch.setattr(handle,'remove',fail_once)
    container=owner.bank.handles
    captured=tuple(container)
    assert owner._owned_handles==captured and all(a is b for a,b in zip(owner._pending_handles,captured))
    container.append(foreign)
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert owner.closing and owner.cleanup_pending and not owner.closed
    assert calls==[]and container[-1]is foreign and foreign.id in target._forward_hooks
    assert all(h.id in h.hooks_dict_ref()for h in captured)
    container.pop()  # Explicitly resolve the foreign-list conflict; never auto-detach it.
    with pytest.raises(BaseExceptionGroup)as failure:owner.close()
    assert failure.value.cleanup_owner is owner
    assert owner.closing and owner.cleanup_pending and not owner.closed and owner.bank.handles==[handle]
    with pytest.raises(RuntimeError):owner.update()
    assert not releases and foreign.id in target._forward_hooks
    # Replacing the public list is also refused without touching foreign hooks
    # or retrying the one remaining owned removal.
    owner.bank.handles=[foreign]
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert owner.bank.handles==[foreign]and foreign.id in target._forward_hooks and calls==['remove']
    owner.bank.handles=container
    # A foreign callback cannot be replaced by cleanup, and stays pending.
    operation=next(op for op in owner._operations if op[1]=='_gradient_checkpointing_func')
    host,key,original,installed=operation;foreign_callback=lambda *args,**kwargs:None
    setattr(host,key,foreign_callback)
    with pytest.raises(BaseExceptionGroup):owner.close()
    assert getattr(host,key)is foreign_callback and not releases
    assert calls==['remove','remove']  # Successfully removed handle is not replayed.
    setattr(host,key,installed);owner.close();owner.close()
    assert owner.closed and not owner.cleanup_pending and releases==['release']
    assert foreign.id in target._forward_hooks
    # Partial constructor ownership is captured by attach's finally before
    # cleanup, even when the original bank constructor completed then raised.
    original_init=training.AdapterBank.__init__;partial={}
    def fail_after_original(bank,*args,**kwargs):
        original_init(bank,*args,**kwargs)
        partial['bank']=bank;partial['handles']=tuple(bank.handles)
        raise RuntimeError('NEW synthetic post-constructor attachment fault')
    with monkeypatch.context()as scoped:
        scoped.setattr(training.AdapterBank,'__init__',fail_after_original)
        with pytest.raises(RuntimeError,match='post-constructor attachment fault'):
            open_training(runtime,checkpoint_cache,tmp_path/'partial',mode='fresh',cohort_id='partial',
                config=checkpoint_config,owner_admission=admitted,release=lambda:releases.append('partial-release'))
    assert partial['handles']and partial['bank'].handles==[]and releases==['release','partial-release']
    assert all(h.id not in h.hooks_dict_ref()for h in partial['handles'])
    assert foreign.id in target._forward_hooks
    foreign.remove()
    assert all(not module._forward_hooks for module in runtime.model.modules())
