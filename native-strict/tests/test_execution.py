import pytest
import torch
from qwen_image_strict.execution import policy,initialization,canonical_uuid

def test_policy_is_explicit_and_distinct():
    a=policy(conditioning_precision='float32');b=policy(conditioning_precision='bfloat16')
    assert a['profile']!=b['profile']
    assert a['decoder']=='ownedCUDA0 BF16 from original FP32 weights'
    with pytest.raises(ValueError):policy(conditioning_precision='auto')

def test_scoped_cpu_rng_success_and_failure(monkeypatch):
    monkeypatch.setattr(torch,'manual_seed',lambda *_:pytest.fail('CUDA-wide seeding'))
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *_:pytest.fail('CUDA-wide seeding'))
    initial=torch.random.get_rng_state().clone()
    with initialization():a=torch.rand(7)
    assert torch.equal(initial,torch.random.get_rng_state())
    with pytest.raises(RuntimeError):
        with initialization():
            assert torch.equal(a,torch.rand(7))
            raise RuntimeError('test')
    assert torch.equal(initial,torch.random.get_rng_state())

def test_real_uuid_forms():
    bare='72c1b506-891d-b8bc-b353-e020585e1c47'
    assert canonical_uuid(bare)==canonical_uuid('GPU-'+bare)
    for wrong in (None,'foo','GPU-notuuid'):
        with pytest.raises(ValueError):canonical_uuid(wrong)

def test_active_diffusers_dispatch_is_bound(monkeypatch):
    from diffusers.models import attention_dispatch as d
    old=d._AttentionBackendRegistry.get_active_backend
    active,fn=old()
    monkeypatch.setattr(d._AttentionBackendRegistry,'get_active_backend',lambda:(active,lambda *a:None))
    with pytest.raises(ValueError,match='dispatch'):policy(conditioning_precision='float32')
