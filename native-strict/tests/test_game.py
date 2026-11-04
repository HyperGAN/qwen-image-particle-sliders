from copy import deepcopy
import pytest
import torch
from qwen_image_strict.execution import initialization
from qwen_image_strict.game import build_game
from qwen_image_strict.cache import identity

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

@pytest.fixture
def factory(monkeypatch):
    from qwen_image_sliders.tiny import TinyRuntime
    from qwen_image_sliders.contracts import formulation
    from qwen_image_sliders.teachers import paired_records
    from particle_sliders.lifecycle import advance
    import qwen_image_strict.native as native
    monkeypatch.setattr(native,'require_live',lambda _:None)
    # The release fixture seeds globally. Scope its setup before forbidding
    # global seed calls in the actual new bank/critic/game factory under test.
    with initialization():runtime=TinyRuntime(41)
    records=paired_records(runtime,[dict(neutral='blue square',positive='red square'),
        dict(neutral='blue circle',positive='red circle'),dict(neutral='forest',positive='forest',control=True)],
        seed=7,height=32,width=32,trajectory_steps=2,records_per_pair=1)
    stamp,declared=formulation(batch=2,lr=.001)
    def make(**rates):
        return runtime,build_game(runtime,records,stamp,declared,7,steps=2,lr=.001,**rates)
    yield make,records

def test_real_game_exact_replay_and_optimizer_rates(factory,monkeypatch):
    from particle_sliders.lifecycle import advance
    make,records=factory
    monkeypatch.setattr(torch,'manual_seed',lambda *_:pytest.fail('global seed'))
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *_:pytest.fail('global CUDA seed'))
    rng=torch.random.get_rng_state().clone()
    runtime,full=make(d_lr_mult=18,prior_lr_mult=30)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert [g['lr'] for g in full.opt_g.param_groups]==[.001,.03]
    assert [g['lr'] for g in full.opt_d.param_groups]==[.001*18]
    initial=deepcopy(full.state_dict())
    advance(full)
    partial=deepcopy(full.state_dict())
    advance(full)
    expected=deepcopy(full.state_dict())
    full.backend.adapter.close()
    _,resumed=make(d_lr_mult=18,prior_lr_mult=30)
    same(initial,resumed.state_dict())
    resumed.load_state_dict(partial)
    advance(resumed)
    same(expected,resumed.state_dict())
    resumed.backend.adapter.close()
    _,wrong=make(d_lr_mult=19,prior_lr_mult=30)
    try:
        with pytest.raises(ValueError):wrong.load_state_dict(partial)
    finally:wrong.backend.adapter.close()
    bad=deepcopy(records);bad[0]['target'].flatten()[0].add_(1)
    assert identity(records)!=identity(bad)
