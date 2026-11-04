from copy import deepcopy
import pytest
import torch
from qwen_image_strict.execution import policy,admission

def test_gpu_guard_refuses_before_cuda_query(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('CUDA queried without guardian'))
    with pytest.raises(ValueError,match='guardian'):
        admission('GPU-72c1b506-891d-b8bc-b353-e020585e1c47')

def test_inference_source_mismatch_refuses_before_native_load(monkeypatch):
    from types import SimpleNamespace
    import qwen_image_strict.inference as inf
    import qwen_image_strict.provenance as provenance
    execution=dict(numerical=policy(conditioning_precision='float32'),sources={'changed':True})
    actual=dict(numerical=policy(conditioning_precision='float32'),sources={'changed':False})
    monkeypatch.setattr(provenance,'execution_identity',lambda **_:actual)
    spec=SimpleNamespace(configuration=dict(execution=execution,assets={},declared_prompts=['a'],base_precision='bfloat16'))
    with pytest.raises(ValueError,match='before allocation'):
        inf.validate(dict(model_dir='/unused',expected_gpu_uuid='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'),spec)

@pytest.mark.parametrize('bad', [dict(latent=torch.zeros(1,8,1,2,2)),dict(timestep=torch.tensor(float('nan'))),dict(prompt='undeclared')])
def test_native_response_request_refuses_wrong_geometry_time_or_prompt(bad):
    from qwen_image_strict.inference import check_request
    row=dict(prompt='declared',latent=torch.zeros(1,64,1,2,2),timestep=torch.tensor(1000.))
    row.update(bad)
    with pytest.raises(ValueError):check_request(dict(declared_prompts=['declared']),dict(operation='response',native=row))

def test_actual_resolved_config_ema_is_disabled():
    from qwen_image_strict.lifecycle import config
    c=config(conditioning_precision='float32')
    assert c.options['ema_decay'] is None
    assert c.cohort=='qwen-image21-cpu-fp32-encoder-rounded-bf16-cuda-bf16-dit-vae-math-v1'
