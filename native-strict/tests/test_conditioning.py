import json
from pathlib import Path
import pytest
import torch
from qwen_image_strict.conditioning import CPUConditioner, verify_parameters, tensor_digest
from qwen_image_strict.execution import initialization, policy

SNAPSHOT = Path('/mnt/nvme/ai/caches/huggingface/main/hub/models--Qwen--Qwen-Image-2.1/snapshots/b3179ad355be050328e483a9dfdd9e60cd62adfa')

@pytest.fixture
def tiny(tmp_path):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoProcessor
    from diffusers import FlowMatchEulerDiscreteScheduler
    config = Qwen3VLConfig(text_config=dict(vocab_size=151936, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        rope_parameters=dict(rope_type='default',rope_theta=1000000.,mrope_section=[1,1,2])),
        vision_config=dict(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,
            out_hidden_size=16,deepstack_visual_indexes=[]))
    config._attn_implementation = 'eager'
    with initialization():
        model = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
    model.eval().requires_grad_(False)
    # A tiny actual native checkpoint, independent from the pretrained payload.
    model.save_pretrained(tmp_path)
    processor = AutoProcessor.from_pretrained(SNAPSHOT/'processor',local_files_only=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(SNAPSHOT/'scheduler',local_files_only=True)
    return model, processor, scheduler, tmp_path

@pytest.mark.parametrize('precision', ['bfloat16','float32'])
def test_actual_native_encoder_cast_prefix_and_transport(tiny, precision):
    model, processor, scheduler, directory = tiny
    model.to(getattr(torch,precision))
    weights=verify_parameters(model,directory,source_dtype=torch.bfloat16,loaded_dtype=getattr(torch,precision))
    numerical=policy(conditioning_precision=precision)
    initial=torch.random.get_rng_state().clone()
    conditioner=CPUConditioner(model,processor,scheduler,numerical=numerical,weights=weights)
    try:
        row=conditioner.encode('A blue square.')
        again=conditioner.encode('A blue square.')
        assert torch.equal(initial,torch.random.get_rng_state())
        assert row['token_ids'].shape[1] <= 512
        assert row['original']['prompt_embeds'].dtype == getattr(torch,precision)
        assert row['transport']['prompt_embeds'].dtype == torch.bfloat16
        assert torch.equal(row['original']['prompt_embeds'].to(torch.bfloat16), row['transport']['prompt_embeds'])
        assert row['transport_identity']==again['transport_identity']
        assert row['original']['prompt_embeds'].shape[-1]==16
        assert not row['original']['image_pad_mask'].any()
        assert row['weights']['exact_source_cast']
        torch.backends.cuda.matmul.allow_tf32=True
        with pytest.raises(ValueError,match='numerical'):conditioner.encode('A blue square.')
    finally:conditioner.close()

def test_loaded_wrong_tensor_refused(tiny):
    model, _, _, directory=tiny
    with torch.no_grad():next(model.parameters()).flatten()[0].add_(1)
    with pytest.raises(ValueError,match='parameter differs'):
        verify_parameters(model,directory,source_dtype=torch.bfloat16,loaded_dtype=torch.bfloat16)

@pytest.mark.parametrize('precision',['bfloat16','float32'])
def test_actual_tiny_native_lazy_pread_loader_matches_original_weights_and_outputs(tiny,precision,monkeypatch):
    from qwen_image_strict.encoder_loader import load_encoder
    from qwen_image_strict.conditioning import frozen_state,buffer_identity,config_precision
    import safetensors
    from transformers import Qwen3VLForConditionalGeneration
    model,processor,scheduler,directory=tiny
    # Compare with the actual native mmap loader, not a model created in FP32
    # and then .to(BF16): that cast also rounds nonpersistent rotary buffers.
    default=torch.get_default_dtype()
    with initialization():
        model=Qwen3VLForConditionalGeneration.from_pretrained(directory,dtype=getattr(torch,precision),
            device_map='cpu',attn_implementation='eager',local_files_only=True,low_cpu_mem_usage=True)
    model.eval().requires_grad_(False)
    native=safetensors.safe_open;backends=[]
    def checked(*args,**kwargs):
        backends.append(kwargs.get('backend'));assert kwargs.get('backend')=='pread'
        return native(*args,**kwargs)
    monkeypatch.setattr(safetensors,'safe_open',checked)
    initial=torch.random.get_rng_state().clone()
    loaded=load_encoder(directory,dtype=getattr(torch,precision),guard=lambda:None)
    loaded.eval().requires_grad_(False)
    assert torch.equal(initial,torch.random.get_rng_state())
    assert torch.get_default_dtype()==default
    assert backends and frozen_state(loaded)==frozen_state(model)
    assert buffer_identity(loaded)==buffer_identity(model)
    assert config_precision(loaded)==config_precision(model)
    weights=verify_parameters(loaded,directory,source_dtype=torch.bfloat16,loaded_dtype=getattr(torch,precision))
    a=CPUConditioner(model,processor,scheduler,numerical=policy(conditioning_precision=precision),weights=weights)
    b=CPUConditioner(loaded,processor,scheduler,numerical=policy(conditioning_precision=precision),weights=weights)
    try:
        actual=a.encode('A blue square.');current=b.encode('A blue square.')
        assert actual['original_identity']==current['original_identity']
        assert actual['transport_identity']==current['transport_identity']
    finally:a.close();b.close()

def test_registered_and_plain_rotary_buffers_refused_after_change(tiny):
    model,processor,scheduler,directory=tiny
    weights=verify_parameters(model,directory,source_dtype=torch.bfloat16,loaded_dtype=torch.bfloat16)
    numerical=policy(conditioning_precision='bfloat16')
    conditioner=CPUConditioner(model,processor,scheduler,numerical=numerical,weights=weights)
    rotary=model.model.language_model.rotary_emb
    try:
        rotary.original_inv_freq.add_(1)
        with pytest.raises(ValueError,match='rotary/buffer'):conditioner.encode('A blue square.')
    finally:conditioner.close()

def test_streamed_tensor_hash_preserves_exact_header_and_payload_bytes():
    import hashlib,json
    for value in (torch.tensor(1000.),torch.arange(32).to(torch.bfloat16).reshape(4,8)):
        header=json.dumps([list(value.shape),str(value.dtype)],separators=(',',':')).encode()
        expected=hashlib.sha256(header+value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        assert tensor_digest(value)==expected

def test_pretrained_loader_refuses_before_allocation(monkeypatch):
    import qwen_image_strict.provenance as p
    from transformers import Qwen3VLForConditionalGeneration
    monkeypatch.setattr(p,'execution_identity',lambda **_: {'numerical':policy(conditioning_precision='float32')})
    monkeypatch.setattr('qwen_image_strict.assets.inspect_assets',lambda *_,**__: {})
    monkeypatch.setattr('qwen_image_strict.conditioning.require_host',lambda *_: None)
    monkeypatch.setattr(Qwen3VLForConditionalGeneration,'from_pretrained',lambda *_,**__:pytest.fail('allocated before guardian'))
    with pytest.raises(ValueError,match='guardian'):
        CPUConditioner.load_local('/unused', numerical=policy(conditioning_precision='float32'))
