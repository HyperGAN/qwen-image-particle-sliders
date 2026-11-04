import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import pytest
import torch

@pytest.fixture(autouse=True)
def strict_policy():
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()
