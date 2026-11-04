"""Opt-in QwenImage21 staged native source; release/historical cohorts remain separate."""
BASE="7486f21604e60cc1f3dfa0b8ebfd190a5df8ca19"
CORE="ddb41423bcefc1fe76cd94d0f9b367353b3ea137"
DIFFUSERS="1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458"
BACKEND_NAME="qwen-image-native-strict"
COHORT="qwen-image21-staged-strict-native-v1"
PROFILES={"bfloat16":"qwen-image21-cpu-bf16-encoder-cuda-bf16-dit-vae-math-v1", "float32":"qwen-image21-cpu-fp32-encoder-rounded-bf16-cuda-bf16-dit-vae-math-v1"}
