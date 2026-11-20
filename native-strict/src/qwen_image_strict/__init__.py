"""Opt-in QwenImage21 staged native source; release/historical cohorts remain separate."""
BASE="95704c27d76168aec1ceeefe70c0346084a8db2d"
CORE="c13890f8e2a1ecf1dc8eb9f2a37f9187cb9fea78"
DIFFUSERS="1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458"
BACKEND_NAME="qwen-image-native-strict"
COHORT="qwen-image21-staged-strict-native-engine-v1"
PROFILES={"bfloat16":"qwen-image21-cpu-bf16-pread-encoder-cuda-bf16-dit-vae-math-v2", "float32":"qwen-image21-cpu-fp32-pread-encoder-rounded-bf16-cuda-bf16-dit-vae-math-v2"}
