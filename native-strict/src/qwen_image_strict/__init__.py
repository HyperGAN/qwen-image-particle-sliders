"""Opt-in QwenImage21 staged native source; release/historical cohorts remain separate."""
BASE="718f8d54e5f7ecffa5953f40604a1253d685b034"
CORE="ec1161f5d74481be3540cb0dfa63d0e8f8b54e90"
DIFFUSERS="1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458"
BACKEND_NAME="qwen-image-native-strict"
COHORT="qwen-image21-staged-strict-native-engine-v1"
PROFILES={"bfloat16":"qwen-image21-cpu-bf16-pread-encoder-cuda-bf16-dit-vae-math-v2", "float32":"qwen-image21-cpu-fp32-pread-encoder-rounded-bf16-cuda-bf16-dit-vae-math-v2"}
