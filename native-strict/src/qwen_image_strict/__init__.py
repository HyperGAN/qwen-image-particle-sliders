"""Opt-in QwenImage21 staged native source; release/historical cohorts remain separate."""
BASE="a0090e51a819adaae2890038767dd39c9a231928"
CORE="3dd672a7f9b8c169daef90fd63e4bb42b9ddc423"
DIFFUSERS="1d5d056ecff3a8c1d887a7dd1aa66ca4a0323458"
BACKEND_NAME="qwen-image-native-strict"
COHORT="qwen-image21-staged-strict-native-pg095-v1"
PROFILES={"bfloat16":"qwen-image21-cpu-bf16-pread-encoder-cuda-bf16-dit-vae-math-v2", "float32":"qwen-image21-cpu-fp32-pread-encoder-rounded-bf16-cuda-bf16-dit-vae-math-v2"}
