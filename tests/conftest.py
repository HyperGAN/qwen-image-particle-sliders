import os

# All tests are offline CPU tests; native fixtures do not fetch tokenizer/model files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import torch

torch.set_num_threads(2)
