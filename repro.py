import os
import sys

# Ensure venv's bin (e.g. ninja) is on PATH when launched via deepspeed
if sys.executable:
    path_prepend = os.path.normpath(os.path.dirname(sys.executable))
    if path_prepend not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = path_prepend + os.pathsep + os.environ.get("PATH", "")

import deepspeed
import torch
from transformers import AutoModel, AutoConfig

num_gpus = max(1, torch.cuda.device_count())
batch_size = num_gpus

model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
c = AutoConfig.from_pretrained(model)
c.num_hidden_layers=16 # cut the number of layers for faster loading
m = AutoModel.from_pretrained(model, config=c)
dsc = dict(
    train_micro_batch_size_per_gpu=1,
    train_batch_size=batch_size,
    zero_optimization=dict(stage=2),
    bf16=dict(enabled=True),
    optimizer=dict(type="AdamW"),
    gradient_accumulation_steps=1,
)
deepspeed.initialize(model=m, config=dsc)
torch.distributed.destroy_process_group()
