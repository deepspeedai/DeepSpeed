import sys, deepspeed, torch
from transformers import AutoModel, AutoConfig;
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
c = AutoConfig.from_pretrained(model)
c.num_hidden_layers=16 # cut the number of layers for faster loading
m = AutoModel.from_pretrained(model, config=c)
dsc = dict(
    train_micro_batch_size_per_gpu=1,
    train_batch_size=1,
    zero_optimization=dict(stage=2),
    bf16=dict(enabled=True),
    optimizer=dict(type="AdamW"),
    gradient_accumulation_steps=1,
)
deepspeed.initialize(model=m, config=dsc)
torch.distributed.destroy_process_group()
