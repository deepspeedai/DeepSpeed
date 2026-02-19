import os

# Suppress tokenizers parallelism warning (must be before importing transformers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from datetime import datetime

import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, enable_full_determinism
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import torch
import random
import numpy as np
import time
import os

from distributed_attention import ulysses_attention_forward
# from ring_attention import ring_attention_forward
from sp_dp_registry import get_group, populate_registry, get_registry

torch.set_float32_matmul_precision("high")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = 12 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_autosp_inputs(input_id: torch.Tensor, label_id: torch.Tensor, position_id: torch.Tensor, attention_mask: torch.Tensor, seq_dim: int):
    torch._dynamo.decorators.mark_dynamic(input_id, seq_dim)
    torch._dynamo.decorators.mark_dynamic(label_id, seq_dim)
    torch._dynamo.decorators.mark_dynamic(position_id, seq_dim)
    torch._dynamo.decorators.mark_dynamic(attention_mask, seq_dim)
    input_id.tag = "input_id"
    label_id.tag = "label_id"
    position_id.tag = "position_id"
    return input_id, label_id, position_id, attention_mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--compile", type=str, default="deepcompile")
    parser.add_argument("--passes", type=str, default=None)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_memory", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--profile_dir", type=str, default="profiles")
    parser.add_argument("--bench_step", type=int, default=1)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--experiment_folder", type=str, default="")
    parser.add_argument("--sp_size", type=int, default=2)
    parser.add_argument("--dp_size", type=int, default=1)

    return parser.parse_args()

def main():
    args = get_args()
    set_seed(12)

    if args.deterministic:
        enable_full_determinism(12)
        from torch._inductor import config
        config.fallback_random = True
        torch.use_deterministic_algorithms(True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    assert accelerator.num_processes == args.sp_size * args.dp_size, 'Incorrect dp/sp sizing'

    ## Set sp/dp groups accordingly. ##
    if args.compile in ['compile', 'eager', 'ringattn']:
        populate_registry(args.sp_size, args.dp_size)

    if accelerator.is_main_process:
        print(f'GROUP_REGISTRY: {get_registry()}')

    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading model and tokenizer...")

    model_name = args.model_name
    if args.compile == "deepcompile":
        attention_backend = "sdpa"
    else:
        if args.compile == "eager" or args.compile == "compile":
            from transformers.models.llama import modeling_llama
            attention_backend = "ulyssess"
            modeling_llama.ALL_ATTENTION_FUNCTIONS["ulyssess"] = ulysses_attention_forward
        elif args.compile == "ringattn":
            from transformers.models.llama import modeling_llama
            attention_backend = "ringattn"
            modeling_llama.ALL_ATTENTION_FUNCTIONS["ringattn"] = ring_attention_forward 

    if args.num_layers is not None:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if accelerator.is_main_process:
            print(f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
        model_config.num_hidden_layers = args.num_layers
        model_config._attn_implementation = attention_backend
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config._attn_implementation = attention_backend
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config, trust_remote_code=True)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if accelerator.is_main_process:
        print("Loading dataset...")

    g = torch.Generator()
    g.manual_seed(12)
    dataset = load_dataset('ag_news', split='train[:1%]')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=args.seq_length, truncation=True) ## Fix max_length and generate fake data instead to not exhaust disk.

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    num_replicas_ = args.dp_size
    rank_ = accelerator.process_index // args.sp_size

    sampler = DistributedSampler(tokenized_dataset, num_replicas=num_replicas_, rank=rank_, seed=12, shuffle=False)  
    data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, worker_init_fn=seed_worker, generator=g)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    print(f"Model prepared: {model.__class__}")

    if args.compile == "deepcompile":
        print(f"Running deepcompile with backend={args.backend}")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        model.compile(backend=args.backend)
    elif args.compile in ["compile", "ringattn"]:
        print(f"Running torch.compile with backend={args.backend}")
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, backend=args.backend)
    else:
        print(f"Running eager")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = args.model_name.split("/")[-1]
    exp_name = f"{model_name}_np{accelerator.num_processes}_{args.compile}_" \
               f"B{args.backend}_" \
               f"L{0 if args.num_layers is None else args.num_layers}_" \
               f"bs{args.batch_size}_seq{args.seq_length}_" \
               f"T{timestamp}"
    if args.profile_dir:
        if accelerator.is_main_process and args.profile_dir:
            os.makedirs(args.profile_dir, exist_ok=True)
            if args.profile:
                prof_dir = f"{args.profile_dir}/{exp_name}"
                os.makedirs(prof_dir, exist_ok=True)
        accelerator.wait_for_everyone()
    
    # Training loop
    model.train()
    global_step = 0
    print(f"Using global sequence length: {args.seq_length}")

    os.makedirs("logs", exist_ok=True)
    loss_log_file = open(f"logs/loss_{args.compile}_{args.seq_length}_{accelerator.process_index}.csv", "w")
    loss_log_file.write("step,loss\n")

    sp_rank = dist.get_rank() % args.sp_size
    for epoch in range(args.num_epochs):
        start_iter = time.time()

        for step, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device) # [B, S]
            B, S = input_ids.shape

            label_ids = input_ids.clone() # [B, S]
            position_ids = torch.arange(S, device=device).unsqueeze(0)
            attention_mask = batch['attention_mask'].to(device)

            #HACK: store the padding mask to be accessed directly in local attention
            from distributed_attention import set_padding_mask
            set_padding_mask(attention_mask)

            if args.compile == 'deepcompile':
                input_ids, label_ids, position_ids, attention_mask = prepare_autosp_inputs(input_ids, label_ids, position_ids, attention_mask, seq_dim=1)
            else:
                chunk_size = S // args.sp_size 
                start = sp_rank * chunk_size
                end = start + chunk_size
                input_ids = input_ids[:, start:end]               # [B, S_shard]
                label_ids = label_ids[:, start:end]               # [B, S_shard] - must match input_ids
                position_ids = position_ids[:, start:end]

            outputs = model(input_ids=input_ids, labels=label_ids, position_ids=position_ids, attention_mask=attention_mask)
            loss = outputs.loss
            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item()} time: {time.time() - start_iter} alloc_mem: {torch.cuda.memory_allocated() / (1024 ** 3)} peak_mem: {torch.cuda.max_memory_allocated() / (1024 ** 3)}")

            accelerator.backward(loss)

            loss_log_file.write(f"{global_step},{loss.item()}\n")
            loss_log_file.flush()

            global_step += 1
            if global_step > args.steps:
                break

if __name__ == "__main__":
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    main()

    
