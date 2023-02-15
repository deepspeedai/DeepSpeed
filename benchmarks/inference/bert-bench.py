import torch
import time
import deepspeed
import argparse
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, help="hf model name")
parser.add_argument("--deepspeed", action="store_true", help="use deepspeed inference")
parser.add_argument("--dtype", type=str, default="fp16", help="fp16 or fp32")
parser.add_argument("--max-tokens", type=int, default=50, help="max new tokens")
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--trials", type=int, default=30, help="number of trials")
parser.add_argument("--kernel-inject", action="store_true", help="inject kernels on")
parser.add_argument("--graphs", action="store_true", help="CUDA Graphs on")
args = parser.parse_args()


def print_latency(latency_set, title, warmup=3):
    # trim warmup queries
    latency_set = latency_set[warmup:]
    count = len(latency_set)
    if count > 0:
        latency_set.sort()
        n50 = (count - 1) * 0.5 + 1
        n90 = (count - 1) * 0.9 + 1
        n95 = (count - 1) * 0.95 + 1
        n99 = (count - 1) * 0.99 + 1
        n999 = (count - 1) * 0.999 + 1

        avg = sum(latency_set) / count
        p50 = latency_set[int(n50) - 1]
        p90 = latency_set[int(n90) - 1]
        p95 = latency_set[int(n95) - 1]
        p99 = latency_set[int(n99) - 1]
        p999 = latency_set[int(n999) - 1]

        print(f"====== latency stats {title} ======")
        print("\tAvg Latency: {0:8.2f} ms".format(avg * 1000))
        print("\tP50 Latency: {0:8.2f} ms".format(p50 * 1000))
        print("\tP90 Latency: {0:8.2f} ms".format(p90 * 1000))
        print("\tP95 Latency: {0:8.2f} ms".format(p95 * 1000))
        print("\tP99 Latency: {0:8.2f} ms".format(p99 * 1000))
        print("\t999 Latency: {0:8.2f} ms".format(p999 * 1000))


deepspeed.init_distributed("nccl")

print(args.model, args.max_tokens, args.dtype)

if args.dtype.lower() == "fp16":
    dtype = torch.float16
else:
    dtype = torch.float32

pipe = pipeline("fill-mask", model=args.model, framework="pt", device=args.local_rank)

if dtype == torch.half:
    pipe.model.half()

mask = pipe.tokenizer.mask_token

br = pipe(f"Hello I'm a {mask} model")
if args.deepspeed:
    pipe.model = deepspeed.init_inference(pipe.model,
                                          dtype=dtype,
                                          mp_size=1,
                                          replace_with_kernel_inject=args.kernel_inject,
                                          enable_cuda_graph=args.graphs)
    pipe.model.profile_model_time()

responses = []
times = []
mtimes = []
for i in range(args.trials):
    torch.cuda.synchronize()
    start = time.time()
    r = pipe(f"Hello I'm a {mask} model")
    torch.cuda.synchronize()
    end = time.time()
    responses.append(r)
    times.append((end - start))
    mtimes += pipe.model.model_times()
    #print(f"{pipe.model.model_times()=}")

print_latency(times, "e2e latency")
print_latency(mtimes, "model latency")

print(responses[0:3])
