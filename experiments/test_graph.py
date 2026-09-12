# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
import os, sys, time, torch

sys.path.insert(0, os.environ.get("DS_SRC", "."))
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.runtime.rollout import HybridEngineRollout, RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRolloutConfig
from deepspeed.accelerator import get_accelerator

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B-Base"
USE_KI = os.environ.get("USE_KI", "0") == "1"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(get_accelerator().device_name()).eval()
deepspeed.init_distributed(dist_backend="nccl")
engine, _, _, _ = deepspeed.initialize(model=model,
                                       config={
                                           "train_micro_batch_size_per_gpu": 1,
                                           "bf16": {
                                               "enabled": True
                                           },
                                           "hybrid_engine": {
                                               "enabled": True,
                                               "inference_tp_size": 1,
                                               "max_out_tokens": 256
                                           }
                                       })
engine.eval()
if USE_KI:
    from deepspeed.module_inject.segment_ki import apply_segment_ki
    print("KI", apply_segment_ki(engine.module, kernel="all", backend="auto"), flush=True)

rollout = HybridEngineRollout(engine=engine, tokenizer=tok, cfg=HybridEngineRolloutConfig(use_graph_capture=True))
dev = get_accelerator().device_name()
enc = tok("The capital of France is", return_tensors="pt")
req = RolloutRequest(prompt_ids=enc.input_ids.to(dev), prompt_attention_mask=enc.attention_mask.to(dev))
samp = SamplingConfig(max_new_tokens=64, temperature=0.0, top_p=1.0)

# warmup then timed
t0 = time.perf_counter()
batch = rollout.generate(req, samp)
get_accelerator().synchronize()
print(f"warm1 {time.perf_counter()-t0:.2f}s text={tok.decode(batch.input_ids[0, enc.input_ids.shape[1]:])[:80]!r}",
      flush=True)
for tag in ("run1", "run2"):
    t0 = time.perf_counter()
    batch = rollout.generate(req, samp)
    get_accelerator().synchronize()
    dt = time.perf_counter() - t0
    print(f"{tag} {dt:.2f}s tok/s={64/dt:.1f}", flush=True)
