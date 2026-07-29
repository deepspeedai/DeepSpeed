# DeepSpeed v0.19.3 MUSA adaptation notes

Minimal port based on the MLU accelerator pattern in upstream DeepSpeed 0.19.3, mapped to `torch.musa` + MCCL.

**Detailed notes (Chinese):** `docs/MUSA_ADAPTATION_CN.md`

## Added / changed
- `accelerator/musa_accelerator.py` (+ `deepspeed/accelerator/` mirror)
- `op_builder/musa/` (+ `deepspeed/ops/op_builder/musa/` mirror)
- Registration / override check / auto-detect / factory in both `real_accelerator.py` copies
