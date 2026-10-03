---
title: "Hybrid Engine OPT cache compatibility"
---

Hybrid Engine's injected OPT layers implement the legacy tuple-cache interface.
OPT decoder layers exposing `cache_position` or `past_key_values` use a newer
cache contract. Those models now retain native Hugging Face generation, with
an explicit warning that inference acceleration is unavailable. Training and
`engine.module.generate()` remain available; this does not add native-kernel
support for the newer Cache interface.

Legacy OPT models can use kernel injection. The fallback does not provide
inference tensor parallelism, continuous-batching native-cache operations, or
CUDA Graph acceleration. Use the supported legacy path when those features
are required.

The offline regression in
`tests/unit/hybrid_engine/test_he_opt_cache.py` uses two data-parallel ranks,
a tiny randomly initialized OPT, and repeated training/generation transitions.
Each rank compares greedy output tokens with an independent Hugging Face model
loaded from the updated weights. No model download is required.

## Legacy inference tensor parallelism

Legacy BLOOM and OPT models can use ZeRO-3 with pinned parameters and
`inference_tp_size > 1` without an external model-parallel unit. Each rank
copies its inference weight shards while the training parameters are gathered.
Generation releases those copies before training resumes.

The legacy-inference CI job uses Transformers 4.43.4 separately from the
DeepSpeed-Chat environment. Its offline tests compare generated tokens and
logits against independent Hugging Face models on one and two GPUs, check
repeated generation and a training step, and verify that inference shards
are released. Projection-only tests cover GPT-NeoX's fused QKV layout and
Llama's gated MLP; they do not qualify those models' generation or cache APIs.

Run the tests from the `tests` directory:

```bash
pytest --forked -m seq_inference \
  unit/hybrid_engine/test_he_all.py::TestHybridEngineTensorParallel \
  unit/hybrid_engine/test_he_all.py::TestHybridEngineProjectionShards
```
