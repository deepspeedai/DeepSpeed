---
title: "Hybrid Engine OPT cache compatibility"
---

Hybrid Engine's injected OPT layers implement the legacy tuple-cache interface.
OPT decoder layers exposing `cache_position` or `past_key_values` use a newer
cache contract. Those models now retain native Hugging Face generation, with
an explicit warning that inference acceleration is unavailable. Training and
`engine.module.generate()` remain available; this does not add native-kernel
support for the newer Cache interface.

The legacy OPT injection path is unchanged. The fallback does not provide
inference tensor parallelism, continuous-batching native-cache operations, or
CUDA Graph acceleration. Use the supported legacy path when those features
are required.

The offline regression in
`tests/unit/hybrid_engine/test_he_opt_cache.py` uses two data-parallel ranks,
a tiny randomly initialized OPT, and repeated training/generation transitions.
Each rank compares greedy output tokens with an independent Hugging Face model
loaded from the updated weights. No model download is required.
