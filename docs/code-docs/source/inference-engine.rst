Inference API
=============

:func:`deepspeed.init_inference` returns an *inference engine*
of type :class:`InferenceEngine`.

.. code-block:: python

    for step, batch in enumerate(data_loader):
        #forward() method
        loss = engine(batch)

Forward Propagation
-------------------
.. autofunction:: deepspeed.InferenceEngine.forward

HybridEngine Rollout Profiling
------------------------------

``HybridEngineRollout`` can record synchronized stage timings for a rollout.
Profiling is disabled by default because synchronization changes execution
behavior and adds overhead. Enable it through ``HybridEngineRolloutConfig``::

    from deepspeed.runtime.rollout.hybrid_engine_rollout import (
        HybridEngineRollout,
        HybridEngineRolloutConfig,
    )

    rollout = HybridEngineRollout(
        engine,
        tokenizer,
        cfg=HybridEngineRolloutConfig(enable_profiling=True),
    )
    output = rollout.generate(request, sampling)
    profile = rollout.get_last_profile()

The profile contains synchronized times for prompt expansion, generation,
post-processing, and the complete rollout. Generation is further divided into
the first model forward (``prefill_forward_ms``), all later model forwards
(``decode_forward_ms``), and residual generation work
(``generation_overhead_ms``). The residual includes sampling, generation-loop
bookkeeping, shared-cache expansion, and other work outside the top-level model
forwards. ``num_decode_forwards`` reports how many forwards contributed to the
decode time.

Forward timings use accelerator events where supported and synchronize once at
the end of generation instead of after every generated token. Synchronous
accelerators without events, such as CPU, use wall-clock timings. The forward
breakdown is unavailable when an asynchronous accelerator lacks event timing,
such as MPS, and for the CUDA graph path because graph replays bypass model
forward hooks. In both cases its forward fields are ``None`` and its complete
generation time is reported as generation overhead.

Times are reported in milliseconds. ``num_generated_tokens`` counts all
returned response positions across the expanded batch, including padding
positions. ``tokens_per_second`` divides that count by the end-to-end rollout
time. The profile also records the input batch size, samples per prompt, prompt
length, and returned response length.
For benchmark matrices, cases execute from the largest effective batch to the
smallest because HybridEngine sizes its inference workspace on the first
forward. Results remain in the user-requested matrix order.

Shared Prompt Prefill
---------------------

When one prompt branches into multiple response samples,
``HybridEngineRolloutConfig(use_shared_prefill=True)`` computes the prompt
forward once and repeats its KV cache before decoding the independent response
branches. The option is disabled by default.

Shared prefill currently requires HybridEngine kernel injection, ZeRO stage 0,
inference tensor-parallel size 1, an internal KV cache, and a prompt longer than
one token. It can be combined with native HybridEngine decode graphs enabled by
``hybrid_engine.enable_cuda_graph`` for greedy fixed-length generation, but not
with ``HybridEngineRolloutConfig(use_graph_capture=True)`` or
``release_inference_cache``. Sampling still happens independently for every
response branch after the shared prompt forward.
