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
post-processing, and the complete rollout. Times are reported in milliseconds.
``num_generated_tokens`` counts all returned response positions across the
expanded batch, including padding positions. ``tokens_per_second`` divides
that count by the end-to-end rollout time. The profile also records the input
batch size, samples per prompt, prompt length, and returned response length.
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
one token. It cannot be combined with CUDA graph capture or
``release_inference_cache``. Sampling still happens independently for every
response branch after the shared prompt forward.

Continuous-batching prototype
-----------------------------

``deepspeed.runtime.rollout.continuous_batching`` provides the scheduling and
slot-lifecycle primitive needed to build a continuous decode batch. Requests
are admitted in FIFO order up to a configured capacity. When a request retires,
the update identifies the surviving cache rows to compact and the pending
requests that can be prefetched into the newly free rows. The model backend is
responsible for applying that update, running prompt prefill, and constructing
the attention metadata for the active rows.

The prototype intentionally does not implement paged attention or change the
default ``HybridEngineRollout.generate`` path. For a first end-to-end trial,
``HybridEngineRollout.generate_continuous`` accepts one request per prompt row
and a matching list of greedy ``SamplingConfig`` objects. It dynamically
prefills admitted prompts and decodes surviving rows until every request has
finished. CUDA Graph capture, sampling, multiple samples per prompt, and
different prompt widths are intentionally rejected until the scheduling
semantics are validated on real workloads.

``DeepSpeedStaticCache`` accepts one write position per row and can compact
active rows while preserving its static tensor addresses. This mirrors the
scheduler/cache separation used by systems such as vLLM and SGLang without
copying their backend-specific kernels.
