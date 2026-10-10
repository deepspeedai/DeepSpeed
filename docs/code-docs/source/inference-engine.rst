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
one token. It cannot be combined with CUDA graph capture,
``release_inference_cache``, or continuous batching
(``SamplingConfig.continuous_batch_size``). Sampling still happens independently
for every response branch after the shared prompt forward.

Continuous batching (experimental)
-----------------------------------

Continuous batching is enabled through ``SamplingConfig.continuous_batch_size``
on the regular ``HybridEngineRollout.generate(request, sampling)`` entry point.
When unset, generation keeps its existing behavior. When set to a positive
value, at most that many prompt rows are active at once; completed rows retire
and pending rows are prefetched into the released slots. The returned
``RolloutBatch`` remains in the original ``RolloutRequest`` row order.
When ``HybridEngineRolloutConfig(enable_cache_trimming=True)`` is enabled, the
experimental path periodically trims unused cache columns from the left to
keep long-running staggered-EOS workloads within the allocated cache span.

When ``HybridEngineRolloutConfig(enable_profiling=True)`` is enabled, this path
also records a snapshot in ``get_last_profile()``. In addition to the common
rollout fields, the snapshot reports ``scheduler_overhead_ms`` for scheduler
transitions, ``cache_management_overhead_ms`` for cache compaction, trimming,
reset, and admitted-row copies, and separate ``prefill_forward_ms`` and
``decode_forward_ms`` totals. ``num_prefill_forwards`` counts each admitted
prompt batch, while ``num_decode_forwards`` counts decode steps that had
surviving rows. ``num_generated_tokens`` counts tokens actually produced by
all requests (padding is excluded), and ``active_batch_size`` is the maximum
number of simultaneously active rows; ``continuous_batch_size`` is the
configured capacity.

The experimental path intentionally does not implement paged attention or change the
default generation semantics. It currently requires one padded prompt width for
all rows, a model with cache-class support, greedy decoding, and one sample per
prompt. CUDA Graph capture and shared prompt prefill are rejected until the scheduling semantics are
validated on real workloads. Models that explicitly declare no cache-class
support are rejected; models with unknown support should be validated against
the default ``generate()`` path before use.
``align_decode_fronts=False`` is the default equal-width padded-prompt baseline:
all requests use the same padded prompt width. Different effective lengths
encoded by the attention masks retain the legacy staggered logical decode
positions; physical decode-front alignment is enabled only when
``align_decode_fronts=True``. Cache trimming is independently controlled by
``enable_cache_trimming`` and is disabled by default. When disabled, the
rollout does not trim periodically, but it reclaims a dead prefix when that is
necessary to avoid exhausting the configured cache capacity.

Set ``HybridEngineRolloutConfig(align_decode_fronts=True)`` to enable the
follow-up alignment path. It derives each request's effective prompt width from
its attention mask, orders requests from longest to shortest internally, and
restores the original row order in the returned batch. Retired rows are refilled
from that pre-sorted pending queue. Cache trimming remains disabled unless
``enable_cache_trimming=True`` is also set. You can override the derived cache
span with ``continuous_cache_capacity``; if the span is exhausted, the rollout
raises an error that names both remedies.

Set ``HybridEngineRolloutConfig(adaptive_prefill=True)`` to choose ordinary
batched generation or aligned CB with cost-based prefill buckets. Auto is
opt-in and takes precedence over ``align_decode_fronts`` without modifying the
caller's configuration. A length-sorted O(n²) dynamic program minimizes the
additive estimated cost of contiguous buckets, including one large bucket and
one request per bucket as candidates. Each bucket uses ordinary left padding
and a two-dimensional attention mask; no packed/varlen kernel is required.
The CB cache layout remains right-aligned for the entire call. Newly admitted
requests are replanned together on every refill, without reordering survivors.

The estimated bucket cost is a fixed Forward term plus linear padded-token
work, quadratic Attention work, and KV traffic. Attention work uses the model's
layer count, query heads and ``head_dim``; KV traffic uses KV heads, ``head_dim``
and parameter element size. Projection/MLP work is represented by the calibrated
linear coefficient rather than inferred from head dimension alone.

The following coefficients are starting estimates, not portable guarantees:

* ``prefill_fixed_cost_ms=25.0``: fixed cost per prefill Forward.
* ``prefill_token_cost_ms=0.1``: milliseconds per padded token position.
* ``prefill_attention_cost_ms=0.01``: milliseconds per estimated GFLOP of
  Attention work (``4 * B * L² * layers * query_heads * head_dim``).
* ``prefill_kv_cost_ms=0.04``: milliseconds per MiB of estimated KV traffic.
* ``continuous_decode_cost_ms=10.0``: extra CB milliseconds per Decode step
  (generation budget minus the first token returned by Prefill), used when
  comparing with ordinary generation.

All costs must be finite and non-negative. Calibrate them for the model,
device, precision and attention backend. DP is optimal for these estimates;
it cannot guarantee globally minimal measured GPU latency. Ordinary generation
is eligible only when all requests fit the active-row and prefill limits, the
physical input width fits the model position limit, and neither explicit static
capacity nor trimming is requested. Otherwise Auto remains on the CB path.

Auto validates masks and reads effective lengths in one batch transfer. When
ordinary generation beats a lower bound on every CB partition, it skips DP;
uniform prompt lengths also admit an exact single-bucket estimate. The fallback
reuses the resolved generation configuration and normalizes the result once.
This keeps planning overhead small for short generations and large request counts.

``benchmarks/rollout_prefill.py`` provides separate calibration and evaluation:

.. code-block:: bash

   python benchmarks/rollout_prefill.py --model Qwen/Qwen3-32B --dtype bfloat16 \
       --repeats 3 --calibrate costs.json
   python benchmarks/rollout_prefill.py --model Qwen/Qwen3-32B --dtype bfloat16 \
       --cost-config costs.json --repeats 3 --output sweep.json

Use ``--model tiny-qwen2 --dtype float32`` for a seeded small-model smoke test.
Calibration uses six uniform shapes distinct from the evaluation workloads,
fits non-negative fixed/token/Attention terms, measures KV management rates,
and estimates extra CB Decode cost from one-token and eight-token calls.
The saved JSON records model, GPU, precision, backend and library versions;
the sweep rejects a calibration from a different configuration. Applications
can pass its ``costs`` dictionary to ``HybridEngineRolloutConfig``.

The four evaluation workloads cover 128 short requests, a 128/4 tail, a
16..512 spread and a 512/16 tail. Timing includes routing and cache work, with
one warmup and interleaved repetitions of all four paths. The report includes
``auto_vs_best`` and the worst-case regression, plus selected routes and output
ID agreement. Generation is deliberately limited to two tokens with EOS
disabled to expose short-generation overhead; this is a controlled performance
test rather than a natural-EOS rollout trace. Use ``--new-tokens`` to evaluate
other budgets and recalibrate when the deployment configuration changes.

``prefill_max_tokens=65536`` limits padded token positions in each Forward;
set it to ``None`` to remove this planning limit. A single prompt above the
configured limit raises an error; this does not implement chunked prefill.
Long prompts also need a model position limit covering the prompt and generation
budget, sufficient static KV capacity, and enough temporary Forward memory.
The token limit does not bound allocated Decode KV memory; reduce the configured
active-row capacity or cache capacity when needed.

Automatic selection currently requires single-process greedy rollout with one
sample per prompt and a supported cache-class model. Other non-default generation
settings beyond repetition penalty, token IDs and the supported length/sampling
settings are rejected; for example, beam search, no-repeat n-grams, forced tokens
and dictionary outputs cannot be reproduced by the continuous path. Auto resolves
HF's effective generation configuration before validation, including legacy model
settings, while retaining explicit generation-configuration overrides. Repetition
penalty retains the original padded prompt history even when bucket model input
is trimmed.

With profiling, ``get_last_profile()["generation_strategy"]`` reports
``"batched"`` or ``"bucketed"`` for Auto; ``num_prefill_forwards`` counts actual
bucket Forwards on the CB path. End-to-end time includes initial route planning
and result normalization. Ordinary generation has no static-cache snapshot,
so ``get_last_continuous_stats()`` returns ``None`` for that call. Across Auto
paths, productive-token counts and throughput exclude structural padding and
retain EOS plus PAD-valued tokens generated before termination; active/configured
row-capacity fields retain their continuous-call meaning. Manual alignment and
equal-width modes retain their existing profile strategy labels.

The most recent cache statistics are available from
``rollout.get_last_continuous_stats()``. They include ``cache_capacity``,
``peak_cache_length``, ``cache_memory_bytes``, ``trim_count``,
``trimmed_columns``, ``trim_frequency``, and ``trim_bytes_moved``.
``cache_memory_bytes`` covers
the preallocated KV tensors and active cache metadata. With profiling enabled,
``trim_latency_ms``, ``end_to_end_ms``, and ``tokens_per_second`` are also
measured with accelerator synchronization; otherwise those timing fields are
``None`` or zero.
``trim_frequency`` is the number of trims divided by decode steps. Trimming
statistics remain zero when ``enable_cache_trimming`` is false unless a
capacity-exhaustion fallback reclaims a dead prefix.

``DeepSpeedStaticCache`` accepts one write position per row and can compact
active rows while preserving its static tensor addresses. This mirrors the
scheduler/cache separation used by systems such as vLLM and SGLang without
copying their backend-specific kernels.
