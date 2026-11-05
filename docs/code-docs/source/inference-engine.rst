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

Sampling on the Decode Paths
----------------------------

The graph path, and continuous batching with or without graphs, apply
``temperature``, ``top_k``, ``top_p`` and ``repetition_penalty`` from
``SamplingConfig`` in the order ``generate()`` applies them. A
``repetition_penalty`` of ``None`` uses the value in the model's
``generation_config``, and 1.0 disables the penalty. ``return_log_probs`` warns
about a penalty other than 1.0, since a training forward pass cannot reproduce
those log-probs.

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
all rows, a model with cache-class support, and one sample per
prompt. Shared prompt prefill is rejected until the scheduling semantics are
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

CUDA Graph capture is supported for CUDA inputs with the default
``align_decode_fronts=False`` and ``enable_cache_trimming=False`` settings. It
uses a fixed ``continuous_batch_size`` layout, keeps every KV-cache row and
decode tensor at a stable address, and masks inactive rows while the scheduler
refills them. Graph state is reused across matching rollout calls and rebuilt
when the model parameters or static decode shape changes. Aligned decode fronts
and cache trimming are rejected while graph capture is enabled because they
move cache rows or columns. ``flash_attention_2`` is also rejected because the
captured decode uses a four-dimensional additive attention mask.

``DeepSpeedStaticCache`` accepts one write position per row and can compact
active rows while preserving its static tensor addresses. This mirrors the
scheduler/cache separation used by systems such as vLLM and SGLang without
copying their backend-specific kernels.

``DeepSpeedStaticCache.get_query_offset(layer_idx)`` supports the causal-mask
query-offset interface used by Transformers 5.20. It returns zero before a
write position is supplied, the scalar write position for fixed-batch decoding,
or the largest per-row write position for continuous batching. Per-row attention
masks determine which cache entries are valid below that shared frontier.
The offset excludes the current write, unlike ``get_seq_length()``; querying it
does not advance the cache or synchronize a device tensor to the host.
This is a targeted query-offset compatibility fix, not a guarantee of support
for every Transformers 5.20 cache API.

HybridEngine Rollout Records
----------------------------

``RolloutRequest`` and ``RolloutBatch`` hold a list of ``RolloutRecord``. A
record is one prompt and its rollout: ``prompt`` (chat messages, or text that
is already rendered), ``tokens`` and ``attention_mask`` across every turn,
``response`` and ``response_length``, a ``loss_mask`` over the response (1 on
model tokens, 0 on tool output), ``rollout_log_probs``, ``reward`` and
``status`` (``COMPLETED``, ``TRUNCATED`` or ``ABORTED``). Records generated
from one prompt share its ``group_index``. A record without ``tokens`` is
rendered with the tokenizer's chat template.

``SamplingConfig(return_log_probs=True)`` records each generated token's
log-probability under ``softmax(logits / temperature)``, the distribution a
training forward pass can reproduce. ``generate()`` therefore rejects
``return_log_probs`` combined with ``top_k`` or ``top_p`` when sampling, and
warns about a repetition penalty other than 1.0.

Multi-turn Rollout and Tool Calling
-----------------------------------

With ``RolloutConfig(multi_turn=True, max_turns=...)``, ``HybridEngineRollout``
runs each record over several turns. Each turn generates for all records still
active, parses tool calls from every response that ends in EOS, runs the tools,
and appends their output as tool messages. A record ends ``COMPLETED`` when a
turn calls no tool, and ``TRUNCATED`` when it runs out of turns or tokens.
Model tokens are appended exactly as sampled; tool messages are tokenized from
the chat template's own rendering of a tool turn, so earlier tokens are never
re-tokenized.

Tools subclass ``Tool``: set ``name``, ``description`` and a JSON-schema
``parameters``, and implement ``async execute(arguments) -> str``. Pass them to
``build_rollout(..., tools=[...])``. Prompts are rendered with the tools'
specs, calls in one turn run concurrently, and parse errors, unknown tools and
exceptions are returned to the model as text instead of failing the rollout.
``RolloutConfig.tool_call_parser`` selects the call format: ``"hermes"``
(Qwen2.5, Qwen3) or ``"qwen3_coder"`` (Qwen3-Coder).

.. code-block:: python

    from deepspeed.runtime.rollout import RolloutConfig, RolloutRequest, SamplingConfig, Tool, build_rollout
    from deepspeed.runtime.rollout.base import RolloutRecord

    class Calculator(Tool):
        name = "calculator"
        description = "Evaluates an arithmetic expression."
        parameters = {"type": "object", "properties": {"expression": {"type": "string"}}}

        async def execute(self, arguments):
            return str(evaluate(arguments["expression"]))

    rollout = build_rollout(RolloutConfig(multi_turn=True, max_turns=8), engine, tokenizer, tools=[Calculator()])
    request = RolloutRequest(samples=[RolloutRecord(prompt=[{"role": "user", "content": question}])])
    sampling = SamplingConfig(max_new_tokens=8192, max_new_tokens_per_turn=2048, n_samples_per_prompt=8,
                              return_log_probs=True)
    records = rollout.generate(request, sampling).samples

``max_new_tokens`` caps a record's generated tokens over all turns,
``max_new_tokens_per_turn`` caps each turn, and
``RolloutConfig.max_context_len`` caps the whole conversation.
``RolloutConfig.chat_template_kwargs`` is passed to every chat-template call,
for example ``{"enable_thinking": False}``.

``SamplingConfig.continuous_batch_size`` schedules a multi-turn rollout by turn
instead of by record. It sets the number of decode rows, which needs
``max_context_len`` or ``HybridEngineRolloutConfig.continuous_cache_capacity`` to
size the cache. A row is freed when a turn ends; the tools run on the host while
the row serves another trajectory, and the next turn is prefilled into any free
row. ZeRO-3 is not supported on this path.

GRPO Loss
---------

``group_advantages(records, rewards)`` normalizes each reward by the mean and
standard deviation of its ``group_index`` group; ``scale_by_std=False`` keeps
only the mean subtraction. ``grpo_policy_loss`` is the PPO-clip loss on
per-token log-probs, with separate ``clip_low`` and ``clip_high`` ranges, an
optional k3 KL penalty against reference log-probs, and ``"token_mean"`` or
``"sample_mean"`` aggregation. It takes log-probs rather than logits, so it
works with a plain ``log_softmax`` and with vocabulary-parallel log-prob ops.
``old_log_probs_from_rollout`` lays out the records' ``rollout_log_probs`` like
``ResponseTokenBatch``'s targets:

.. code-block:: python

    from deepspeed.runtime.rollout import (ResponseTokenBatch, group_advantages, grpo_policy_loss,
                                           old_log_probs_from_rollout)

    advantages = group_advantages(rollout_batch.samples, rewards).to(device)
    response_batch = ResponseTokenBatch.from_rollout(rollout_batch, pad_token_id=tokenizer.pad_token_id, device=device)
    logits = response_batch.select_causal_logits(model(input_ids=response_batch.input_ids,
                                                       attention_mask=response_batch.attention_mask).logits)
    log_probs = torch.log_softmax(logits.float() / temperature, dim=-1)
    log_probs = log_probs.gather(-1, response_batch.target_ids.unsqueeze(-1)).squeeze(-1)
    output = grpo_policy_loss(log_probs, old_log_probs_from_rollout(rollout_batch, device), advantages,
                              response_batch.response_mask, clip_low=0.2, clip_high=0.28)
    engine.backward(output.loss)

``loss_sum`` and ``loss_normalizer`` are available for normalizing the loss
across ranks; ``clip_fraction`` and ``kl`` are detached logging metrics.

OPSD response-token utilities
-----------------------------

``ResponseTokenBatch.from_rollout`` left-pads a ``RolloutBatch``'s samples and
derives causal training targets and a response-only loss mask. A sample's
response is its last ``response_length`` tokens. The mask excludes prompt and
padding tokens, and response tokens whose ``loss_mask`` is 0 (multi-turn tool
output), while retaining an attended EOS token. Use
``generalized_jsd_loss`` to compare aligned student and teacher logits on that
mask:

.. code-block:: python

    from deepspeed.runtime.rollout import ResponseTokenBatch, generalized_jsd_loss

    response_batch = ResponseTokenBatch.from_rollout(rollout_batch, pad_token_id=tokenizer.pad_token_id)
    student_logits = response_batch.select_causal_logits(student_model(**model_inputs).logits)
    teacher_logits = aligned_teacher_logits
    loss_output = generalized_jsd_loss(
        student_logits,
        teacher_logits,
        response_batch.response_mask,
    )
    loss = loss_output.loss

The rollout supplies only the student on-policy trajectory. Construct the
teacher inputs, including any privileged context, separately and align the
teacher logits with the student causal-logit shape before calling the loss.
``loss_sum`` and ``valid_token_count`` are available for callers that need
explicit distributed loss normalization.
