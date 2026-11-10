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

HybridEngine Rollout CUDA Graph Decode
--------------------------------------

``HybridEngineRolloutConfig(use_graph_capture=True)`` captures the decode
forward pass in a CUDA graph and replays it for every generated token. Only the
model forward is captured; the next token is chosen from the graph's output
logits after each replay. The graph path, and continuous batching with or
without graphs, apply ``temperature``, ``top_k``, ``top_p`` and
``repetition_penalty`` from ``SamplingConfig`` in the order ``generate()``
applies them, so every path samples from the same distribution. A
``repetition_penalty`` of ``None`` uses the value in the model's
``generation_config``, and 1.0 disables the penalty. Other
``generation_config`` settings are not applied on these paths. With
``return_log_probs``, ``generate()`` warns about a penalty other than 1.0,
since a training forward pass cannot reproduce those log-probs.

Shared Prompt Prefill
---------------------

When one prompt branches into multiple response samples,
``HybridEngineRolloutConfig(use_shared_prefill=True)`` computes the prompt
forward once and repeats its KV cache before decoding the independent response
branches. The option is disabled by default.

Shared prefill runs on HybridEngine's inference kernels, so it needs a model
with a DeepSpeed inference policy; models without one, such as Qwen3, are
rejected when the rollout is constructed, as are CUDA graph capture and
continuous batching (``continuous_batch_size``). It also requires ZeRO stage 0,
inference tensor-parallel size 1, no ``release_inference_cache``, and a prompt
longer than one token. In a multi-turn rollout it applies to each trajectory's
first turn: the copies of a prompt differ once they hold their own responses.
Sampling still happens independently for every response branch after the
shared prompt forward.

Continuous batching (experimental)
-----------------------------------

Continuous batching is enabled through ``continuous_batch_size`` in
``RolloutConfig`` or ``HybridEngineRolloutConfig``. It sizes the KV cache and
the captured graph, so it is fixed for the rollout engine rather than passed
with each ``generate(request, sampling)`` call. When unset, turns run in static
batches. When set, at most that many turns decode at once: a row is freed as
soon as its turn ends, and the next queued turn is prefilled into it. A
single-turn call is a rollout of one turn per sample; with ``multi_turn``, a
turn that calls a tool frees its row while the tool runs, and the trajectory's
next turn joins the front of the queue. The returned ``RolloutBatch`` remains in
the original ``RolloutRequest`` order, and any ``n_samples_per_prompt`` is
supported. Finished turns are collected after every decode step.

The cache layout depends on graph capture:

* **Without graph capture**, all live rows share one write column. A turn is
  admitted into the columns just before it; a turn whose context is longer than
  the column moves the column forward, and the live rows' attention masks skip
  the columns in between. Freed rows are compacted away, so each decode step
  runs only the live rows. By default every admission is prefilled in one
  left-padded forward and placed at that padded width.
  ``align_decode_fronts=True`` instead places each context with its real length,
  so no padding enters the cache, and orders the first admissions longest
  prompt first. Columns no live row attends are trimmed from the left when the
  cache is full, or as soon as they reach the prompt width with
  ``enable_cache_trimming=True``, at the cost of copying the live columns. If
  the cache is full and nothing can be trimmed, the rollout raises an error that
  names ``continuous_cache_capacity`` and ``enable_cache_trimming``.
* **With graph capture**, every row decodes at its own position, so the
  captured decode keeps one batch shape and stable tensor addresses. A turn is
  prefilled in one left-padded forward and only its real tokens are copied into
  its row, from slot 0. All rows decode every step; rows without a turn produce
  throwaway tokens. The graph is captured once per call, when the first turn is
  admitted. ``align_decode_fronts`` and ``enable_cache_trimming`` are rejected,
  since rows never share a column.

The cache size is ``continuous_cache_capacity`` when set. Otherwise it follows
from the longest context a turn can be admitted with, ``C``, and the most one
turn can generate, ``b`` (``max_new_tokens_per_turn`` when set, else
``max_new_tokens``). In a single-turn call ``C`` is the prompt width, and in a
multi-turn rollout it is ``max_context_len``, which, without
``continuous_cache_capacity``, a multi-turn rollout must set: tool output has no
bound. The shared write column gets ``C + b`` columns, which is enough because
trimming at full leaves the oldest live row at column 0. Each graph row gets
``max_context_len``, or the longest prompt plus ``b`` in a single-turn call.

Continuous batching rejects ZeRO-3, ``flash_attention_2``, shared prefill, and
models that declare no cache-class support.

When ``HybridEngineRolloutConfig(enable_profiling=True)`` is enabled,
``get_last_profile()`` records ``generation_ms``, ``prefill_forward_ms``,
``decode_forward_ms``, ``num_decode_forwards``, ``tool_ms``, ``num_turns``,
``num_generated_tokens`` (tokens actually generated, padding excluded) and
``tokens_per_second``. Without graph capture it also reports
``scheduler_overhead_ms``, ``cache_management_overhead_ms`` for compaction,
trimming, reset and admitted-row copies, ``num_prefill_forwards`` and
``active_batch_size``, the most rows live at once. With graph capture it reports
``continuous_rows``, ``continuous_decode_occupancy`` and
``continuous_step_trace``, one entry per decode step.

Without graph capture, the most recent cache statistics are available from
``rollout.get_last_continuous_stats()``. They include ``cache_capacity``,
``peak_cache_length``, ``cache_memory_bytes``, ``trim_count``,
``trimmed_columns``, ``trim_frequency``, and ``trim_bytes_moved``.
``cache_memory_bytes`` covers the preallocated KV tensors and active cache
metadata. With profiling enabled, ``trim_latency_ms``, ``end_to_end_ms``, and
``tokens_per_second`` are also measured with accelerator synchronization;
otherwise those timing fields are ``None`` or zero. ``trim_frequency`` is the
number of trims divided by decode steps.

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
model tokens, 0 on tool output), ``rollout_log_probs`` and ``status``
(``COMPLETED`` or ``TRUNCATED``). Records generated
from one prompt share its ``group_index``. A record without ``tokens`` is
rendered with the tokenizer's chat template.

``RolloutConfig(return_log_probs=True)`` records each generated token's
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

    rollout = build_rollout(RolloutConfig(multi_turn=True, max_turns=8, return_log_probs=True), engine, tokenizer,
                            tools=[Calculator()])
    request = RolloutRequest(samples=[RolloutRecord(prompt=[{"role": "user", "content": question}])])
    sampling = SamplingConfig(max_new_tokens=8192, max_new_tokens_per_turn=2048, n_samples_per_prompt=8)
    records = rollout.generate(request, sampling).samples

``max_new_tokens`` caps a record's generated tokens over all turns,
``max_new_tokens_per_turn`` caps each turn, and
``RolloutConfig.max_context_len`` caps the whole conversation.
``RolloutConfig.chat_template_kwargs`` is passed to every chat-template call,
for example ``{"enable_thinking": False}``.

``RolloutConfig.continuous_batch_size`` schedules a multi-turn rollout by turn
instead of by record (see "Continuous batching" above). It needs
``max_context_len`` or ``HybridEngineRolloutConfig.continuous_cache_capacity`` to
size the cache. A row is freed when a turn ends; the tools run on the host while
the row serves another trajectory, and the next turn is prefilled into any free
row.

Rollout runs independently on every data-parallel rank. Multi-turn rollout,
graph capture and continuous batching reject ZeRO-3: every forward gathers
parameters, so ranks must run the same forwards, and they run different numbers
of turns and decode steps. Single-turn static rollout supports ZeRO-3 by running
every rank to the same length: ``max_new_tokens``, or with ``max_context_len``
the longest budget any rank's prompts leave.

AutoTP (``tensor_parallel.autotp_size``) is supported with static batching.
Tensor-parallel ranks must run the same forwards, so before each generate call
every rank takes its group's first rank's random state, which makes sampling
draw the same tokens from the gathered logits, and tool calls run on the
group's first rank only, which broadcasts their output. Continuous batching
rejects AutoTP: a turn is admitted when its tools return, which happens at
different steps on each rank. HybridEngine's ``inference_tp_size`` is rejected,
since the rollout calls the module directly and skips the input gather of the
hybrid engine's ``generate()``.

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
