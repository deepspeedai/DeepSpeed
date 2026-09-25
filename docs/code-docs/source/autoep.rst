AutoEP (Automatic Expert Parallelism)
=====================================

AutoEP automatically detects MoE layers in Hugging Face models and replaces them
with EP-enabled versions, requiring zero model code changes. It follows the
pattern of AutoTP (Automatic Tensor Parallelism).

This API is separate from the explicit ``deepspeed.moe.layer.MoE`` layer API.
For the explicit DeepSpeed MoE layer API, see :doc:`moe`.

**Built-in AutoEP presets:** ``mixtral`` (Mixtral), ``qwen3_moe`` (Qwen3-MoE),
``qwen3_5_moe`` (Qwen3.5-MoE), ``deepseek_v2`` (DeepSeek-V2), and
``deepseek_v3`` (DeepSeek-V3).

The preset name means AutoEP knows the router, expert, and weight naming
patterns for that model family. Running a Hugging Face model also requires a
Transformers build that exposes the matching config/model classes,
``model.config.model_type`` value, and fused expert layout.

.. list-table:: AutoEP preset compatibility by Transformers version
   :header-rows: 1

   * - Preset
     - Minimum Transformers version
     - Notes
   * - ``mixtral``
     - ``5.0.0``
     -
   * - ``qwen3_moe``
     - ``5.0.0``
     - Also covers Qwen2-MoE when the installed Transformers build uses the
       validated fused expert layout. Qwen3-MoE classes appear in ``4.51.3``,
       but the tested ``4.x`` builds do not match the validated AutoEP layout.
   * - ``qwen3_5_moe``
     - ``5.2.0``
     - Requires the Qwen3.5 text-backbone ``qwen3_5_moe_text`` model type;
       for performance on Qwen3.5's Gated DeltaNet layers, install optimized
       kernels. See the `Hugging Face Transformers kernel loading docs
       <https://huggingface.co/docs/transformers/kernel_doc/loading_kernels>`__
       and the `Qwen FlashQLA blog <https://qwen.ai/blog?id=flashqla>`__.
   * - ``deepseek_v2``
     - ``5.0.0``
     - ``load_balance_coeff`` / expert-bias auxiliary-loss-free load balancing
       is not currently supported; non-null values are rejected.
   * - ``deepseek_v3``
     - ``5.0.0``
     - ``load_balance_coeff`` / expert-bias auxiliary-loss-free load balancing
       is not currently supported; non-null values are rejected.

**ZeRO compatibility:** Stages 0, 1, and 2, plus constrained Stage 3
support. Stage 3 requires AutoEP-managed MoE layers and does not support native
DeepSpeed MoE layers, AutoTP, tensor model parallelism from ``mpu``, sequence
parallelism, hpZeRO secondary tensor groups, non-1 expert tensor
parallelism, or quantized gradients. Stage 3 AutoEP checkpoints are saved
partition-natively in the ``zero_pp_rank_*`` shard files and support
same-topology load, module-only loads (``load_module_only``),
optimizer-state-free loads (``load_optimizer_states=False``), and Universal
Checkpoint conversion. Optimizer-including Universal Checkpoint loads can
resume with a different data-parallel world size, a different ``autoep_size``,
or both, when the target ``autoep_size`` divides the model's expert count.
Weights-only/module-only Universal Checkpoint loads use the converted
``fp32.pt`` parameter files and support the same data-parallel and
``autoep_size`` topology changes.

**Usage:**

.. code-block:: json

    {
        "expert_parallel": {
            "enabled": true,
            "autoep_size": 4,
            "preset_model": "mixtral"
        }
    }

Experimental regional ``torch.compile``
----------------------------------------

AutoEP can keep its router, token movement, expert computation, and collectives
in eager mode while compiling the surrounding decoder blocks with vanilla
``torch.compile``. This targets fragmented attention, normalization, residual,
and dense backward work without capturing AutoEP communication in the graph.
The path is opt-in. Enable it in the DeepSpeed configuration, then call
``engine.compile()`` after initialization:

.. code-block:: json

    {
      "compile": {
        "autoep_non_moe": true
      }
    }

.. code-block:: python

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    engine.compile()

The call must happen after ``deepspeed.initialize()`` so AutoEP replacement is
complete. DeepSpeed discovers each ``AutoEPMoELayer`` and regionally compiles
its direct callable parent with ``fullgraph=False`` and ``dynamic=False``.
This is usually a decoder block; a callable model root with a direct AutoEP
child is also supported. The AutoEP layer is an explicit compiler-disabled
graph break, so routing, AllToAll dispatch/combine, and expert execution remain
eager. An AutoEP layer used as the model root has no surrounding region and
is rejected.

The decoder blocks contain repeated attention, normalization, residual, and
dense work targeted by this optimization. Compiling these regions bounds the
traced code and lets structurally identical blocks reuse compiled graphs.
Modules outside the selected regions, such as top-level embeddings and the
language-model head, remain eager. They are not intrinsically incompatible
with compilation, but extending the region would need separate validation
and performance measurements. If the selected region is the model root,
its non-MoE operations are included.

``compile.autoep_non_moe`` defaults to ``false``, preserving the existing
full-model behavior of ``engine.compile()``. Setting the option alone does
not compile the model; execution stays eager until ``engine.compile()`` is
called.

The initial experimental path supports vanilla ``torch.compile`` with the
standard ``comm`` backend, sequence and pipeline parallel sizes of one, and
ZeRO stages 0, 1, and 2. Distributed performance and parity validation currently
target ZeRO stage 1. It rejects DeepEP, DeepCompile, AutoEP+AutoTP folding,
sequence or pipeline parallelism, ZeRO stage 3, optimizer or parameter offload,
compiled autograd, DeepCompile schedules, and any ``fullgraph`` or ``dynamic``
value other than ``False`` instead of silently changing the requested behavior.

**How it works:**

1. During ``deepspeed.initialize()``, AutoEP scans the model for MoE layers
   using preset-defined patterns (router name, expert name, weight shapes).
2. Detected MoE blocks are replaced with ``AutoEPMoELayer``, which uses
   TorchTitan's grouped GEMM kernels and AllToAll token dispatch.
3. EP/EDP process groups are created automatically based on ``autoep_size``.
4. Expert parameters are marked for expert-data-parallel gradient reduction;
   router and shared-expert parameters use standard data-parallel reduction.

**Router outputs and activation checkpointing:**

Models using Hugging Face's model-level router-logit recording capture the
existing gate output; AutoEP does not compute a second projection just to
populate an unused cache. Models whose MoE blocks return router logits
directly compute them locally when constructing the return value, preserving
that return contract and its gradients. No router-logit tensor is stored on
the layer, so checkpoint replay early-stop and exceptions cannot leave a
router-logit cache keeping the autograd graph alive between training steps.

**Communication backend (optional):**

The expert AllToAll can be carried by `DeepEP <https://github.com/deepseek-ai/DeepEP>`__
instead of the default collectives. This is opt-in and off by default; jobs
that set nothing keep the existing path unchanged.

.. code-block:: json

    {
      "expert_parallel": {
        "enabled": true,
        "autoep_size": 8,
        "comm_backend": "deepep",
        "comm_num_sm": 12,
        "comm_qp_margin": 4,
        "comm_max_tokens_per_rank": 4096
      }
    }

- ``comm_backend``: ``"comm"`` (default) uses ``deepspeed.comm`` collectives;
  ``"deepep"`` uses DeepEP's dispatch and combine kernels.
- ``comm_num_sm``: SMs given to communication. Default 12.
- ``comm_qp_margin``: RDMA queue pairs reserved beyond one per SM. Default 4.
- ``comm_max_tokens_per_rank``: largest per-rank token count the job will
  produce, which is ``micro_batch_size * seq_len`` when sequences are padded to
  a fixed length. Required when ``comm_backend`` is ``"deepep"`` because the
  DeepEP buffer is sized statically and must use the same capacity on every
  rank. A batch that exceeds it is an error.

For ``autoep_size > 1``, DeepEP receives the router output directly, bypassing
the collective backend's sorting, token expansion, and split-count exchange.
Shared experts and router-logit outputs retain the same behavior. The EP
communicator is initialized once before each layer's first DeepEP buffer is
constructed, including when the caller supplied a lazily initialized process
group. This initialization does not run on subsequent forwards. The standard
``comm`` and ``autoep_size=1`` paths are unchanged.

On 16 H100s across two nodes, replaying routing captured from real training,
DeepEP reduced payload AllToAll time from roughly 100 ms to 48 ms per step. A
full SFT step on Qwen3.5-MoE went from roughly 325 ms to 266 ms, a 1.2x speedup
that removes about 18% of the step, reproduced across two independent jobs
(1.21x and 1.24x). Both backends are measured in the same job, on the same pods
and alternating, since the same measurement varied by a quarter between jobs;
the figures are medians rather than single observations, and DeepEP's own
median moved by 0.3% between the two jobs while the collective baseline moved
by 2.5%. The advantage grows with routing imbalance: at the most skewed
step measured, the collective path degraded to 116 ms while DeepEP stayed flat.

Within one model, all MoE layers that agree on EP group, expert count, top-k,
hidden size, capacity, ``comm_num_sm`` and ``comm_qp_margin`` share a single
DeepEP buffer, which is every layer of a normal model. A buffer reserves fabric
resources that are not reported as device memory and that run out: measured on
32 H100s across four nodes, the twenty-eighth buffer per rank fails inside
``ncclDevCommCreate``, so one buffer per layer put a 27-layer ceiling on the
backend there. Buffers are also slow to build, about 15 seconds each on 16
H100s and 22 on 32, so sharing removes minutes of startup as well. The buffer is
released once the last layer holding it is torn down.

Sharing stops at the model. Two models converted separately get their own
buffers even on the same EP group with identical geometry, because they are
driven independently: an actor and a frozen reference model in a reinforcement
learning loop need not reach their MoE layers in any fixed order relative to
each other, and a shared DeepEP communication context would make that order
matter. The cost is one extra buffer per model against a ceiling of 27.

``comm_num_sm`` matters because communication competes with the expert GEMM for
SMs. The default of 12 was chosen by measuring whole steps: 8 SMs gave a median
297.9 ms against 265.4 ms at 12, and larger budgets were slower again.
``comm_qp_margin`` exists because DeepEP's automatic queue-pair count assumes
it is alone on the fabric, which exhausts the queue pairs ZeRO and the
data-parallel groups have already claimed in a training step.

Requirements and limits:

- The ``deep_ep`` package must be installed. It is imported only when this
  backend is selected, so installations without it are unaffected.
- DeepEP v2 requires NCCL 2.30.4 or newer, built with GIN support. Below that
  version the transport is unavailable regardless of the network.
- DeepEP v1 (the legacy ``Buffer`` API, using NVSHMEM and IBGDA) is not
  supported.
- bfloat16 only. DeepEP's dispatch kernel takes bfloat16 rows, so selecting
  this backend for an fp16 or fp32 run is rejected rather than silently
  downgraded.
- Not compatible with folded tensor parallelism
  (``expert_tensor_parallel_size > 1``), which is rejected at setup.

**Python cyclic GC (experimental):**

Large Python model graphs can accumulate cyclic objects during training, and a
generation-2 collection pauses one rank's Python thread, which is then exposed
as collective wait time on every expert-parallel rank. The top-level
``disable_python_gc`` option addresses this. It is process-wide rather than
AutoEP-specific, so it is documented with the general configuration options.

**Fused RMSNorm (experimental):**

RMSNorm is a model module rather than an AutoEP expert-parallel component, so
it is not configured under ``expert_parallel``. DeepSpeed provides an opt-in
installer that runs Hugging Face RMSNorm modules with fused Triton kernels:

.. code-block:: python

    from deepspeed.ops.triton_ops.fused_rms_norm import replace_rms_norm

    replaced = replace_rms_norm(model)

The kernels reproduce the RMSNorm expression of Llama-style models: the
variance and normalization are computed in FP32, the normalized value is cast
back to the input dtype once, and only then multiplied by the weight (gamma).
A class name ending in ``RMSNorm`` with a 1-D ``weight`` and a
``variance_epsilon`` does not imply that order: ``GptOssRMSNorm``, for one,
multiplies by the weight before the cast, and ``Olmo2RMSNorm`` moved to that
order between Transformers releases. The installer therefore replaces only
instances of the classes in ``SUPPORTED_RMS_NORM_CLASSES``: the RMSNorm
classes of Llama, Mistral, Mixtral, Phi-3, Qwen2, Qwen2-MoE, Qwen3, Qwen3-MoE,
DeepSeek-V2 and DeepSeek-V3. A module is replaced only if it still runs its
class's own ``forward`` and is at most 2048 wide, so subclasses, modules whose
``forward`` was already patched or wrapped (by another kernel installer or by
an earlier call) and wider norms are left alone. Replaced modules keep their
own weight Parameter and ``variance_epsilon``. The return value is the number
of modules replaced: 193 for Qwen3-30B-A3B, four per decoder layer plus the
final norm.

A replaced module runs the kernels when its input and weight are bfloat16 or
float16 CUDA tensors of the same dtype and Triton is available. For any other
input it runs its class's own eager forward, so the result is exactly eager's,
and logs a warning once. The installer can therefore run before the model is
moved to the GPU. ``fused_rms_norm(hidden, weight, eps)`` in the same module
applies the kernels directly and raises on unsupported inputs instead.

Where the kernels run, results are not bitwise identical to eager RMSNorm: the
kernels' FP32 reductions sum in a different order, so a small fraction of
outputs and gradients differ in their last bits.

For norms at most 128 wide, the fused kernels schedule several rows per
program and combine the input- and weight-gradient reads in backward. Wider
norms retain the one-row kernels. The different FP32 reduction order still
uses the existing ULP accuracy contract, not bitwise equality with eager.
At width one, the pre-gamma input derivative in exact arithmetic is
``eps / (x**2 + eps)**(3/2)``, which is nonzero when ``eps > 0``. With a
very small epsilon and large ``abs(x)``, low-precision rounding can hide this
gradient; width-one tests use a measurable positive epsilon and an FP64
reference rather than assuming the derivative is zero.

Requirements and limits:

- The kernels need CUDA with Triton. On ROCm or without Triton, replaced
  modules run eager.
- bfloat16 or float16, with the input and the weight in the same dtype. Other
  inputs run eager, including float32 weights under autocast.
- Norms wider than 2048 are not replaced. That covers the hidden size and head
  dimension of Qwen3-30B-A3B; in models with a wider hidden size, such as
  Qwen3-235B-A22B (4096), only the narrower norms, such as the 128-wide query
  and key norms, are replaced.
- Any leading shape. Non-contiguous inputs and weights are copied before the
  kernels run. The output and the input gradient the kernels produce are
  contiguous, whereas eager RMSNorm follows the strides of its input and of
  the incoming gradient.
- First-order gradients for the input and the weight; the epsilon is a
  constant. Differentiating those gradients again (double backward) raises.
  ``torch.compile`` and ``torch.func`` transforms are not covered.

**Fused weighted restore (experimental):**

After the combine all-to-all, AutoEP holds one row per routed assignment and has
to turn it back into one row per token. ``combine_impl`` selects how:

.. code-block:: json

    {
        "expert_parallel": {
            "enabled": true,
            "autoep_size": 16,
            "preset_model": "qwen3_moe",
            "combine_impl": "fused_weighted_sum"
        }
    }

``"auto"`` (default) resolves to ``"weighted_sum"``, which scatters the rows into
a zero-filled ``[tokens * top_k, hidden]`` buffer, widens it to FP32 to apply the
routing weights, and reduces over top-k. ``"fused_weighted_sum"`` computes the
same result in a single pass: each program owns one token and one slice of the
hidden dimension, walks its top-k rows in registers and accumulates in FP32, so
neither the scattered buffer nor the FP32 intermediate is allocated. At the
canonical shape the FP32 intermediate alone is 64 MiB per layer.

Routing weights are still accumulated in FP32 and cast once, so the result
matches the eager reduction to within the order of the top-k summation. Only the
reduction changes: the collectives, the router, the grouped GEMM and the
expert-major reorder are untouched.

``"fused_weighted_sum"`` is rejected, rather than quietly ignored, when it would
have nothing to replace or would change semantics:

- ``tensor_parallel.autotp_size`` greater than 1, which uses folded tensor
  parallelism and restores combined tokens from assignment metadata instead;
- ``expert_tensor_parallel_size`` greater than 1;
- ``comm_backend="deepep"`` with expert parallelism, because DeepEP already
  restores and reduces its routed rows;
- a resolved ``score_apply`` other than ``"post"``;
- activations that are not bfloat16, float16, or float32, a non-CUDA device, or
  a build without Triton.

Failing fast matters for measurement: a run that asked for the fused reduction
and silently got the eager one would report the difference between an
implementation and itself.

**Constraints:**

- ``autoep_size`` must divide ``num_experts`` for all detected MoE layers.
- ``autoep_size=1`` is valid: all experts remain local (no AllToAll), useful
  for functional testing on a single GPU.
- AutoEP currently cannot be combined with AutoTP
  (``tensor_parallel.autotp_size > 1``) or tensor model parallelism from
  ``mpu``; support is planned as follow-up work.
- AutoEP with ZeRO Stage 3 is supported only without sequence parallelism,
  hpZeRO secondary tensor groups, non-1 expert tensor parallelism, or
  quantized gradients.
- Regular checkpoint save/load requires matching ``autoep_size``. To change
  ``autoep_size`` or data-parallel world size across runs for the same
  AutoEP-detected model topology, convert the checkpoint to Universal
  Checkpoint format and load it with ``checkpoint.load_universal``; see the
  `Universal Checkpointing tutorial </tutorials/universal-checkpointing/>`__
  for the detailed flow and constraints.
- DeepSeek-V2 and DeepSeek-V3 AutoEP do not support load-balance expert bias
  yet. The built-in DeepSeek presets disable it by default; explicit non-null
  values fail.
