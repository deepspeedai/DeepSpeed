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
parallelism, MiCS, hpZeRO secondary tensor groups, non-1 expert tensor
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

**How it works:**

1. During ``deepspeed.initialize()``, AutoEP scans the model for MoE layers
   using preset-defined patterns (router name, expert name, weight shapes).
2. Detected MoE blocks are replaced with ``AutoEPMoELayer``, which uses
   TorchTitan's grouped GEMM kernels and AllToAll token dispatch.
3. EP/EDP process groups are created automatically based on ``autoep_size``.
4. Expert parameters are marked for expert-data-parallel gradient reduction;
   router and shared-expert parameters use standard data-parallel reduction.

**Constraints:**

- ``autoep_size`` must divide ``num_experts`` for all detected MoE layers.
- ``autoep_size=1`` is valid: all experts remain local (no AllToAll), useful
  for functional testing on a single GPU.
- AutoEP currently cannot be combined with AutoTP
  (``tensor_parallel.autotp_size > 1``) or tensor model parallelism from
  ``mpu``; support is planned as follow-up work.
- AutoEP with ZeRO Stage 3 is supported only without sequence parallelism,
  MiCS, hpZeRO secondary tensor groups, non-1 expert tensor parallelism, or
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
