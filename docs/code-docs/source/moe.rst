Mixture of Experts (MoE)
========================

Layer specification
--------------------
.. autoclass:: deepspeed.moe.layer.MoE
    :members:

AutoEP (Automatic Expert Parallelism)
---------------------------------------

AutoEP automatically detects MoE layers in HuggingFace models and replaces them
with EP-enabled versions, requiring zero model code changes. It follows the
pattern of AutoTP (Automatic Tensor Parallelism).

**Built-in AutoEP presets:** ``mixtral`` (Mixtral), ``qwen3_moe`` (Qwen3-MoE),
``qwen3_5_moe`` (Qwen3.5-MoE), ``deepseek_v2`` (DeepSeek-V2),
``deepseek_v3`` (DeepSeek-V3), and ``llama4`` (LLaMA-4).

The preset name means AutoEP knows the router, expert, and weight naming
patterns for that model family. Running a HuggingFace model also requires a
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
   * - ``llama4``
     - ``5.0.0``
     -

**ZeRO compatibility:** Stages 0, 1, and 2. Stage 3 is not supported.

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
- AutoTP and sequence parallelism cannot both be active simultaneously.
- Checkpoint save/load requires matching ``autoep_size``.
- DeepSeek-V2 and DeepSeek-V3 AutoEP do not support load-balance expert bias
  yet. The built-in DeepSeek presets disable it by default; explicit non-null
  values fail.
