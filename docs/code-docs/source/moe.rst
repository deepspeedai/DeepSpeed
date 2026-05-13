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

**Built-in AutoEP presets:** ``mixtral`` (Mixtral), ``qwen2_moe`` (Qwen2-MoE),
``qwen3_moe`` (Qwen3-MoE), ``qwen3_5_moe`` (Qwen3.5-MoE),
``deepseek_v2`` (DeepSeek-V2), ``deepseek_v3`` (DeepSeek-V3), and ``llama4``
(LLaMA-4).

The preset name means AutoEP knows the router, expert, and weight naming
patterns for that model family. Running a HuggingFace model still requires an
installed Transformers build that exposes the corresponding config/model
classes and ``model.config.model_type`` value. For example, the
``qwen3_5_moe`` preset is available in AutoEP, but Qwen3.5-MoE models require
Transformers Qwen3.5-MoE support, including the ``qwen3_5_moe`` and
``qwen3_5_moe_text`` config model types; older Transformers builds that do not
expose those classes must be upgraded or used with another supported
preset/model.

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
