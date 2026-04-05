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

**Supported models:** Mixtral, Qwen3-MoE, DeepSeek-V2, DeepSeek-V3, LLaMA-4
(via built-in presets).

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
