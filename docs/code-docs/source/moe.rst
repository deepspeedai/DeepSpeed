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
patterns for that model family. Running a HuggingFace model also requires a
Transformers build that exposes the matching config/model classes,
``model.config.model_type`` value, and fused expert layout.

The compatibility table below records the tiny HuggingFace smoke coverage used
for this AutoEP surface. ``Forward parity`` means a one-layer CPU CausalLM smoke
matched native HuggingFace logits and loss after AutoEP replacement. ``Replace``
means detection and replacement succeeded, but this smoke did not establish
end-to-end forward parity.

.. list-table:: AutoEP preset compatibility by Transformers version
   :header-rows: 1

   * - Preset
     - Minimum Transformers version
     - Smoke status
     - Notes
   * - ``mixtral``
     - ``5.0.0``
     - Forward parity
     - ``4.48.0`` through ``4.57.6`` expose the classes but do not match the
       preset's fused expert layout.
   * - ``qwen2_moe``
     - ``5.0.0``
     - Forward parity
     - ``4.48.0`` through ``4.57.6`` expose Qwen2-MoE classes but use a
       structure that AutoEP does not detect with this preset.
   * - ``qwen3_moe``
     - ``5.0.0``
     - Forward parity
     - Qwen3-MoE classes appear in ``4.51.3``, but ``4.x`` builds tested here
       do not match the preset's fused expert layout.
   * - ``qwen3_5_moe``
     - ``5.2.0``
     - Forward parity
     - Requires the Qwen3.5 text-backbone ``qwen3_5_moe_text`` model type;
       earlier builds tested here do not expose the required classes.
   * - ``deepseek_v2``
     - ``5.0.0``
     - Replace
     - ``4.54.1`` and ``4.57.6`` expose classes but do not match the preset;
       forward parity was not established by this smoke.
   * - ``deepseek_v3``
     - ``5.0.0``
     - Replace
     - ``4.51.3`` through ``4.57.6`` expose classes but do not match the
       preset; forward parity was not established by this smoke.
   * - ``llama4``
     - ``5.0.0``
     - Forward parity
     - Some ``4.x`` builds pass the tiny smoke, but ``4.51.3`` and ``4.54.1``
       did not; use ``5.0.0`` or newer for the validated path.

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
