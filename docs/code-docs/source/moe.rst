Mixture of Experts (DeepSpeed MoE)
==================================

DeepSpeed MoE is the explicit ``deepspeed.moe.layer.MoE`` API for constructing
MoE layers in model code.

For AutoEP (Automatic Expert Parallelism), which automatically detects and
replaces supported Hugging Face MoE layers from DeepSpeed config without model
code changes, see :doc:`autoep`.

See the `Mixture of Experts (DeepSpeed MoE) tutorial
<https://www.deepspeed.ai/tutorials/mixture-of-experts/>`__ for training
examples and configuration details.

.. autoclass:: deepspeed.moe.layer.MoE
    :members:
