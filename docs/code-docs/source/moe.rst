Mixture of Experts (DeepSpeed MoE)
==================================

DeepSpeed provides two forms of MoE support: DeepSpeed MoE and :doc:`AutoEP
(Automatic Expert Parallelism) <autoep>`. DeepSpeed MoE is the explicit
``deepspeed.moe.layer.MoE`` API for constructing MoE layers in model code. This
page introduces the DeepSpeed MoE API.

See also the `Mixture of Experts (DeepSpeed MoE) tutorial
<https://www.deepspeed.ai/tutorials/mixture-of-experts/>`__ for training
examples and configuration details.

.. autoclass:: deepspeed.moe.layer.MoE
    :members:
