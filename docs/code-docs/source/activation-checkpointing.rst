Activation Checkpointing
========================

The activation checkpointing API's in DeepSpeed can be used to enable a range
of memory optimizations relating to activation checkpointing. These include
activation partitioning across GPUs when using model parallelism, CPU
checkpointing, contiguous memory optimizations, etc.

Please see the `DeepSpeed JSON config <https://www.deepspeed.ai/docs/config-json/>`_
for the full set.

Here we present the activation checkpointing API. Please see the enabling
DeepSpeed for `Megatron-LM tutorial <https://www.deepspeed.ai/tutorials/megatron/>`_
for example usage.

Configuring Activation Checkpointing
------------------------------------
.. autofunction:: deepspeed.checkpointing.configure

.. autofunction:: deepspeed.checkpointing.is_configured


Using Activation Checkpointing
------------------------------
.. autofunction:: deepspeed.checkpointing.checkpoint

.. autofunction:: deepspeed.checkpointing.reset


Configuring and Checkpointing Random Seeds
------------------------------------------
.. autofunction:: deepspeed.checkpointing.get_cuda_rng_tracker

.. autofunction:: deepspeed.checkpointing.model_parallel_cuda_manual_seed

.. autoclass:: deepspeed.checkpointing.CudaRNGStatesTracker

.. autoclass:: deepspeed.checkpointing.CheckpointFunction
