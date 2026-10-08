Optimizers
===================

DeepSpeed offers high-performance implementations of ``Adam`` optimizer on CPU; ``FusedAdam``, ``FusedLamb`` optimizers on GPU.

Adam (CPU)
----------------------------
.. autoclass:: deepspeed.ops.adam.DeepSpeedCPUAdam

FusedAdam (GPU)
----------------------------
.. autoclass:: deepspeed.ops.adam.FusedAdam

PyTorch fused AdamW (GPU)
----------------------------
DeepSpeed can also use PyTorch's built-in fused AdamW implementation. This avoids building the DeepSpeed
``fused_adam`` extension and can be useful when the installed PyTorch release already supports the target GPU:

.. code-block:: json

   {
     "optimizer": {
       "type": "Adam",
       "params": {
         "lr": 1e-5,
         "torch_adam": true,
         "adam_w_mode": true,
         "fused": true
       }
     }
   }

With ``torch_adam`` enabled, DeepSpeed forwards the remaining optimizer parameters to
``torch.optim.AdamW`` (or ``torch.optim.Adam`` when ``adam_w_mode`` is false). PyTorch keeps ``fused`` opt-in,
so confirm that the installed PyTorch version, accelerator, and parameter dtypes support it. Do not set both
``foreach`` and ``fused`` to true. Fused and foreach implementations can follow different floating-point
execution orders; validate convergence and memory use for the target workload before adopting the fused path.

FusedLamb (GPU)
----------------------------
.. autoclass:: deepspeed.ops.lamb.FusedLamb
