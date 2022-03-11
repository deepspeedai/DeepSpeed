Optimizers
===================

DeepSpeed offers high-performance implementations of ``Adam`` optimizer on CPU; ``FusedAdam``, ``FusedLamb``, ``OnebitAdam``, ``OnebitLamb`` optimizers on GPU.

Adam (CPU)
----------------------------
.. autoclass:: deepspeed.ops.adam.DeepSpeedCPUAdam

FusedAdam (GPU)
----------------------------
.. autoclass:: deepspeed.ops.adam.FusedAdam

FusedLamb (GPU)
----------------------------
.. autoclass:: deepspeed.ops.lamb.FusedLamb

OneBitAdam (GPU)
----------------------------
.. autoclass:: deepspeed.runtime.fp16.onebit.adam.OnebitAdam

ZeroOneAdam (GPU)
----------------------------
.. autoclass:: deepspeed.runtime.fp16.onebit.zoadam.ZeroOneAdam

OnebitLamb (GPU)
----------------------------
.. autoclass:: deepspeed.runtime.fp16.onebit.lamb.OnebitLamb
