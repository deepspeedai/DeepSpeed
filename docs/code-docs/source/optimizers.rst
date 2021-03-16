Optimizers
===================

DeepSpeed offers high-performance implementations of ``Adam`` optimizer on CPU; ``FusedAdam``, ``FusedAdam``, ``OneBitAdam`` optimizers on GPU.

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
.. autoclass:: deepspeed.runtime.fp16.onebit.adam.OneBitAdam
