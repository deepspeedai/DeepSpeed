Optimizers
===================

DeepSpeed offers high-performance implementations of ``Adam`` optimizer on CPU; ``FusedAdam``, ``FusedLamb`` optimizers on GPU.

Adam (CPU)
----------------------------
.. autoclass:: deepspeed.ops.adam.DeepSpeedCPUAdam

Reflow Adam (CPU, ZeRO-3 asynchronous offload)
----------------------------------------------
.. autoclass:: deepspeed.runtime.reflow.reflow_cpu_adam.ReflowCPUAdam

Reflow Lion (CPU, ZeRO-3 asynchronous offload)
----------------------------------------------
.. autoclass:: deepspeed.runtime.reflow.reflow_cpu_lion.ReflowCPULion

FusedAdam (GPU)
----------------------------
.. autoclass:: deepspeed.ops.adam.FusedAdam

FusedLamb (GPU)
----------------------------
.. autoclass:: deepspeed.ops.lamb.FusedLamb
