Learning Rate Schedulers
===================

DeepSpeed offers implementations of ``LRRangeTest``, ``OneCycle``, ``WarmupLR``, ``WarmupDecayLR``, ``WarmupCosineLR`` learning rate schedulers. When using a DeepSpeed's learning rate scheduler (specified in the `ds_config.json` file), DeepSpeed calls the `step()` method of the scheduler at every training step (when `model_engine.step()` is executed). When not using a DeepSpeed's learning rate scheduler:
  * if the schedule is supposed to execute at every training step, then the user can pass the scheduler to `deepspeed.initialize` when initializing the DeepSpeed engine and let DeepSpeed manage it for update or save/restore.
  * if the schedule is supposed to execute at any other interval (e.g., training epochs), then the user should NOT pass the scheduler to DeepSpeed during initialization and must manage it explicitly.

LRRangeTest
---------------------------
.. autoclass:: deepspeed.runtime.lr_schedules.LRRangeTest


OneCycle
---------------------------
.. autoclass:: deepspeed.runtime.lr_schedules.OneCycle


WarmupLR
---------------------------
.. autoclass:: deepspeed.runtime.lr_schedules.WarmupLR


WarmupDecayLR
---------------------------
.. autoclass:: deepspeed.runtime.lr_schedules.WarmupDecayLR


WarmupCosineLR
---------------------------
.. autoclass:: deepspeed.runtime.lr_schedules.WarmupCosineLR
