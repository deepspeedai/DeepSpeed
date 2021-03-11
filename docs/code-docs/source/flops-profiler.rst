Flops Profiler

==============

The flops profiler in DeepSpeed profiles the forward pass of a model and measures its parameters, latency, and floating point operations. The DeepSpeed flops profiler can be used with the DeepSpeed runtime or as a standalone package.

When using DeepSpeed for model training, the flops profiler can be configured in the deepspeed_config file without user code changes. To use the flops profiler outside of the DeepSpeed runtime, one can simply install DeepSpeed and import the flops_profiler package to use the APIs directly.

Please see the `Flops Profiler tutorial <https://www.deepspeed.ai/tutorials/flops-profiler/>`_ for usage details.

Flops Profiler
---------------------------------------------------

.. automodule:: deepspeed.profiling.flops_profiler.profiler
   :members:
   :show-inheritance:
