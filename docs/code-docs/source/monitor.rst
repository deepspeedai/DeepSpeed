Monitoring
==========

Deepspeed’s Monitor module can log training details into a
Tensorboard-compatible file, to WandB, or to simple CSV files. Below is an
overview of what DeepSpeed will log automatically.

.. csv-table:: Automatically Logged Data
    :header: "Field", "Description", "Condition"
    :widths: 20, 20, 10

    `Train/Samples/train_loss`,The training loss.,None
    `Train/Samples/lr`,The learning rate during training.,None
    `Train/Samples/loss_scale`,The loss scale when training using `fp16`.,`fp16` must be enabled.
    `Train/Eigenvalues/ModelBlockParam_{i}`,Eigen values per param block.,`eigenvalue` must be enabled.
    `Train/Samples/elapsed_time_ms_forward`,The global duration of the forward pass.,`flops_profiler.enabled` or `wall_clock_breakdown`.
    `Train/Samples/elapsed_time_ms_backward`,The global duration of the forward pass.,`flops_profiler.enabled` or `wall_clock_breakdown`.
    `Train/Samples/elapsed_time_ms_backward_inner`,The backward time that does not include the the gradient reduction time. Only in cases where the gradient reduction is not overlapped, if it is overlapped then the inner time should be about the same as the entire backward time.,`flops_profiler.enabled` or `wall_clock_breakdown`.
    `Train/Samples/elapsed_time_ms_backward_allreduce`,The global duration of the allreduce operation.,`flops_profiler.enabled` or `wall_clock_breakdown`.
    `Train/Samples/elapsed_time_ms_step`,The optimizer step time,`flops_profiler.enabled` or `wall_clock_breakdown`.

TensorBoard
-----------
.. _TensorBoardConfig:
.. autopydantic_model:: deepspeed.monitor.config.TensorBoardConfig

WandB
-----
.. _WandbConfig:
.. autopydantic_model:: deepspeed.monitor.config.WandbConfig

CSV Monitor
-----------
.. _CSVConfig:
.. autopydantic_model:: deepspeed.monitor.config.CSVConfig
