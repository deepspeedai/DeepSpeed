---
title: "Using PyTorch Profiler with DeepSpeed for performance debugging"
tags: profiling performance-tuning
---

This tutorial describes how to use [PyTorch Profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/) with DeepSpeed.

PyTorch Profiler is an open-source tool that enables accurate and efficient performance analysis and troubleshooting for large-scale deep learning models.  The profiling results can be outputted as a `.json` trace file and viewed in Google Chrome's trace viewer (chrome://tracing).
Microsoft Visual Studio Code's Python extension integrates TensorBoard into the code editor, including the support for the PyTorch Profiler.

For more details, refer to [PYTORCH PROFILER](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#pytorch-profiler).

## Profile the model training loop

Below shows how to profile the training loop by wrapping the code in the profiler context manager. The Profiler assumes that the training process is composed of steps (which are numbered starting from zero). PyTorch profiler accepts a number of parameters, e.g. `schedule`, `on_trace_ready`, `with_stack`, etc.

In the example below, the profiler will skip the first `5` steps, use the next `2` steps as the warm up, and actively record the next `6` steps. The profiler will stop the recording after the first two cycles since `repeat` is set to `2`.
For the detailed usage of the `schedule`, please refer to [Using profiler to analyze long-running jobs](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs).

```python
from torch.profiler import profile, record_function, ProfilerActivity

with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=5, # During this phase profiler is not active.
        warmup=2, # During this phase profiler starts tracing, but the results are discarded.
        active=6, # During this phase profiler traces and records data.
        repeat=2), # Specifies an upper bound on the number of cycles.
    on_trace_ready=tensorboard_trace_handler,
    with_stack=True # Enable stack tracing, adds extra profiling overhead.
) as profiler:
    for step, batch in enumerate(data_loader):
        print("step:{}".format(step))

        #forward() method
        loss = model_engine(batch)

        #runs backpropagation
        model_engine.backward(loss)

        #weight update
        model_engine.step()
        profiler.step() # Send the signal to the profiler that the next step has started.
```

## Label arbitrary code ranges

The `record_function` context manager can be used to label arbitrary code ranges with user provided names. For example, the following code marks `"model_forward"` as a label:

```python
with profile(record_shapes=True) as prof: # record_shapes indicates whether to record shapes of the operator inputs.
    with record_function("""):"
        model_engine(inputs)
```

## Profile CPU or GPU activities

The `activities` parameter passed to the Profiler specifies a list of activities to profile during the execution of the code range wrapped with a profiler context manager:
- ProfilerActivity.CPU - PyTorch operators, TorchScript functions and user-defined code labels (`record_function`).
- ProfilerActivity.CUDA - on-device CUDA kernels.
**Note** that CUDA profiling incurs non-negligible overhead.

The example below profiles both the CPU and GPU activities in the model forward pass and prints the summary table sorted by total CUDA time.

```python
with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_forward"):
        model_engine(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```


## Profile memory consumption

By passing `profile_memory=True` to PyTorch profiler, we enable the memory profiling functionality which records the amount of memory (used by the model’s tensors) that was allocated (or released) during the execution of the model’s operators. For example:

```python
with profile(activities=[ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```

`self` memory corresponds to the memory allocated (released) by the operator, excluding the children calls to the other operators.
