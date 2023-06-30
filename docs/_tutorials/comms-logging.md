---
title: "Communication Logging"
excerpt: "Log all DeepSpeed communication calls"
tags: profiling performance-tuning
---

In this tutorial, we introduce DeepSpeed communication logging and provide examples of its usage.

  - [Overview](#overview)
  - [Usage](#usage)

## Overview

NOTE: All logging communication calls are synchronized in order to provide accurate timing information. This may hamper performance if your model heavily uses asynchronous communication operations.

Logging communication calls is vital to ensure networking resources are fully utilized. The DeepSpeed communication logger enables the detection and logging of all communication operations launched under `deepspeed.comm`. Each communication operation can all be directly printed to the console immediately after completion (via the `verbose` config option), or a summary may be printed with a call to `deepspeed.comm.log_summary()` or `deepspeed.com.log_summary(show_straggler=True)` in the client code at the completion of training, an epoch, after N training iterations, etc.

## Usage

Communication logging in DeepSpeed is configured within the deepspeed [configuration file](/docs/config-json/#communication-logging). DeepSpeed will automatically log communication either all operations (`prof_all`), or user-specified operations (`prof_ops`).

  - [Configuration Setup](#configuration-setup)
  - [Verbose Logging](#verbose-logging)
  - [Log Summaries](#log-summaries)

### Configuration Setup

Communication logging can be configured in the DeepSpeed [configuration file](/docs/config-json/#communication-logging). Communication logging can be enabled by adding the following field to DeepSpeed's configuration json file. Refer to [Communication Logging](/docs/config-json/#communication-logging) for details.

```json
"comms_logger": {
  "enabled": true,
  "verbose": false,
  "prof_all": true,
  "debug": false
}
```

There are currently two ways to view communication log records:

1. Print all communication operations with `verbose` config option. See [Verbose Logging](#verbose-logging)
2. (Recommended) Print log summary with `deepspeed.comm.log_summary()` function call. See [Log Summaries](#log-summaries)

### Verbose Logging

If the `enabled` configuration option is selected, all communication operations will be immediately printed to the console. This mode is intended for detailed debugging, and is not recommended for most users. The following is an example snippet of `verbose` output:

```
[2022-06-26 01:39:55,722] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 | comm op: reduce_scatter_tensor | time (ms): 9.46 | msg size: 678.86 MB | algbw (Gbps): 1204.52  | busbw (Gbps): 1129.23
[2022-06-26 01:39:56,470] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 | comm op: all_gather_into_tensor | time (ms): 0.11 | msg size: 6.0 MB | algbw (Gbps): 954.41  | busbw (Gbps): 894.76
[2022-06-26 01:39:56,471] [INFO] [logging.py:69:log_dist] [Rank 0] rank=0 | comm op: all_gather_into_tensor | time (ms): 0.08 | msg size: 6.0 MB | algbw (Gbps): 1293.47  | busbw (Gbps): 1212.63
```

For advanced users, the `debug` option will append the calling function of each communication operation to that operation's `log_name`. See [Log Summaries](#log-summaries) for an example of a `deepspeed.comm.log_summary()` call with `debug` enabled.


### Log Summaries

It's recommended that users add a call to `deepspeed.comm.log_summary()` at training milestones (e.g. every epoch or N iterations). This enables high-level communication logging without having to sift through logs from `verbose`.

The steps to add DeepSpeed communication log summaries are as follows:

1. Modify configuration file with desired settings
2. (Optional) If your application contains `torch.distributed` calls that you wish to log, import `deepspeed.comm` package and modify `torch.distributed` calls to use `deepspeed.comm` (Note: The `deepspeed.comm` collective and pt2pt APIs exactly match `torch.distributed`)
3. Call `deepspeed.comm.log_summary`

For example usage, see the following modified [DeepSpeedExamples/cifar](https://github.com/microsoft/DeepSpeedExamples/tree/master/cifar) example:

```python
# Step 2: (Optional) Import deepspeed.comm
import deepspeed.comm as dist

# Note that any communication operations using `import torch.distributed as dist` calls can remain unchanged, and will be automatically logged under deepspeed.comm!
dist.all_reduce(tensor)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        pre = time.time()
        inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
            model_engine.local_rank)
        if fp16:
            inputs = inputs.half()
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)

        model_engine.backward(loss)
        model_engine.step()
        post = time.time()
    # Step 3: Call `deepspeed.comm.log_summary()`
    dist.log_summary()
```

The following is a truncated example output of `deepspeed.comm.log_summary()` at the end of 10 iterations of Megatron-DeepSpeed with ZeRO-3:

```
Comm. Op            Message Size        Count               Total Latency(ms)   Avg Latency(ms)     tput_avg (Gbps)     busbw_avg (Gbps)
broadcast
                    2.0 KB              146                 11.12               0.08                0.43                0.41
                    98.25 MB            1                   8317.12             8317.12             0.20                0.19
reduce_scatter_tensor
                    678.86 MB           40                  602.29              9.69                1468.06             1376.31
```


And the following is a call to `deepspeed.comm.log_summary` under the same configuration with `debug` enabled:

```
Comm. Op            Message Size        Count               Total Latency(ms)   Avg Latency(ms)     tput_avg (Gbps)     busbw_avg (Gbps)
broadcast | [Caller Func: _broadcast_model]
                    2.0 KB              146                 9.39                0.06                0.52                0.48
                    98.25 MB            1                   8540.60             8540.60             0.19                0.18
reduce_scatter_tensor | [Caller Func: reduce_scatter_fn]
                    678.86 MB           80                  1527.17             13.94               1211.75             1136.01
```

Straggler effect can be shown by supplying optional argument `show_straggler=True` to `deepspeed.comm.log_summary()` call.   Straggler effect is defined as the time a rank waits for the slowest rank to start communication.  For each collective, `log_summary` would get the minimum collective time among all ranks, compute straggler effect as follows:

```
straggler = sum(t_collectives - allreduce(t_collectives, MIN))
```

Print straggler effect with the following `log_summary` call in the example above:
```
    dist.log_summary(show_straggler=True)
```
