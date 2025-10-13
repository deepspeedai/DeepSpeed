# Study of ZenFlow and ZeRO offload performance with DeepSpeed CPU core binding
**TL;DR:** ZenFlow is an improvement to ZeRO Offload contributed to DeepSpeed by Tingfeng Lan et al. After testing this feature, we explored the relationship between ZenFlow performance and DeepSpeed CPU core binding.

## ZenFlow technology introduction
[ZenFlow](https://arxiv.org/abs/2505.12242) is a recent improvement to ZeRO Offload implemented in DeepSpeed. Its primary goal is to address the GPU stalls caused by ZeRO Offload. These stalls mainly originate from two sources: 1) the data transfer from the GPU to the CPU, which is limited by PCIe bandwidth, and 2) the computational overhead of executing the Adam optimizer on the CPU, which is constrained by CPU performance and memory bandwidth.

The core idea of ZenFlow is to separate gradients into two groups based on their norm. A very small portion of gradients, which have larger norms, are classified as important gradients and are updated directly on the GPU. The vast majority of gradients, which have smaller norms, are used to update the weights on the CPU at a lower frequency than the important gradients. If the gradients are not scheduled for an update in the current training iteration, they are accumulated into a copy of the gradients. These accumulated gradients are then used for the weight update in a subsequent iteration.

Furthermore, the weight updates on the CPU are designed to run in parallel with the computations on the GPU, thereby achieving the objective of reducing GPU stall.

To achieve the goal of parallelizing weight updates on the CPU with GPU computations, ZenFlow creates an additional process for each rank. This dedicated process handles the weight updates, while the original process for each rank can continue executing GPU computation code. This design enables the concurrency between weight updates and GPU computations.  In addition to these optimizations, ZenFlow also performs CPU core binding for the weight update processes. It binds the CPU update processes of different ranks to distinct CPU cores to enhance CPU performance.

## DeepSpeed CPU core binding feature and its improvement to CPU offloading performance
This reminds us that DeepSpeed itself supports CPU core binding through the `--bind_cores_to_rank` flag.  This switch was originally designed to improve multi-socket CPU inference performance. By binding cores, different workers can run on distinct CPU cores without interfering with each other, thereby enhancing locality.  Additionally, DeepSpeed's core binding feature automatically configures the `OMP_NUM_THREADS` environment variable to ensure the OpenMP thread pool size matches the number of allocated cores.

This raised a question: Could this switch also benefit ZeRO Offload?  We conducted tests to explore this possibility.

### Improvement to ZeRO Offload performance from DeepSpeed CPU core binding
|             | Avg. time of first 51 iterations (1st run) | 2nd run | 3rd run | Average |
|-------------|--------------------------------------------|---------|---------|---------|
| No bind core| 2707.32ms | 3127.24ms | 2826.04ms | 2887ms |
| Bind core   | 2649.06ms | 2641.82ms | 2200.76ms | 2497ms |

**Model:** Qwen2.5-3B

**Test environment:** 2xDGX-A100-SXM4-40GB, 2xAMD EPYC 7742 64-Core Processor, 1TB memory

**Test URL:** [DeepSpeedExamples/training/DeepSpeed-ZenFlow/finetuning](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/DeepSpeed-ZenFlow/finetuning) (All following tests are using the same URL)

**Test command:**
  - **No core binding:** `deepspeed --num_gpus=2 finetune_llama.py --model_name Qwen/Qwen2.5-3B --output_dir output --lr 2e-5 --batch_size 8 --deepspeed_config zo_config.json --num_train_epochs 1`
  - **With core binding:** `deepspeed --num_gpus=2 --bind_cores_to_rank finetune_llama.py --model_name Qwen/Qwen2.5-3B --output_dir output --lr 2e-5 --batch_size 8 --deepspeed_config zo_config.json --num_train_epochs 1`

**Config file** (`zo_config.json`):
```json
{
    "train_batch_size": 8,
    "bf16": { "enabled": true },
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      }
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "zero_allow_untested_optimizer": true,
    "wall_clock_breakdown": true
}
```

From this data, DeepSpeed's core binding provides approximately a 15% performance improvement for ZeRO Offload. So, could it also benefit ZenFlow's performance? With this question in mind, we decided to comment out the core binding logic within ZenFlow and instead directly use the `--bind_cores_to_rank` flag to run ZenFlow:

### Improvement to ZenFlow performance from DeepSpeed CPU core binding
|                    | Avg. time from iteration 5-51 (1st run) | 2nd run | 3rd run | Average |
|--------------------|-----------------------------------------|---------|---------|---------|
|ZenFlow core binding| 1337.66ms | 1443.87ms | 1475.04ms | 1419ms |
|DeepSpeed core binding| 1233.6ms | 1228.36ms | 1235ms | 1232ms |

**Model:** Qwen2.5-3B

**Test environment:** 2xDGX-A100-SXM4-40GB, 2xAMD EPYC 7742 64-Core Processor, 1TB memory

**DeepSpeed commit:** 1d7b90adc48d57c2283e8825f5c668a3730ff899

*ZenFlow use 4 iterations to compute gradient importance, so we start from 5th iteration to measure time*

**Test command:**
  - **No core binding:** `deepspeed --num_gpus=2 finetune_llama.py --model_name Qwen/Qwen2.5-3B --output_dir output --lr 2e-5 --batch_size 8 --deepspeed_config zf_config.json --num_train_epochs 1`
  - **With core binding:** `deepspeed --num_gpus=2 --bind_cores_to_rank finetune_llama.py --model_name Qwen/Qwen2.5-3B --output_dir output --lr 2e-5 --batch_size 8 --deepspeed_config zf_config.json --num_train_epochs 1`


**Config file** (`zf_config.json`):
```json
{
    "train_batch_size": 8,
    "bf16": { "enabled": true },
    "zero_optimization": {
      "stage": 2,
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      },
      "zenflow": {
            "topk_ratio": 0.1,
            "update_interval": 4,
            "full_warm_up_rounds": 0,
            "overlap_step": true,
            "pt_reserved_cores_perc": 0.5
        }
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 2e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "zero_allow_untested_optimizer": true
}
```


We observed a performance improvement of approximately 15% from DeepSpeed CPU core binding against ZenFlow core binding. Why did this happen?

## Our improvements to ZenFlow CPU core binding mechanism
After communicating with the authors of ZenFlow, we gained a new understanding of the core binding mechanism required by ZenFlow.

First, the ZenFlow worker processes need to use a dedicated set of CPU cores, separate from those used by the main process of each rank. Second, the ZenFlow workers and the main processes should be bound to different physical cores, avoiding binding to virtual cores (hyper-threads). Third, the OpenMP thread pool size should be appropriately set to match the number of cores allocated to the ZenFlow workers.

In the original ZenFlow implementation, all cores (including the virtual cores corresponding to physical cores) were used for core binding, meaning the workers were not properly isolated at the physical core level. In contrast, DeepSpeed's core binding specifically binds processes to physical cores only, which explains the performance improvement we observed.

Based on this understanding, we collaborated with the ZenFlow authors to update its core binding mechanism.

First, before each rank launches a ZenFlow worker process, it needs to enumerate the list of available physical cores. If these lists of physical cores differ across ranks, it indicates that DeepSpeed has already performed physical core binding. Otherwise, each rank needs to allocate its own list of available cores from the total pool.

Finally, each rank allocates a subset of cores from its own list to the ZenFlow worker process and sets the corresponding `OMP_NUM_THREADS` environment variable. This ensures that all processes use distinct CPU cores, preventing interference, and also allows for proper configuration of the OpenMP thread pool size. [code](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zenflow/zenflow_stage_1_and_2.py)

Under this new core binding mechanism, we re-evaluated the performance of ZenFlow:

### ZenFlow perf. with new core binding mechanism
|                    | Avg. time from iteration 5-51 (1st run) | 2nd run | 3rd run | Average | Improvement over original binding |
|--------------------|-----------------------------------------|---------|---------|---------|------|
| New ZenFlow worker core binding | 1321.21ms | 1269.83ms | 1384.47ms | 1325ms | 7% |
| DeepSpeed core binding + new ZenFlow worker core binding | 1111.68ms | 1125.38ms | 1111.91ms | 1116ms | 10% |

**Model:** Qwen2.5-3B

**Test environment:** 2xDGX-A100-SXM4-40GB, 2xAMD EPYC 7742 64-Core Processor, 1TB memory

**DeepSpeed commit:** 80033a82938f6cd8ce4988a63c914941e7a8f324

The results indicate that ZenFlow's performance was further enhanced under the new core binding mechanism. Compared to the original binding method, performance improved by 7% when not using DeepSpeed's core binding. When DeepSpeed's core binding was enabled, the performance gain reached 10%.

Why does DeepSpeed binding still provide an additional performance boost on top of the new ZenFlow binding?

We initially hypothesized that it might be because DeepSpeed uses numactl, which can bind a process to a specific NUMA node, ensuring the process always accesses local memory. However, upon examining the DeepSpeed logs, we found that the -m switch was not enabled during runtime. Furthermore, when we replaced numactl with taskset, we still observed the performance improvement.

Our current conjecture is that the difference lies in how the binding is applied. numactl (and taskset in this context) operates at the process level, applying the binding to the entire process from the start. In contrast, ZenFlow's binding is applied within the code at the point of use. This distinction in the scope and timing of the binding application could be the source of the performance difference. This point may require more detailed investigation in the future.

Regardless, the key finding remains: the new ZenFlow binding mechanism improves performance irrespective of whether DeepSpeed binding is used. This conclusively demonstrates the effectiveness of physical core isolation for performance.

We conducted a comparative analysis of the performance across several configurations: ZeRO Offload without core binding, ZeRO Offload with core binding, and ZenFlow both before and after our improvements. The results are summarized as follows:

### Perf comparison table
|             | Average time | Perf. improv. vs. baseline |
|-------------|--------------|----------------------------|
| ZeRO Offload without binding -- baseline | 2887ms | 1x |
| ZeRO Offload with DeepSpeed core binding | 2497ms | 1.16x |
| ZenFlow original worker core binding | 1419ms | 2.03x |
| DeepSpeed core binding +ZenFlow new worker core binding | 1116ms | 2.59x |

**Model:** Qwen2.5-3B

**Test environment:** 2xDGX-A100-SXM4-40GB, 2xAMD EPYC 7742 64-Core Processor, 1TB memory

The result clearly shows that the improved ZenFlow achieves a 2.59x speedup compared to ZeRO Offload without core binding, and a 2.24x speedup compared to ZeRO Offload with core binding.

Given that ZenFlow's core innovations involve reducing the frequency of weight updates and parallelizing CPU/GPU execution, the 2.24x improvement over the core-bound ZeRO Offload is particularly significant. This comparison provides a more accurate reflection of ZenFlow's inherent performance advantages. By using the core-bound ZeRO Offload as the baseline, we effectively isolate and quantify the performance gains attributable specifically to ZenFlow's algorithmic optimizations, rather than those coming from general core-binding techniques. This strongly validates the effectiveness of ZenFlow's fundamental design.

Through our collaboration with the ZenFlow authors, the new core-binding mechanism has been integrated into the main branch of DeepSpeed. As a result, users can now achieve optimal offload performance by simply using ZenFlow in conjunction with the DeepSpeed `--bind_cores_to_rank` flag. This integration provides an out-of-the-box, high-performance experience that leverages the combined strengths of both the algorithmic innovations in ZenFlow and the low-level system optimizations in DeepSpeed's core binding.

## Practicality metric, a metric to evaluate offloading technology
In addition to comparisons with ZeRO Offload, a performance comparison against scenarios without offloading better demonstrates the practicality of ZenFlow or ZeRO Offload. While it's true that ZeRO Offload or ZenFlow enables model optimization with relatively limited VRAM, achieving a breakthrough from impossibility to possibility, if the performance gap is too significant, the decision to use offloading becomes a dilemma. We consider the performance difference between scenarios with and without offloading as a practicality metric. A value of 1 represents the ideal scenario, indicating that offloading has no impact on performance. The smaller this value, the poorer the practicality, as users would need to wait considerably longer for fine-tuning.

Since we couldn't run Qwen2.5-3B with ZeRO2 using the same config on two GPUs in our test environment, we conducted the practicality test using Qwen2.5-1.5B instead:

### Practicality test
|  | Average time | Practicality metric |
|-------------|--------------|---------------------|
| ZeRO2 | 240ms | |
| ZeRO Offload with DeepSpeed core binding | 1365ms | 17.6% |
| DeepSpeed core binding + new ZenFlow worker core binding | 569ms | 42.2% |

**Model: Qwen2.5-1.5B**

**Test environment:** 2xDGX-A100-SXM4-40GB, 2xAMD EPYC 7742 64-Core Processor, 1TB memory

Based on the tests conducted on 2xA100 GPUs, the practicality metric for ZeRO Offload was 17.6%, while ZenFlow achieved a practicality metric of 42.2%. This result demonstrates that ZenFlow significantly improves the practicality of offloading.

## Summary
ZeRO Offload is an effective technique for reducing VRAM pressure, making the fine-tuning of large models possible. We have now seen that ZenFlow, as a new technology, achieves a breakthrough improvement in the practicality of ZeRO Offload, bringing it to a usable level. When combined with DeepSpeed's core binding, ZenFlow is able to deliver its optimal performance.

## Disclaimer
All performance data presented in this article is measured for the sole purpose of discussing the effects of specific optimization techniques. There is no guarantee that the data was obtained under optimal software or hardware configurations, nor does it represent a performance evaluation of any software or hardware products mentioned. This article discusses only the relative performance changes resulting from specific optimization methods.  The performance gain depends on specific software or hardware configuration and may vary in your own run.
