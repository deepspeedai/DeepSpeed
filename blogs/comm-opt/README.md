<div align="center">

# Communication Optimizations for Large-Scale Training

</div>


## Table of Contents
1. [Introduction](#introduction)
2. [Gradient AllReduce Optimization for ZeRO stages 1 and 2](#ar-opt)
3. [Optimizing Parameter All-Gather for ZeRO2 Training](#ag-opt)
4. [Optimizing AlltoAll for Sequence-Parallel Training](#sp-opt)


## 1. Introduction <a name="introduction"></a>
Training LLMs on large datasets can be extremely costly both in terms of hardware resources and time. An important step to minimize such costs is to carefully combine an appropriate number of resources together with a scalable library that guarantees training completion within a time limit. In this post, we discuss a key aspect of the scalability features of DeepSpeed, the communication optimization. Communication collectives (e.g., all-reduce, all-gather, etc.) are critical pieces of many popular DeepSpeed technologies (e.g., ZeRO, MoE, AutoTP, etc.), and in the following sections we discuss our new optimizations of some of these collectives. These optimizations are available in DeepSpeed versions >= 0.x.x.

## 2. Gradient AllReduce Optimization for ZeRO stages 1 and 2 <a name="ar-opt"></a>

Before diving into this optimization, let's take a step back and show some of the case studies that demonstrate the need.

AllReduce operation is an important part of the training process. In ZeRO, we handle this in buckets, which can be configured to get good communication throughput. As the number of GPUs increases, we encounter smaller-partition AllReduces. In this case, the current bucketing scheme cannot help with the communication overhead. This mostly becomes an issue when training smaller-scale models (like Llama-7B) with large number of GPUs.

For instance, when training a dense-7B architecture with Zero stages 1 or 2, we encounter a 1 and 2 second increase for the AllReduce time by increasing from 256 to 512 and 1024 A100 GPUs. This issue mostly arises from the fact that, the gradient-averaging happens with smaller partitions (#parameters / #GPUs) per-GPU rank. This issue gets more serious when training MoE architectures (3 - 12 second) for which the expert's parameters can be farther away due to the current parallelism layout of data and expert parallelism.

In this section, we introduce two main optimization techniques for alleviating these communication bottleneck.

First, Multi-rank bucketing for the same process group: for this optimization, we simply pack all data that requires to be reduced from different ranks into one big flattened tensor and call AllReduce instead of reduce operations. After the reduction, we scatter the right portion of data to the corresponding ranks.

Second, add new layout for the expert-data parallelism: the default parallelism layout for MoE architecture (as shown in Fig 1) is planned in a way that the experts are placed first on E parallel GPUs and replicated D times (data-parallel). With this layout, we encounter slower AllReduce as data-parallel ranks are placed farther away especially when we have cross-rank communication. We call this layout E + D.

<div align="center">
  <img src="assets/images/e+d.png" alt="" width=800 /><br>

  *Fig 1: Different MoE parallel layout. left) E + D, which places the GPUs in EP dimension first before adding DP, right) D + E, that replicates each expert by DP size, before constructing EP. We get faster AllReduce for the second layout while increasing the AlltoAll time. It potentially results in faster e2e training time, as the communication volume for AllReduce (total parameter size) is normally much more than AlltoAll (MLP activation memory).*<br>
</div>
By changing this layout from E + D to D + E (shown in Fig 1), where we first replicate each expert by D times and then add them across expert-parallel dimension, we can reduce the AllReduce time substantially. On an A100-DGX cluster, where each node has 8 GPUs, we see about 8x reduction in cross-node infiniband communication-volume for the parameter update process, which are now processed faster using the intra-node NVLinks. Note that by adding this optimization, we increase the cost of AlltoAll happening for the MoE part of the model, however, we have seen that the performance benefit of AllReduce overweighs this cost.

Table 1 summarizes the saving observed for training a 7B dense and a MoE architecture by using the optimized AllReduce scheme. After applying the multi-rank bucketing technique, we reduce the AllReduce time by 4x for dense architecture and 5x - 8x for the MoE one. In addition, we obtain an extra 3x saving using the new D + E layout for the MoE architecture. Therefore, we see higher performance gain on MoE architectures when using large number of GPUs. For instance, when training a 7B-base MoE architecture, we reduce iteration-time from 13 sec to 9.5 sec on 512 GPUs (37%) and from 16.1 sec to 5.1 sec on 1k-GPU setup (3.2x).
<div align="center">

|  | GPUs | AllReduce time | Iteration time |
|----------|:------:|:------:|:------:|
baseline (dense)	| 1024|	1.2 | 5.4
optimized (dense)	| 1024|	0.36 | 4.5
baseline (MoE)	| 1024 |	11.5 | 16.1
optimized (MoE)	| 1024	| 0.45 | 5.1

Table 1. AllReduce saving observed for both dense and MoE architectures.

</div>

## 3. Optimizing Parameter All-Gather for ZeRO2 Training <a name="ag-opt"></a>

The same as with AllReduce, all-gather takes longer as we have more partitions. As the parameters are stored in a flattened buffer for ZeRO stage-2, we can simply have a one call to all-gather the parameters into this tensor.

When all-gathering the updated parameters at Zero-Stage2, the bucketing scheme uses several narrow operations and creates a list of tensors with the bucket size from each partition. We needed this scheme to align with the `all_gather` operation from PyTorch.
However, by adding the support for the `all_gather_into_tensor`, operation that has been added to the newer versions of PyTorch, we can simply have a kernel call to do the full-parameter all-gather. With this optimization, we see about 2x reduction in the step time for large-scale training.

## 4. Optimizing AlltoAll for Sequence-Parallel Training <a name="sp-opt"></a>

For this part of the optimization, we add some fusion for the communication that is required for the DeepSpeed-Ulysses to provide a more scalable approach for when we increase the SP from 2 to 8 (for this study, we consider A100-DGX hardware, which has 8 GPUs per-node and by increasing the parallelism more than 8, we encounter performance-hit by the cross-node communication).

These fusions are done at two levels:
1. Fuse the sequence AlltoAll for q,k, and v: we Scatter the heads using the mixed tensor rather than splitting them beforehand. For this part, we need to get some more information from the modeling side (such as the number of q and kv heads), to split the heads before calling AlltoAll. We have added some new changes on the Megatron-DeepSpeed repo that incorporate these changes for the sequence-parallelism.
2. Fuse the AlltoAll tensors and call the PyTorch's AlltoAll-single API: we reshape the tensors for the scatter dimension and use a single tensor for AlltoAll which alleviates the overhead of using a list of tensors which requires a contiguous call for each element of the list.

By adding these optimizations, we see about 10 to 15% speedup compared to the previous design, and obtain good scalability across different SP-degree and context-lengths. In the following table, we show the improvement achieved by using SP, when doubling the GPU-count and increasing the SP-degree. We obtain over 80% of efficiency when increasing from 256 to 512 GPUs using SP-2. Furthermore, by increasing the sequence-length and SP, while keeping the processed tokens similar, we achieve over 75% of efficiency for 2x more resources. On the other hand, if we can double the number of tokens (shown on the last row of table 2), we can improve the performance to 1.81x.

<div align="center">

| GPUs | bsz | seq | Tokens (M) | SP | Sample (4K)-per-second | Speedup (x) |
|----------|:------:|:------:|:------:|:------:|:------:|:------:|
256	| 256|	8192	|2|1	| 60.71	 |1
512	| 256|	8192	|2|2	| 111.18 |	1.83
512	| 128|	16384 |2|4 | 108.81 |	1.79
512	| 64	|32768	|2|8	| 106.54 |	1.75
512	| 64	|65536	|4|8	| 110.05 |	1.81

Table 2. Sequence-Parallelism scalability using DeepSpeed-Ulysses.

</div>
