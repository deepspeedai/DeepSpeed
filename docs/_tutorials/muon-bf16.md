---
title: "Muon with BF16 weights and FP32 gradient accumulation"
---

Muon supports the dense data-parallel `BF16_Optimizer` path selected by BF16
weights, `data_types.grad_accum_dtype: fp32`, and ZeRO stage 1. Each optimizer
step first completes gradient accumulation and reduction, then global clipping,
then applies the existing standard or Gram Newton–Schulz implementation to each
original matrix. Auxiliary Adam groups retain their usual update path.

Committed momentum is FP32 and partitioned exactly like the master parameters;
ordinary optimizer checkpoints preserve it. The implementation temporarily
gathers one optimizer group's momentum, plus a local staging partition and the
matrix kernel workspace. This adds state communication and temporary device
memory: the largest Muon group determines the gather workspace. Parameters and
momentum are committed only to their local partition intersections; alignment
padding is excluded.

This initial path supports dense two-dimensional matrices and eager DP execution.
It rejects model-parallel MPU, expert groups, graph harvesting and other gradient
accumulation dtypes. CPU-offloaded Muon and topology-changing checkpoints are
outside this support boundary. Standard NS retains its BF16 arithmetic; Gram NS
retains its FP16 arithmetic and restart. Compilation can introduce small rounding
differences in the upstream NS implementation; bitwise cross-process equivalence
is not guaranteed.
