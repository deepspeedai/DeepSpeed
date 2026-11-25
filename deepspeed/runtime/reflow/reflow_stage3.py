# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Container, List, NamedTuple

import psutil
import torch

import deepspeed.comm as dist
from deepspeed.comm import pg_utils
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.reflow.ready_batch import ReadyBatch
from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
from deepspeed.runtime.reflow import reflow_utils
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum
from deepspeed.runtime.zero.partition_parameters import Parameter, Tensor
from deepspeed.runtime.zero.partitioned_param_coordinator import iter_params
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3, INIT_OPTIMIZER_TIMER
from deepspeed.runtime.zero.utils import get_norm_dtype
from deepspeed.runtime.utils import (combine_grad_norm_groups, has_inf_or_nan, is_invalid_grad_norm,
                                     mask_nan_or_inf_with_val_inplace)
from deepspeed.utils import logger, z3_leaf_module
from deepspeed.utils.nvtx import instrument_w_nvtx

_REFLOW_FLAT_GRAD_MSG = "Reflow uses its own gradient buffers; {api} is not supported under reflow."
_DEFAULT_BUCKETWISE_SHARD_ELEMS = 50000000
# A power of two, so every full slice is a whole number of SIMD blocks and the kernel's scalar tail is the same
# elements as in an unsliced call; the sliced commit stays bit-exact.
_STATE_COMMIT_SLICE_ELEMS = 1 << 25


class _StagedReady(NamedTuple):
    sub_group_id: int
    ranges: list


class ReflowOptimizer_Stage3(DeepSpeedZeroOptimizer_Stage3):
    """ZeRO-3 optimizer subclass adding the Reflow async CPU-offload feature.

    Adds three behaviors to ZeRO-3 stage-3 offloading, driven by a
    ``ReflowCPUAdam`` optimizer:
      - async_state: split the parameter update from the optimizer-state update so the
        state update runs on a background CPU worker thread.
      - cpu_conversion: send FP16/BF16 grads to the CPU optimizer, which promotes them to
        FP32 internally, avoiding an extra host-side cast.
      - cpu_bucketwise: run the optimizer per gradient bucket as soon as it is ready
        during backward, instead of once after backward completes.

    Half-precision (FP16/BF16) gradients are promoted to FP32 inside the
    CPU-conversion kernel path, where they must be FP32 before the Adam update.
    """

    def __init__(self, module, init_optimizer, param_names, timers, ds_config, reflow_config, **kwargs):
        # zero.Init models bypass the engine's parameter cast, so validate their actual dtype too.
        if any(param.is_floating_point() and param.dtype not in (torch.float16, torch.bfloat16)
               for param in module.parameters()):
            raise ValueError("Reflow requires FP16 or BF16 model parameters; FP32 model parameters "
                             "are not supported.")
        self._reflow_config = reflow_config
        # These attributes must exist before super().__init__ runs: the parent constructor
        # calls initialize_optimizer_states, which sizes the Reflow gradient slots.
        self._cpu_grad_double_buffers = []
        self._active_grad_buffer_idx = 0
        self._pending_grad_ready = False
        self._reflow_window_grad_ids = set()
        self._reflow_boundary_grad_ids = set()
        self._reflow_gradient_partitions = None
        self._reflow_checkpoint_visits = {}
        self._reflow_modified_gradient_ids = set()
        self._ipg_read_events = {}
        self._last_state_update_future = None
        self._pending_state_updates_by_subgroup = {}
        self._state_commit_overlaps_forward = False
        self._main_thread_affinity = None
        # Prototype, not fully implemented; the double grad buffer is the default. With offload_param='cpu' one
        # grad slot only duplicates fp16_partitioned_groups_flat, so this writes the FP16 params there and saves
        # ~2 bytes/param. Read before super().__init__, whose initialize_optimizer_states sizes the buffers.
        # Unsupported runs fall back with a warning (see _resolve_single_grad_buffer_eligibility).
        self._single_grad_buffer = os.getenv("REFLOW_SINGLE_GRAD_BUFFER", "0") == "1"

        super().__init__(module, init_optimizer, param_names, timers, ds_config, **kwargs)
        if self.dtype == torch.float16:
            # Accumulate overflow before D2H. ZeRO-3 reduces and clears this flag in has_overflow().
            with get_accelerator().stream(self.reduce_and_partition_stream):
                self.inf_or_nan_tracker = torch.zeros(1,
                                                      dtype=torch.uint8,
                                                      device=get_accelerator().current_device_name())
        self._track_checkpointed_parameters()
        self._warn_per_param_partition_groups()
        # FP16/BF16 grads go straight to the CPU optimizer, which promotes them to FP32
        # internally, so the accumulation dtype stays half here.
        self.gradient_accumulation_dtype = self.dtype
        # The CPU optimizer is submitted per bucket during backward, so blocking after the base default of two
        # in-flight reduce events stalls the main thread that dispatches it.
        self.max_param_reduce_events = max(self.max_param_reduce_events, 4)
        # Dedicated GPU->CPU copy stream so grad copies overlap with reduction; on synchronized
        # devices (no async streams) this is just the default stream.
        self.copy_grad_stream = get_accelerator().Stream() if get_accelerator().is_synchronized_device() is False \
            else get_accelerator().default_stream()

        self._setup_optimizer_workers()
        # The stream-level waits in step() do not keep the next forward's all-gather from reading a partition
        # before its H2D write-back lands; without this host sync OPT-30B showed loss spikes. Leave it on except
        # for measurement.
        self._sync_bucketwise_h2d_before_forward = os.getenv("DS_REFLOW_SYNC_H2D_BEFORE_FORWARD",
                                                             "1") not in ("0", "false", "off")
        # Pending-range bookkeeping is sized to the number of fp16 groups, which only exist
        # after super().__init__, so initialize it here.
        num_groups = len(self.fp16_groups)
        self._bucketwise_pending_ranges_by_group = [[] for _ in range(num_groups)]
        self._bucketwise_pending_elems_by_group = [0] * num_groups

    @property
    def _pending_grad_buffer_idx(self):
        # Single-buffer mode keeps grads (and reads) in the lone slot 0; otherwise the pending
        # double-buffer index is the complement of the active one.
        if self._single_grad_buffer:
            return 0
        return 1 - self._active_grad_buffer_idx

    # ------------------------------------------------------------------ #
    # __init__ helpers
    # ------------------------------------------------------------------ #
    def _track_checkpointed_parameters(self):
        # Reentrant checkpoint forwards run with gradients disabled. A later
        # recomputation must read the original weights, even if an earlier
        # checkpoint segment has already produced a gradient for the same param.
        original_forward = self.parameter_offload.pre_sub_module_forward_function
        self._reflow_original_pre_sub_module_forward = original_forward

        def track_forward(sub_module):
            if not torch.is_grad_enabled():
                for param in iter_params(sub_module, recurse=z3_leaf_module(sub_module)):
                    if hasattr(param, "ds_id") and param.requires_grad:
                        param_id = self.get_param_id(param)
                        last_step, _ = self._reflow_checkpoint_visits.get(param_id, (None, False))
                        self._reflow_checkpoint_visits[param_id] = (self.micro_step_id,
                                                                    last_step == self.micro_step_id)
            return original_forward(sub_module)

        self.parameter_offload.pre_sub_module_forward_function = track_forward

    def _warn_per_param_partition_groups(self):
        # The grad-norm reduction follows the base per-subgroup process groups, but the rest of the Reflow pipeline
        # has only been validated with every subgroup partitioned over the data-parallel group.
        for sub_group in self.fp16_groups:
            for param in sub_group:
                is_autoep_expert = getattr(param, "ds_zero_placement_family", None) == "autoep_expert"
                has_own_partition_group = getattr(param, "ds_zero_partition_process_group", None) is not None
                if is_autoep_expert or has_own_partition_group:
                    if dist.get_rank() == 0:
                        logger.warning("Reflow with per-parameter ZeRO-3 partition groups (e.g. AutoEP expert "
                                       "parallelism) is not validated and may produce wrong results.")
                    return

    def _setup_optimizer_workers(self):
        self._reset_bucketwise_handles()
        self._worker_cpu_affinity_mask = None
        self._state_update_cpu_affinity_mask = None

        main_thread_cores = self.main_thread_cores
        enable_cpu_affinity = self.enable_cpu_affinity
        # Automatically find NUMA-local cores for this rank, then split them into a main-thread set
        # and per-NUMA-node optimizer-worker groups (so workers never straddle a NUMA boundary).
        total_cpus, available_cores, node_worker_groups, main_cores, worker_cores = \
            reflow_utils.plan_cpu_core_layout(main_thread_cores, self._reflow_config.main_thread_core_type,
                                             self._reflow_config.worker_core_type,
                                             self._reflow_config.pin_main_thread)
        self._worker_node_groups = node_worker_groups

        self._worker_cpu_affinity_mask = set(worker_cores)
        state_update_core_count = self.state_update_cores
        if state_update_core_count is None:
            state_update_cores = worker_cores
        else:
            state_update_cores = worker_cores[-state_update_core_count:]
        self._state_update_cpu_affinity_mask = set(state_update_cores) if state_update_cores else None
        backward_count = self._reflow_config.state_update_backward_cores
        backward_cores = worker_cores[-backward_count:] if backward_count is not None else worker_cores
        self._state_backward_cpu_affinity_mask = set(backward_cores)
        # A running Persist slice may still use its forward mask when backward starts.
        self._state_commit_reserved_cpus = set(state_update_cores).union(backward_cores)

        original_affinity = None
        if self._reflow_config.pin_main_thread:
            if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
                raise ValueError("Reflow pin_main_thread requires OS thread-affinity support")
            original_affinity = (threading.get_native_id(), os.sched_getaffinity(0))

        if enable_cpu_affinity and available_cores:
            try:
                psutil.Process().cpu_affinity(available_cores)
                if dist.get_rank() == 0:
                    logger.info(f"CPU affinity: main process on {len(available_cores)} NUMA-local cores, "
                                f"{len(main_cores)} reserved from workers {sorted(main_cores)}")
            except (AttributeError, OSError, psutil.AccessDenied) as e:
                if dist.get_rank() == 0:
                    logger.warning(f"Failed to set main process CPU affinity: {e}")

        if original_affinity is not None:
            os.sched_setaffinity(0, main_cores)
            self._main_thread_affinity = original_affinity

        if dist.get_rank() == 0:
            logger.info(f"Reflow NUMA-aware CPU allocation: {total_cpus} rank cores across "
                        f"{len(node_worker_groups)} NUMA worker group(s); "
                        f"{len(main_cores)} main-thread cores {sorted(main_cores)}, "
                        f"{len(worker_cores)} optimizer-worker cores {sorted(worker_cores)}"
                        f"{', affinity isolation ENABLED' if enable_cpu_affinity else ''}")

        self._setup_bucketwise_executors()

    def _setup_bucketwise_executors(self):
        cores_per_worker = self.bucketwise_cores_per_worker

        try:
            node_groups = getattr(self, "_worker_node_groups", None)
            if not node_groups:
                fallback_cores = sorted(list(
                    self._worker_cpu_affinity_mask)) or reflow_utils.resolve_available_cores()[1]
                node_groups = [fallback_cores]

            if cores_per_worker is None:
                self._bucketwise_worker_cpu_affinity_masks = [set().union(*node_groups)]
            else:
                self._bucketwise_worker_cpu_affinity_masks = []
                for node_cores in node_groups:
                    for start in range(0, len(node_cores), cores_per_worker):
                        self._bucketwise_worker_cpu_affinity_masks.append(
                            set(node_cores[start:start + cores_per_worker]))
            if not any(self._bucketwise_worker_cpu_affinity_masks):
                self._bucketwise_worker_cpu_affinity_masks = [set(sorted(self._worker_cpu_affinity_mask) or [0])]

            max_bucketwise_workers = len(self._bucketwise_worker_cpu_affinity_masks)
            total_worker_cores = sum(len(node) for node in node_groups)
            if dist.get_rank() == 0:
                logger.info(f"Bucketwise worker configuration (NUMA-aware): {max_bucketwise_workers} workers "
                            f"across {len(node_groups)} NUMA node group(s), "
                            f"{cores_per_worker or total_worker_cores} cores per worker, "
                            f"total {total_worker_cores} worker cores")
        except Exception as e:
            max_bucketwise_workers = 1
            fallback_mask = self._worker_cpu_affinity_mask if self._worker_cpu_affinity_mask else {0}
            self._bucketwise_worker_cpu_affinity_masks = [fallback_mask]
            if dist.get_rank() == 0:
                logger.warning(f"Failed to configure bucketwise worker cores, using default: {e}")

        initializer = None
        if self._reflow_config.bucketwise_worker_affinity == "thread":
            masks = iter(self._bucketwise_worker_cpu_affinity_masks)
            initializer_lock = threading.Lock()

            def initializer():
                with initializer_lock:
                    mask = next(masks)
                self._pin_current_thread(mask)

        self._bucketwise_update_executor = ThreadPoolExecutor(max_workers=max_bucketwise_workers,
                                                              thread_name_prefix="reflow_optimizer",
                                                              initializer=initializer)

        def _submit_bucketwise_task(fn, *args, **kwargs):
            commit = self._last_state_update_future
            if commit is not None and commit.done():
                commit = None
            if initializer is not None and commit is None:
                return self._bucketwise_update_executor.submit(fn, *args, **kwargs)

            mask = None
            if initializer is None:
                worker_idx = self._bucketwise_worker_idx
                masks = self._bucketwise_worker_cpu_affinity_masks
                mask = masks[worker_idx]
                self._bucketwise_worker_idx = (worker_idx + 1) % len(masks)

            def run():
                if mask is not None:
                    self._pin_current_thread(mask)
                if commit is not None and not commit.done():
                    # The entire older commit is already running or queued first. It never waits
                    # on new work from this pool. Disjoint teams can proceed independently.
                    try:
                        cpus = os.sched_getaffinity(0)
                    except (AttributeError, OSError):
                        cpus = None
                    if cpus is None or not self._state_commit_reserved_cpus.isdisjoint(cpus):
                        commit.result()
                return fn(*args, **kwargs)

            return self._bucketwise_update_executor.submit(run)

        self._ready_batches = {}
        self._bucketwise_update_submit = _submit_bucketwise_task
        if max_bucketwise_workers == 1 and len(self._bucketwise_worker_cpu_affinity_masks[0]) > 1 and \
                self._reflow_config.num_threads != 1:
            submit_lock = threading.Lock()
            tail = None

            def submit(fn, *args, **kwargs):
                nonlocal tail
                with submit_lock:
                    if fn == self._bucketwise_chunk_update_worker and self._can_batch_ready(args[0]):
                        item = args[0]
                        if tail is not None and not tail.future.done() and tail.append(item):
                            return tail.future
                        tail = ReadyBatch(self, item)
                        tail.future = _submit_bucketwise_task(tail.run)
                        self._ready_batches[tail.future] = tail
                        return tail.future
                    # Never take work from beyond an accumulation, Persist, or staged Ready.
                    tail = None
                    return _submit_bucketwise_task(fn, *args, **kwargs)

            self._bucketwise_update_submit = submit
        self._bucketwise_inflight_futures = []
        # Backward and step own this registry; executor workers never access it.
        self._bucketwise_accumulation_futures = {}
        # Use the default (lowest) stream priority so H2D weight copies never preempt compute. CUDA priorities
        # are inverted: a negative priority would outrank the compute stream.
        if get_accelerator().is_synchronized_device():
            self._bucketwise_h2d_stream = None
        else:
            self._bucketwise_h2d_stream = get_accelerator().Stream()

        # Run enqueue/merge/flush off the main thread.
        self._bucketwise_enq_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflow_bucketwise_enq")
        self._bucketwise_enq_submit = self._bucketwise_enq_executor.submit

        self._published_ready_groups = set()
        self._pending_checkpoint_groups = set()
        # An early join retains completed ranges here until their replacements finish.
        self._staged_ready_ranges = {}
        self._bucketwise_group_target_step = {}
        self._bucketwise_inflight_h2d_events = []
        self._bucketwise_cancel_version = 0
        # Cap the per-shard size at _DEFAULT_BUCKETWISE_SHARD_ELEMS (also the fallback when
        # reduce_bucket_size is unset) so each subgroup flushes to the CPU-Adam worker partway through
        # backward, overlapping the CPU optimizer, instead of waiting for the whole reduce bucket.
        self._bucketwise_element_shard_size = min(getattr(self, 'reduce_bucket_size', _DEFAULT_BUCKETWISE_SHARD_ELEMS),
                                                  _DEFAULT_BUCKETWISE_SHARD_ELEMS)
        self._bucketwise_enqueue_lock = threading.Lock()
        self._bucketwise_last_copy_event_by_group = {}
        # First real (non-cancel) error raised by a bucketwise worker this iteration; step()
        # all-reduces a flag derived from it and re-raises so the step aborts on every rank.
        self._bucketwise_worker_error = None
        self._bucketwise_error_flag = None  # preallocated device flag for the per-step error all-reduce

    def _reset_bucketwise_handles(self):
        self._bucketwise_update_executor = None
        self._bucketwise_update_submit = None
        self._bucketwise_inflight_futures = None
        self._bucketwise_worker_cpu_affinity_masks = None
        self._worker_node_groups = None
        self._bucketwise_worker_idx = 0
        self._bucketwise_h2d_stream = None
        self._bucketwise_enq_executor = None
        self._bucketwise_enq_submit = None
        self._last_enqueue_future = None

    def destroy(self):
        """
        Drain any background work before shutting the pools down. An undrained bucketwise/state
        worker can otherwise race ReflowCPUAdam.__del__ -> reflow_destroy_adam (use-after-free)
        on an explicit mid-process engine.destroy(). Cancel inflight work first so drains return
        quickly, then wait on the existing drain helpers. This must run before super().destroy(),
        which unpins the offloaded optimizer states the workers are still writing.
        """
        self._bucketwise_cancel_version = getattr(self, '_bucketwise_cancel_version', 0) + 1
        for drain in (self._wait_for_accum_gradient, self._wait_for_pending_state_updates,
                      self._bucketwise_wait_inflight_updates):
            try:
                drain()
            except Exception:
                pass
        for attr in ("_bucketwise_enq_executor", "_bucketwise_update_executor"):
            executor = getattr(self, attr, None)
            if executor is not None:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
        original_forward = getattr(self, "_reflow_original_pre_sub_module_forward", None)
        if original_forward is not None:
            self.parameter_offload.pre_sub_module_forward_function = original_forward
            self._reflow_original_pre_sub_module_forward = None
        # ZeRO clears its bucket mapping before its existing device synchronization. Keep the borrowed
        # storage alive through that synchronization, including copies unassociated with a worker future.
        ipg_buckets = self.ipg_buckets.copy()
        try:
            super().destroy()
            self._ipg_read_events.clear()
        except BaseException:
            # Synchronization may have failed after ZeRO cleared the mapping. Retain ownership for a retry.
            self.ipg_buckets.update(ipg_buckets)
            raise
        finally:
            if self._main_thread_affinity is not None:
                thread_id, cpus = self._main_thread_affinity
                self._main_thread_affinity = None
                os.sched_setaffinity(thread_id, cpus)

    def _unpin_offload_buffers(self):
        # The parent finishes device copies before calling this hook, including D2H for a first micro-step
        # with no accumulation worker and H2D from a cancelled Ready. Native unpin frees the underlying memory.
        for group in self._cpu_grad_double_buffers:
            for buffer in group or ():
                if buffer is not None and buffer.device.type == 'cpu':
                    get_accelerator().unpin_memory(buffer)
        self._cpu_grad_double_buffers = []
        super()._unpin_offload_buffers()

    # ------------------------------------------------------------------ #
    # offload / partition setup
    # ------------------------------------------------------------------ #
    def _configure_offloading(self, offload_optimizer_config, offload_param_config):
        super()._configure_offloading(offload_optimizer_config, offload_param_config)
        assert self.offload_optimizer, "Reflow requires optimizer offload (offload_optimizer device='cpu' or 'nvme')"
        # With NVMe optimizer offload, each subgroup's state is swapped around its CPU step in step() on one
        # thread, because the partitioned optimizer swapper is not thread-safe. Gradients stay in CPU memory.
        self.bucketwise_cores_per_worker = self._reflow_config.bucketwise_cores_per_worker
        self.enable_cpu_affinity = self._reflow_config.enable_cpu_affinity
        self.main_thread_cores = self._reflow_config.main_thread_cores
        self.state_update_cores = self._reflow_config.state_update_cores

    def _create_fp32_partitions(self):
        super()._create_fp32_partitions(create_offloaded_master=self._create_offloaded_master)

    def _create_offloaded_master(self, partition, device):
        dtype = self.master_weights_and_grads_dtype
        if self.offload_optimizer_pin_memory and device == 'cpu':
            master = get_accelerator().pin_empty_like(partition, dtype=dtype)
            master.copy_(partition.detach())
            return master
        return partition.detach().to(device=device, dtype=dtype, copy=True)

    def _setup_for_real_optimizer(self):
        # initialize_optimizer_states already creates the CPU slots used for accumulation and Ready output.
        super()._setup_for_real_optimizer(allocate_grad_partitions=False)

    def _pin_current_thread(self, cpu_affinity_mask):
        if cpu_affinity_mask is None:
            return
        try:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, cpu_affinity_mask)
        except (OSError, AttributeError):
            pass

    def _drain_update_futures_collecting_h2d_events(self, futures, publish_staged=False):
        # A drain for a reentrant gradient or a local edit can leave overlapping old
        # previews. Join their replacements before copying that output slot. Otherwise
        # disjoint shards can start H2D as soon as their own native writer finishes.
        publish_each = publish_staged and not any(self._staged_ready_ranges.values())
        if publish_each and self._staged_ready_ranges and len(futures) > 1:
            futures = as_completed(futures)
        for future in futures:
            try:
                result = future.result()
                if isinstance(result, _StagedReady):
                    if not publish_each:
                        self._staged_ready_ranges[result.sub_group_id].extend(result.ranges)
                    elif self._bucketwise_worker_error is None:
                        self._publish_staged_ready(result.sub_group_id, result.ranges)
                elif result is not None:
                    self._bucketwise_inflight_h2d_events.append(result)
            except CancelledError:
                continue
            except Exception as error:
                self._record_bucketwise_worker_error(error)
            finally:
                self._ready_batches.pop(future, None)
        if publish_staged and not publish_each and self._bucketwise_worker_error is None:
            try:
                for group, ranges in self._staged_ready_ranges.items():
                    if ranges:
                        self._publish_staged_ready(group, reflow_utils.merge_contiguous_ranges(ranges))
            except Exception as error:
                self._record_bucketwise_worker_error(error)

    def _coalesce_bucketwise_h2d_events(self):
        """Return one event covering all pending bucketwise H2D copies."""
        events = [ev for ev in self._bucketwise_inflight_h2d_events if ev is not None]
        self._bucketwise_inflight_h2d_events.clear()
        if len(events) <= 1 or get_accelerator().is_synchronized_device():
            return events

        stream = getattr(self, "_bucketwise_h2d_stream", None)
        if stream is None:
            return events

        with get_accelerator().stream(stream):
            barrier = get_accelerator().Event()
            barrier.record(stream)
        return [barrier]

    def _record_bucketwise_worker_error(self, exc):
        # Record the first real (non-cancel) bucketwise worker failure under the shared enqueue
        # lock; step() reads it to abort deterministically. Keep only the first so the original
        # cause survives later cascading failures.
        with self._bucketwise_enqueue_lock:
            if self._bucketwise_worker_error is None:
                self._bucketwise_worker_error = exc

    def _check_bucketwise_worker_error(self):
        # Abort the step deterministically on every rank if any rank's bucketwise worker hit a real
        # error. A kernel/OOM failure may occur on only some ranks, so all-reduce a flag (MAX) and
        # raise everywhere; a rank that saw the actual exception re-raises it, the rest raise a
        # generic RuntimeError. This is a no-op (flag stays 0) on the happy path.
        local_error = self._bucketwise_worker_error
        # With one rank there is no remote worker error to collect, so avoid a device sync.
        if dist.get_world_size() == 1:
            if local_error is not None:
                raise local_error
            return
        # Reuse a preallocated 1-element flag instead of allocating a new device tensor every step
        # on this foreground collective path.
        error_flag = self._bucketwise_error_flag
        if error_flag is None:
            error_flag = get_accelerator().ByteTensor([0])
            self._bucketwise_error_flag = error_flag
        error_flag.fill_(1 if local_error is not None else 0)
        dist.all_reduce(error_flag, op=dist.ReduceOp.MAX, group=self.dp_process_group)
        self._model_parallel_all_reduce(tensor=error_flag, op=dist.ReduceOp.MAX)
        if error_flag[0].item() == 0:
            return
        if local_error is not None:
            raise local_error
        raise RuntimeError("Reflow bucketwise worker failed on another rank; aborting step on all ranks")

    # ------------------------------------------------------------------ #
    # optimizer state init (double-buffered CPU grad buffers)
    # ------------------------------------------------------------------ #
    def _resolve_single_grad_buffer_eligibility(self):
        """When REFLOW_SINGLE_GRAD_BUFFER=1 is set, keep the single-slot optimization only on runs it can
        serve correctly; otherwise fall back to the default double-buffer path (disable it) with a one-time
        warning instead of failing. The single slot folds the FP16 param output into the CPU param source
        (fp16_partitioned_groups_flat), so it is valid only with: CPU-offloaded params (fp16_flat is a
        real CPU tensor the CPU kernel writes), no NVMe-swapped subgroup (those keep the 2-slot path), and
        gradient_accumulation_steps == 1 (TODO: the accumulation path still needs the second slot as a
        per-chunk scratch)."""
        reason = None
        gradient_accumulation_steps = self.gradient_accumulation_steps
        if callable(gradient_accumulation_steps):
            gradient_accumulation_steps = gradient_accumulation_steps()
        if gradient_accumulation_steps != 1:
            reason = f"gradient_accumulation_steps={gradient_accumulation_steps} != 1 (TODO: not supported yet)"
        elif not self.offload_param:
            reason = "offload_param is not enabled (the FP16 param source is on GPU, not a CPU flat buffer)"
        else:
            for sub_group_id in range(len(self.fp16_groups)):
                fp16_flat = self.fp16_partitioned_groups_flat[sub_group_id]
                if self._swappable_optimizer_subgroup(sub_group_id):
                    reason = f"subgroup {sub_group_id} uses NVMe optimizer offload"
                elif fp16_flat is None:
                    reason = f"subgroup {sub_group_id} has no CPU param flat buffer (NVMe-swapped params)"
                elif fp16_flat.device.type != 'cpu':
                    reason = f"subgroup {sub_group_id} param flat buffer is on {fp16_flat.device}, not CPU"
                elif self.subgroup_to_device[sub_group_id] != 'cpu':
                    reason = (f"subgroup {sub_group_id} grad buffer maps to "
                              f"{self.subgroup_to_device[sub_group_id]}, not CPU")
                if reason is not None:
                    break
        if reason is not None:
            if dist.get_rank() == 0:
                logger.warning(f"REFLOW_SINGLE_GRAD_BUFFER=1 ignored ({reason}); using the default "
                               f"double-buffer path.")
            self._single_grad_buffer = False

    def initialize_optimizer_states(self):
        timer_names = set()

        is_adagrad = isinstance(self.optimizer, torch.optim.Adagrad)

        if self.swap_optimizer:
            self.optimizer_swapper.init_timers()

        timer_names.add(INIT_OPTIMIZER_TIMER)
        self.timers(INIT_OPTIMIZER_TIMER).start()

        self._cpu_grad_double_buffers = []
        self._pending_grad_ready = False

        if self._single_grad_buffer:
            self._resolve_single_grad_buffer_eligibility()

        for i, group in enumerate(self.fp16_groups):
            swappable_optimizer_subgroup = self._swappable_optimizer_subgroup(i)
            swappable_param_subgroup = self.fp16_partitioned_groups_flat[i] is None

            num_elements = int(self.fp16_partitioned_groups_flat_numel[i])

            if swappable_optimizer_subgroup:
                self._optimizer_states_and_gradient_swap_in(i, timer_names)
                # Bootstrap the lazy exp_avg/exp_avg_sq (zeros) NOW, before the init swap-out below
                # registers this subgroup's state in the swap pool. Otherwise the first bucketwise
                # flush creates them outside the pool, the swap-out cannot capture them, and the next
                # swap-in reads uninitialized state -> the CPU-Adam produces NaN weights.
                self.optimizer._ensure_param_state(self.fp32_partitioned_groups_flat[i])
                if self.use_muon and self.sub_groups_using_muon[i] and not self.save_muon_momentum_buffer_in_memory:
                    if "momentum_buffer" not in self.optimizer.state.get(self.fp32_partitioned_groups_flat[i], {}):
                        self._create_momentum_buffer(num_elements, i, self.fp32_partitioned_groups_flat[i].ds_id)
            # Store grads in half precision for the bucketwise CPU-Adam (the kernel promotes to FP32).
            # NVMe-optimizer subgroups use the same CPU half buffer for the reduced grad -- only the
            # optimizer STATE is swapped to disk, never the gradient, so they need this buffer too.
            grad_dtype = self.dtype
            grad_buffers = []
            # Single-buffer mode allocates one slot (kept as a length-1 list so [0] indexing works);
            # the param output is written into fp16_flat instead of a second slot.
            num_grad_slots = 1 if self._single_grad_buffer else 2
            for _ in range(num_grad_slots):
                device = self.subgroup_to_device[i]
                if self.offload_optimizer_pin_memory and device == 'cpu':
                    subgroup_gradient_buffer = get_accelerator().pin_empty(num_elements, dtype=grad_dtype).zero_()
                else:
                    subgroup_gradient_buffer = torch.zeros(num_elements, dtype=grad_dtype, device=device)
                grad_buffers.append(subgroup_gradient_buffer)
            self._cpu_grad_double_buffers.append(grad_buffers)
            # The half-precision grad dtype differs from the FP32 param, so leave .grad unset;
            # the kernels read the grad double-buffer directly.
            self.fp32_partitioned_groups_flat[i].grad = None

            if swappable_param_subgroup:
                self._partitioned_params_swap_out(i)

            if swappable_optimizer_subgroup:
                self._optimizer_states_and_gradient_swap_out(i, timer_names)

        if is_adagrad:
            self.optimizer = torch.optim.Adagrad(self.fp32_partitioned_groups_flat, **self.optimizer.defaults)

        self.timers(INIT_OPTIMIZER_TIMER).stop()
        self.timers.log(timer_names)

        if self.swap_optimizer:
            self.optimizer_swapper.log_timers()

        return

    # ------------------------------------------------------------------ #
    # gradient partitioning
    # ------------------------------------------------------------------ #
    def _add_grad_to_ipg_bucket(self, param):
        # Only a write into this bucket can overwrite a still-pending copy. Oversized gradients bypass it.
        if self._ipg_read_events:
            dtype = self.get_param_comm_dtype(param)
            event = self._ipg_read_events.get(dtype)
            if event is not None:
                bucket = self.ipg_buckets[dtype]
                if bucket.elements + param.grad.numel() <= self._ipg_bucket_capacity(bucket):
                    self.reduce_and_partition_stream.wait_event(event)
                    del self._ipg_read_events[dtype]
        super()._add_grad_to_ipg_bucket(param)

    @staticmethod
    def _unused_ipg_buffer(bucket, dtype):
        buffer = bucket.buffer
        if buffer is None:
            return None
        # Keep scratch outside all live gradient views and aligned for vectorized reductions.
        start = (bucket.elements * buffer.element_size() + 255) // 256 * 256
        element_size = torch.finfo(dtype).bits // 8
        end = buffer.numel() * buffer.element_size() // element_size * element_size
        if start >= end:
            return None
        return buffer.view(torch.uint8)[start:end].view(dtype)

    @instrument_w_nvtx
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        buffers = []

        # Cache loop-invariant values (avoid repeated getattr/property/C++ binding calls).
        is_boundary = self.is_gradient_accumulation_boundary()
        copy_grad_stream = self.copy_grad_stream
        current_stream = get_accelerator().current_stream()
        stream_needs_sync = copy_grad_stream != current_stream
        is_sync_device = get_accelerator().is_synchronized_device()
        accumulate_submit = getattr(self, "_bucketwise_update_submit", None)
        if not params_to_release:
            return buffers
        comm_dtype = self.get_param_comm_dtype(params_to_release[0])
        bucket = self.ipg_buckets[comm_dtype]
        norm_staging = self._unused_ipg_buffer(bucket, get_norm_dtype()) if is_boundary else None
        accum_staging = None
        ipg_storage = None
        if stream_needs_sync and bucket.buffer is not None:
            ipg_storage = bucket.buffer.untyped_storage().data_ptr()
        wait_for_reduction = stream_needs_sync
        checkpoint_grads = bool(self._reflow_checkpoint_visits)
        check_source = stream_needs_sync and (ipg_storage is not None or checkpoint_grads)
        # This output slot is separate from the GPU weights read by checkpoint recomputation.
        stage_checkpoints = self.clip_grad <= 0. and not self.dynamic_loss_scale and not self.offload_param

        for param, grad_partition in zip(params_to_release, grad_partitions):
            copy_event = None

            contains_real_data = param.partition_numel() * self._get_param_partition_rank(param) < param.ds_numel
            if not contains_real_data:
                param.grad = None
                continue

            # No flat grad buffer is allocated in bucketwise mode; use grad_partition directly.
            grad_buffer = grad_partition
            buffers.append(grad_buffer)

            param_id = self.get_param_id(param)
            grad_position_entry = self.grad_position[param_id]
            sub_group_id, dest_offset, _ = grad_position_entry
            checkpoint_step, shared = self._reflow_checkpoint_visits.get(param_id, (None, False))
            checkpointed = checkpoint_step is not None
            # A shared checkpoint weight can receive multiple reductions in this micro-step.
            # Leave its Ready until step() so it only runs on the complete gradient.
            enqueue_ready = (is_boundary and (not checkpointed or (stage_checkpoints and not shared))
                             and not self._swappable_optimizer_subgroup(sub_group_id))
            stage_checkpoint = enqueue_ready and checkpointed
            boundary_grad = is_boundary and (not checkpointed or stage_checkpoint)
            first_grad = param_id not in self._reflow_window_grad_ids
            self._reflow_window_grad_ids.add(param_id)
            if is_boundary and param_id in self._reflow_boundary_grad_ids:
                # A reentrant backward can submit a second gradient after Ready has
                # started reading this range. Finish that use before replacing it.
                self._bucketwise_drain_enq_executor()
                clipped_range = self._bucketwise_clip_range_to_valid_partition(sub_group_id, dest_offset,
                                                                               grad_partition.numel())
                pending_ranges = self._bucketwise_pending_ranges_by_group[sub_group_id]
                if clipped_range in pending_ranges:
                    pending_ranges.remove(clipped_range)
                    self._bucketwise_pending_elems_by_group[sub_group_id] -= clipped_range[1]
                else:
                    self._bucketwise_wait_inflight_updates()
                    for event in self._bucketwise_inflight_h2d_events:
                        if event is not None:
                            event.synchronize()

            if grad_partition.dtype != self.gradient_accumulation_dtype:
                converted_grad = grad_partition.to(self.gradient_accumulation_dtype)
            else:
                converted_grad = grad_partition
            reads_ipg = False
            if check_source and converted_grad is grad_partition:
                storage = converted_grad.untyped_storage().data_ptr()
                reads_ipg = storage == ipg_storage
                if (checkpoint_grads and not reads_ipg and param.grad is not None
                        and storage == param.grad.untyped_storage().data_ptr()):
                    # Autograd can reuse an unbucketed checkpoint gradient during recomputation.
                    converted_grad = converted_grad.clone()
            # One handoff covers the reduced bucket. Only a new producer-side cast or clone needs another.
            if wait_for_reduction or (stream_needs_sync and converted_grad is not grad_partition):
                copy_grad_stream.wait_stream(current_stream)
                wait_for_reduction = False
            if self.dtype == torch.float16:
                self.inf_or_nan_tracker.logical_or_(has_inf_or_nan(converted_grad))

            grad_numel = grad_buffer.numel()

            if not first_grad:
                # Both slices must remain unchanged until the previous worker consumes them.
                self._wait_for_accum_gradient(param_id)

            if first_grad:
                if self._single_grad_buffer:
                    # Slot 0 is also what the previous step's state commit reads, so wait for that commit before
                    # overwriting it. Usually a no-op: the commit finishes during the forward.
                    self._wait_for_pending_state_updates(sub_group_id=sub_group_id)
                pending_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=True)
                pending_slice = pending_buffer.narrow(0, dest_offset, grad_numel)

                with get_accelerator().stream(copy_grad_stream):
                    pending_slice.copy_(converted_grad.view(pending_slice.shape), non_blocking=True)
                    if enqueue_ready or reads_ipg:
                        copy_event = get_accelerator().Event()
                        copy_event.record(copy_grad_stream)
                        if reads_ipg:
                            self._ipg_read_events[comm_dtype] = copy_event
                # No gradient accumulation: this micro-step is also the boundary, so use the GPU
                # grad tensor directly for the norm/enqueue below.
                if is_boundary:
                    grad_buffer = converted_grad
                    grad_numel = grad_buffer.numel()
            else:
                if is_boundary:
                    pending_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=True)
                    dst_slice = pending_buffer.narrow(0, dest_offset, grad_numel)
                    if accum_staging is None and norm_staging is not None:
                        # Norms on the reduction stream reuse the prefix. Give each simultaneous copy-stream
                        # accumulation its own slice beyond the largest norm, so neither stream waits for reuse.
                        largest_partition = max(part.numel() for part in grad_partitions)
                        accum_staging = norm_staging[largest_partition:].view(converted_grad.dtype)
                        norm_staging = norm_staging[:largest_partition]
                    use_bucket_staging = accum_staging is not None and grad_numel <= accum_staging.numel()
                    # The norm only needs the sum. The next bucket write must also wait for its D2H copy.
                    staging_read_needs_event = use_bucket_staging and stream_needs_sync
                    with get_accelerator().stream(copy_grad_stream):
                        if use_bucket_staging:
                            accum_gpu = accum_staging[:grad_numel]
                            accum_staging = accum_staging[grad_numel:]
                            accum_gpu.copy_(dst_slice, non_blocking=True)
                        else:
                            accum_gpu = dst_slice.to(converted_grad.device, non_blocking=True)
                        accum_gpu.add_(converted_grad.view(accum_gpu.shape))
                        if stream_needs_sync:
                            accum_event = get_accelerator().Event()
                            accum_event.record(copy_grad_stream)
                        dst_slice.copy_(accum_gpu, non_blocking=True)
                        if enqueue_ready or staging_read_needs_event:
                            copy_event = get_accelerator().Event()
                            copy_event.record(copy_grad_stream)
                            if staging_read_needs_event:
                                self._ipg_read_events[comm_dtype] = copy_event
                    if stream_needs_sync:
                        current_stream.wait_event(accum_event)
                        # A temporary allocated on the copy stream must also stay live through the norm read.
                        accum_gpu.record_stream(current_stream)
                    grad_buffer = accum_gpu
                    grad_numel = grad_buffer.numel()
                else:
                    # The active slot written below as accumulation scratch still holds the previous step's grads,
                    # which that step's async state commit reads. Wait for that commit first; it is usually done
                    # by now, and the wait is a no-op once the future has been collected.
                    self._wait_for_pending_state_updates(sub_group_id=sub_group_id)
                    # CPU-side BF16 accumulation using AVX (avoid CPU->GPU->CPU round-trip).
                    active_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=False)
                    src_slice = active_buffer.narrow(0, dest_offset, grad_numel)
                    pending_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=True)
                    dst_slice = pending_buffer.narrow(0, dest_offset, grad_numel)
                    with get_accelerator().stream(copy_grad_stream):
                        src_slice.copy_(converted_grad.view(src_slice.shape), non_blocking=True)
                    copy_event = get_accelerator().Event()
                    copy_event.record(copy_grad_stream)
                    if reads_ipg:
                        self._ipg_read_events[comm_dtype] = copy_event
                    if accumulate_submit is not None:
                        # Capture values eagerly via default args (avoid late-binding closure bug).
                        def _accumulate_worker(_copy_event=copy_event, _dst_slice=dst_slice, _src_slice=src_slice):
                            _copy_event.synchronize()
                            ReflowCPUAdam.bf16_accumulate(_dst_slice.view(-1), _src_slice.view(-1))

                        future = accumulate_submit(_accumulate_worker)
                        self._bucketwise_accumulation_futures[param_id] = future
                        grad_buffer = dst_slice
                    else:
                        copy_event.synchronize()
                        ReflowCPUAdam.bf16_accumulate(dst_slice.view(-1), src_slice.view(-1))
                        grad_buffer = dst_slice
                    grad_numel = grad_buffer.numel()

            if is_boundary:
                self._pending_grad_ready = True
                if boundary_grad:
                    self._reflow_boundary_grad_ids.add(param_id)
                if enqueue_ready:
                    self._bucketwise_schedule_enqueue_range(sub_group_id, dest_offset, grad_numel, copy_event,
                                                            stage_checkpoint)
                if norm_staging is not None and grad_numel <= norm_staging.numel():
                    # The copy and the norm both run on the bucket's producer stream before its next write.
                    norm_input = norm_staging[:grad_numel]
                    norm_input.copy_(grad_buffer.view(-1))
                    grad_buffer = norm_input
                self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_buffer)

            if not is_sync_device:
                if param.grad is not None:
                    param.grad.record_stream(current_stream)
                # The copy runs later on copy_grad_stream; record the source against it so the caching allocator does
                # not hand that memory to the next bucket before the copy reads it.
                if stream_needs_sync:
                    converted_grad.record_stream(copy_grad_stream)
            param.grad = None

        return buffers

    def reset_cpu_buffers(self):
        super().reset_cpu_buffers()
        self._reflow_window_grad_ids.clear()
        self._reflow_boundary_grad_ids.clear()
        self._reflow_checkpoint_visits.clear()
        self._reflow_modified_gradient_ids.clear()
        self._pending_grad_ready = False

        # Reset per-iteration pending/inflight state.
        self._published_ready_groups.clear()
        self._pending_checkpoint_groups.clear()
        self._staged_ready_ranges.clear()
        self._bucketwise_group_target_step.clear()
        for sub_group_id in range(len(self._bucketwise_pending_ranges_by_group)):
            self._bucketwise_pending_ranges_by_group[sub_group_id].clear()
            self._bucketwise_pending_elems_by_group[sub_group_id] = 0
        self._bucketwise_inflight_h2d_events.clear()
        self._bucketwise_last_copy_event_by_group.clear()
        if self._bucketwise_inflight_futures is not None:
            self._bucketwise_inflight_futures.clear()
        self._bucketwise_cancel_version = 0
        self._bucketwise_worker_error = None

    def zero_grad(self, set_to_none=True):
        # An explicit zero_grad may discard a window after Ready has published a
        # preview. Drain only on this uncommon path, then restore those weights.
        window_ids = getattr(self, "_reflow_window_grad_ids", None)
        if window_ids and not getattr(self, "_reflow_in_overflow_cleanup", False):
            self._wait_for_accum_gradient()
            self._wait_for_pending_state_updates()
            self._bucketwise_drain_enq_executor()
            self._bucketwise_cancel_inflight_updates()
            self._bucketwise_wait_inflight_updates()
            for event in self._bucketwise_inflight_h2d_events:
                if event is not None:
                    event.synchronize()
            self.copy_grad_stream.synchronize()
            for sub_group_id in self._published_ready_groups:
                self._restore_fp16_params_from_master(sub_group_id)
            if self.dtype == torch.float16:
                with get_accelerator().stream(self.reduce_and_partition_stream):
                    self.inf_or_nan_tracker.zero_()
            checkpoint_visits = self._reflow_checkpoint_visits.copy()
            worker_error = self._bucketwise_worker_error
            self.reset_cpu_buffers()
            self._bucketwise_worker_error = worker_error
            self._reflow_checkpoint_visits.update(checkpoint_visits)
        return super().zero_grad(set_to_none=set_to_none)

    def _overflow_clean_up(self, prev_scale):
        # A staged CPU result has not changed the GPU weight. Restore resident groups
        # that could have published a preview, as well as NVMe parameter partitions.
        restore_groups = [
            i for i, flat in enumerate(self.fp16_partitioned_groups_flat)
            if flat is None or i in self._published_ready_groups
        ]
        self._reflow_in_overflow_cleanup = True
        try:
            super()._overflow_clean_up(prev_scale)
        finally:
            self._reflow_in_overflow_cleanup = False
        if restore_groups:
            if self._bucketwise_h2d_stream is not None:
                self._bucketwise_h2d_stream.synchronize()
            for sub_group_id in restore_groups:
                self._restore_fp16_params_from_master(sub_group_id)

    # ------------------------------------------------------------------ #
    # async state update workers
    # ------------------------------------------------------------------ #
    def _submit_state_updates(self, sub_group_ids, combined_scale):
        # Capture the gradient slot and LR scheduler values before step() returns. One pool
        # job commits subgroups in order because they temporarily share group['params'].
        # Per-subgroup completion still releases readers as soon as their master is ready.
        commits = []
        for sub_group_id in sub_group_ids:
            group_id = self.sub_group_to_group_id[sub_group_id]
            group = self.optimizer.param_groups[group_id]
            hyperparams = {key: value for key, value in group.items() if key != 'params'}
            grad = self._cpu_grad_double_buffers[sub_group_id][self._active_grad_buffer_idx]
            commits.append((sub_group_id, group_id, {group_id: hyperparams}, grad))
        if not commits:
            return

        # The executor's own Future covers the last (often only) subgroup.
        earlier = {entry[0]: Future() for entry in commits[:-1]}

        def commit_all():
            original_cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
            try:
                for sub_group_id, group_id, hyperparams, grad in commits:
                    self._commit_subgroup_state(sub_group_id, group_id, hyperparams, grad, combined_scale)
                    if sub_group_id in earlier:
                        earlier[sub_group_id].set_result(None)
            finally:
                self._pin_current_thread(original_cpus)

        try:
            tail = self._bucketwise_update_executor.submit(commit_all)
        except Exception as error:
            tail = Future()
            tail.set_exception(error)

        if earlier:

            def release_readers_on_failure(done):
                try:
                    done.result()
                except BaseException as error:
                    # Also release readers if the executor failed before commit_all could start.
                    for completed in earlier.values():
                        if not completed.done():
                            completed.set_exception(error)

            tail.add_done_callback(release_readers_on_failure)
            self._pending_state_updates_by_subgroup.update(earlier)
        self._pending_state_updates_by_subgroup[commits[-1][0]] = tail
        self._last_state_update_future = tail

    def _commit_subgroup_state(self, sub_group_id, group_id, hyperparams, grad, combined_scale):
        group = self.optimizer.param_groups[group_id]
        original_params = group['params']
        param = self.fp32_partitioned_groups_flat[sub_group_id]
        group['params'] = [param]
        try:
            buffers = [None] * len(self.optimizer.param_groups)
            buffers[group_id] = grad
            numel = param.numel()
            for start in range(0, numel, _STATE_COMMIT_SLICE_ELEMS):
                self._pin_current_thread(self._state_commit_cpu_affinity_mask())
                slice_numel = min(_STATE_COMMIT_SLICE_ELEMS, numel - start)
                self.optimizer.step_state_halfgrad(half_grad_buffers=buffers,
                                                   combined_scale=combined_scale,
                                                   element_range=(start, slice_numel),
                                                   group_hyperparams=hyperparams)
            self.optimizer.memory_fence()
        finally:
            group['params'] = original_params

    def _record_state_update_error(self, error):
        # Surface a state-update worker failure through the same all-reduced flag the bucketwise
        # workers use, so a failure on only some ranks aborts the step on ALL ranks together (the next
        # _check_bucketwise_worker_error all-reduces it) instead of this rank raising alone and hanging
        # its peers at the following collective.
        if self._bucketwise_worker_error is None:
            self._bucketwise_worker_error = error

    def _wait_for_pending_state_updates(self, sub_group_id=None):
        """Wait for pending state-update workers (one subgroup, or all if sub_group_id is None)."""
        if sub_group_id is not None:
            future = self._pending_state_updates_by_subgroup.get(sub_group_id, None)
            if future is None:
                return
            try:
                future.result()
            except Exception as e:
                if dist.get_rank() == 0:
                    logger.warning(f"Error waiting for state update worker (subgroup {sub_group_id}): {e}")
                self._record_state_update_error(e)
            finally:
                self._pending_state_updates_by_subgroup.pop(sub_group_id, None)
        else:
            future = getattr(self, '_last_state_update_future', None)
            if future is None:
                return
            try:
                # One pool job owns every resident subgroup in this step.
                future.result()
            except Exception as e:
                if dist.get_rank() == 0:
                    logger.warning(f"Error waiting for state update workers: {e}")
                self._record_state_update_error(e)
            finally:
                self._last_state_update_future = None

    def _wait_for_accum_gradient(self, param_id=None):
        """Wait until the requested gradient buffers are no longer used by accumulation workers."""
        if param_id is None:
            futures = list(self._bucketwise_accumulation_futures.values())
            self._bucketwise_accumulation_futures.clear()
        else:
            future = self._bucketwise_accumulation_futures.pop(param_id, None)
            if future is None:
                return
            futures = (future, )
        for future in futures:
            try:
                future.result()
            except Exception as e:
                self._record_bucketwise_worker_error(e)

    def _get_cpu_grad_buffer(self, sub_group_id, pending=False):
        if sub_group_id >= len(self._cpu_grad_double_buffers):
            return self.fp32_partitioned_groups_flat[sub_group_id].grad
        buffers = self._cpu_grad_double_buffers[sub_group_id]
        if buffers is None:
            return self.fp32_partitioned_groups_flat[sub_group_id].grad
        index = self._pending_grad_buffer_idx if pending else self._active_grad_buffer_idx
        return buffers[index]

    def _complete_pending_gradient_window(self):
        # The parent ZeRO-3 fills these partitions with zeros or copies their earlier
        # micro-step gradients at the boundary. ReFlow owns the CPU buffers instead.
        if self._reflow_gradient_partitions is None:
            self._reflow_gradient_partitions = []
            for sub_group_id, group in enumerate(self.fp16_groups):
                for param in group:
                    if not param.requires_grad:
                        continue
                    if param.partition_numel() * self._get_param_partition_rank(param) >= param.ds_numel:
                        continue
                    param_id = self.get_param_id(param)
                    position = self.grad_position.get(param_id)
                    if position is not None and position[2] > 0:
                        self._reflow_gradient_partitions.append((param_id, sub_group_id, position[1], position[2]))
        if len(self._reflow_boundary_grad_ids) == len(self._reflow_gradient_partitions):
            return False, self._reflow_modified_gradient_ids
        ready_after_norm = self._reflow_modified_gradient_ids.copy()
        enqueued_ready = False
        for param_id, sub_group_id, offset, numel in self._reflow_gradient_partitions:
            if param_id in self._reflow_boundary_grad_ids:
                continue
            grad_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=True).narrow(0, offset, numel)
            has_boundary_norm = (param_id in self.norm_for_param_grads
                                 and param_id not in self._reflow_modified_gradient_ids)
            if param_id not in self._reflow_window_grad_ids:
                if self._single_grad_buffer:
                    self._wait_for_pending_state_updates(sub_group_id=sub_group_id)
                grad_buffer.zero_()
                self.norm_for_param_grads[param_id] = 0.0
            elif not has_boundary_norm:
                # A local gradient setter can leave a norm from an earlier micro-step.
                self._wait_for_accum_gradient(param_id)
                grad_for_norm = grad_buffer.to(get_accelerator().current_device_name(), non_blocking=True)
                self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_for_norm)
            self._pending_grad_ready = True
            if not self._swappable_optimizer_subgroup(sub_group_id):
                # Checkpointed gradients with a final norm have no backward or norm computation left
                # to overlap here. Use the final scale for their first Ready if clipping is enabled.
                if has_boundary_norm and self.clip_grad > 0.:
                    ready_after_norm.add(param_id)
                elif param_id not in ready_after_norm:
                    self._bucketwise_schedule_enqueue_range(sub_group_id, offset, numel)
                    enqueued_ready = True
        return enqueued_ready, ready_after_norm

    def _activate_pending_cpu_grads_if_ready(self, flush_pending):
        # The enqueue thread chooses the gradient slot at submission. Once it drains, every
        # running Ready owns its slot and can finish while the host decides clipping/overflow.
        self._bucketwise_drain_enq_executor()
        if self._pending_grad_ready:
            if flush_pending:
                self._bucketwise_flush_all_pending(use_pending_buffer=True, combined_scale=self.loss_scale)
            if not self._single_grad_buffer:
                self._active_grad_buffer_idx = self._pending_grad_buffer_idx
            self._pending_grad_ready = False

    # ------------------------------------------------------------------ #
    # cpu_bucketwise pipeline
    # ------------------------------------------------------------------ #
    def _bucketwise_drain_enq_executor(self):
        """Block until all async enqueue work has been reflected in the pending ranges.

        ``_bucketwise_schedule_enqueue_range`` submits enqueue work asynchronously; before
        flushing we must ensure every gradient range has landed in the pending structures,
        otherwise a range is dropped and the parameter update is silently skipped.
        """
        # A single FIFO worker makes the last real job a fence for all earlier enqueues.
        future = self._last_enqueue_future
        if future is not None:
            try:
                future.result(timeout=30.0)
            except Exception as e:
                self._record_bucketwise_worker_error(e)
            else:
                self._last_enqueue_future = None

    def _bucketwise_schedule_enqueue_range(self,
                                           sub_group_id,
                                           dest_offset,
                                           num_elements,
                                           copy_event=None,
                                           stage_checkpoint=False):
        """Schedule a range enqueue off the main thread (falls back to sync if no executor)."""
        if not self.is_gradient_accumulation_boundary():
            return

        enq_submit = getattr(self, "_bucketwise_enq_submit", None)
        if enq_submit is None:
            self._bucketwise_enqueue_range(sub_group_id, dest_offset, num_elements, copy_event, stage_checkpoint)
            return
        try:
            self._last_enqueue_future = enq_submit(self._bucketwise_enqueue_range, sub_group_id, dest_offset,
                                                   num_elements, copy_event, stage_checkpoint)
        except Exception:
            self._bucketwise_drain_enq_executor()
            self._bucketwise_enqueue_range(sub_group_id, dest_offset, num_elements, copy_event, stage_checkpoint)

    def _bucketwise_get_valid_partition_element_count(self, sub_group_id: int) -> int:
        """Return the flat-buffer bound, including padding between parameters."""
        # Per-parameter padding is interspersed in the flat subgroup, so subtracting
        # its sum as if it were all at the end drops real elements near the end.
        return int(self.fp16_partitioned_groups_flat_numel[sub_group_id])

    def _bucketwise_clip_range_to_valid_partition(self, sub_group_id: int, dest_offset: int, num_elements: int):
        """Clamp a range to the valid partition; return None if it is entirely padding."""
        valid_partition_element_count = self._bucketwise_get_valid_partition_element_count(sub_group_id)
        return reflow_utils.clip_range_to_valid_partition(valid_partition_element_count, dest_offset, num_elements)

    def _can_batch_ready(self, item):
        if item["stage_copy"] or self.swap_optimizer or not self.offload_optimizer_pin_memory:
            return False
        dst = self.fp16_partitioned_groups_flat[item["sub_group_id"]]
        if dst is None or dst.device.type != "cuda" or self._bucketwise_h2d_stream is None:
            return False
        chunk_elems = (self._reflow_config.h2d_chunk_size_mib << 20) // dst.element_size()
        return any(size >= chunk_elems for _, size in item["ranges"])

    def _bucketwise_chunk_update_worker(self, work_item, defer=False):
        """Generate low-precision weights for these ranges. Persist owns the master and moment updates."""
        cancel_version = work_item["cancel_version"]
        direct_cpu_output = False
        try:
            sub_group_id = work_item["sub_group_id"]
            ranges = work_item["ranges"]
            grad_buffer_index = work_item["grad_buffer_index"]
            combined_scale = work_item["combined_scale"]
            last_copy_event = work_item["last_copy_event"]
            target_step = work_item["target_step"]

            if cancel_version != self._bucketwise_cancel_version:
                return None

            # Wait for the GPU->CPU grad copy (only the last event, the stream is sequential).
            if last_copy_event is not None:
                last_copy_event.synchronize()

            if cancel_version != self._bucketwise_cancel_version:
                return None

            # Boundary ranges are enqueued after their accumulation completes; step() drains the rest before reruns.
            # Make sure the previous state update for this subgroup finished.
            self._wait_for_pending_state_updates(sub_group_id=sub_group_id)

            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            # grad_buffer_index holds the grads (slot 0 in single-buffer mode).
            half_grad_buffer = self._cpu_grad_double_buffers[sub_group_id][grad_buffer_index]
            fp16_flat = self.fp16_partitioned_groups_flat[sub_group_id]
            # Keep staged previews private until step() publishes them. NVMe can still need
            # the opposite slot as its swap source. Resident CPU params already own the output.
            direct_cpu_output = (fp16_flat is not None and fp16_flat.device.type == "cpu"
                                 and not work_item["stage_copy"] and not self.swap_optimizer
                                 and not self.params_in_nvme_and_cpu)
            if self._single_grad_buffer or direct_cpu_output:
                half_params_buffer = fp16_flat
            else:
                half_params_buffer = self._cpu_grad_double_buffers[sub_group_id][1 - grad_buffer_index]

            use_lion = hasattr(self.optimizer, "ds_opt_lion")
            if not use_lion and not hasattr(self.optimizer, "ds_opt_adam"):
                raise RuntimeError("cpu_bucketwise requires a Reflow CPU optimizer (ReflowCPUAdam or ReflowCPULion)")

            state = self.optimizer.state[fp32_param]

            lr = work_item["lr"]
            beta1, beta2 = work_item["betas"]
            weight_decay = work_item["weight_decay"]

            master = fp32_param.detach().view(-1)
            grads = half_grad_buffer.view(-1)
            momentum = state['exp_avg'].view(-1)
            variance = None if use_lion else state['exp_avg_sq'].view(-1)
            output = half_params_buffer.view(-1)

            native = self.optimizer.ds_opt_lion if use_lion else self.optimizer.ds_opt_adam
            work = None
            if defer:
                chunk_elems = (self._reflow_config.h2d_chunk_size_mib << 20) // fp16_flat.element_size()
                work = native.ReadyWork(fp16_flat, self._bucketwise_h2d_stream, chunk_elems)

            def prepare_range(offset, size, output_slice):
                if cancel_version != self._bucketwise_cancel_version:
                    return False
                param_slice = master.narrow(0, offset, size)
                grad_slice = grads.narrow(0, offset, size)
                momentum_slice = momentum.narrow(0, offset, size)
                deferred = (work, offset) if defer else ()
                # Ready kernels use the absolute target_step for every range.
                if use_lion:
                    native.reflow_lion_update_params_halfgrad(self.optimizer.opt_id, target_step, lr, beta1, beta2,
                                                              weight_decay, param_slice, grad_slice, momentum_slice,
                                                              output_slice, combined_scale, False, *deferred)
                else:
                    variance_slice = variance.narrow(0, offset, size)
                    native.reflow_adam_update_params_halfgrad(self.optimizer.opt_id, target_step, lr, beta1, beta2,
                                                              work_item["eps"], weight_decay,
                                                              work_item["bias_correction"], param_slice, grad_slice,
                                                              momentum_slice, variance_slice, output_slice,
                                                              combined_scale, False, work_item["maximize"], *deferred)
                return cancel_version == self._bucketwise_cancel_version

            if not defer and fp16_flat is not None and fp16_flat.device.type != "cpu" and not work_item["stage_copy"]:
                return self._copy_ready_ranges(fp16_flat, output, ranges, prepare=prepare_range)

            for offset, size in ranges:
                if not prepare_range(offset, size, output.narrow(0, offset, size)):
                    return None
            if defer:
                return work
            if fp16_flat is None or half_params_buffer is fp16_flat:
                return None
            if work_item["stage_copy"]:
                return _StagedReady(sub_group_id, ranges)
            return self._copy_ready_ranges(fp16_flat, output, ranges)
        except Exception as e:
            if direct_cpu_output:
                # A failed kernel may already have written part of the public CPU partition.
                # This work owns these disjoint ranges, and Persist has not committed this step.
                source = fp32_param.detach().view(-1)
                destination = fp16_flat.detach().view(-1)
                for offset, size in ranges:
                    destination.narrow(0, offset, size).copy_(source.narrow(0, offset, size))
            # A cooperative cancel (version bump) can race the kernel and surface as a transient
            # error; if this work item has been cancelled, treat it as a clean no-op like the
            # explicit cancel returns above. Otherwise this is a real kernel/OOM failure: record it
            # so step() can abort deterministically on all ranks instead of silently advancing.
            if cancel_version != self._bucketwise_cancel_version:
                return None
            self._record_bucketwise_worker_error(e)
            logger.warning(f"Reflow bucketwise worker error (rank {dist.get_rank()}): {e}")
            return None

    def _copy_ready_ranges(self, fp16_flat, output, ranges, prepare=None):
        accelerator = get_accelerator()
        stream = self._bucketwise_h2d_stream
        if stream is None:
            stream = accelerator.current_stream()
        dst, src = fp16_flat.detach().view(-1), output.detach().view(-1)
        if dst.device.type != "cpu":
            chunk_elems = (self._reflow_config.h2d_chunk_size_mib << 20) // dst.element_size()
            ranges = reflow_utils.iter_range_chunks(ranges, chunk_elems)
        submitted = False
        try:
            for offset, size in ranges:
                src_slice = src.narrow(0, offset, size)
                if prepare is not None and not prepare(offset, size, src_slice):
                    break
                with accelerator.stream(stream):
                    dst.narrow(0, offset, size).copy_(src_slice, non_blocking=True)
                submitted = True
            # A cancelled Ready can already have submitted a prefix. Its one event protects
            # that pinned slot until replay, cleanup or the following backward can reuse it.
            if submitted and not accelerator.is_synchronized_device():
                event = accelerator.Event()
                event.record(stream)
                return event
        except Exception:
            # No event will reach the caller if a kernel or copy fails after an earlier DMA.
            if stream is not None:
                stream.synchronize()
            raise

    def _publish_staged_ready(self, sub_group_id, ranges):
        self._published_ready_groups.add(sub_group_id)
        output = self._cpu_grad_double_buffers[sub_group_id][1 - self._active_grad_buffer_idx]
        event = self._copy_ready_ranges(self.fp16_partitioned_groups_flat[sub_group_id], output, ranges)
        if event is not None:
            self._bucketwise_inflight_h2d_events.append(event)

    def _bucketwise_enqueue_range(self,
                                  sub_group_id,
                                  dest_offset,
                                  num_elements,
                                  copy_event=None,
                                  stage_checkpoint=False):
        # Accumulate a scheduled grad range; flush once the shard threshold is reached.
        try:
            clipped_range = self._bucketwise_clip_range_to_valid_partition(sub_group_id, dest_offset, num_elements)
            if clipped_range is None:
                return
            dest_offset, num_elements = clipped_range

            with self._bucketwise_enqueue_lock:
                if stage_checkpoint:
                    self._pending_checkpoint_groups.add(sub_group_id)
                if copy_event is not None:
                    self._bucketwise_last_copy_event_by_group[sub_group_id] = copy_event

                self._bucketwise_pending_ranges_by_group[sub_group_id].append((dest_offset, num_elements))
                self._bucketwise_pending_elems_by_group[sub_group_id] += num_elements
                pending_elem_count = self._bucketwise_pending_elems_by_group[sub_group_id]

                if pending_elem_count >= self._bucketwise_element_shard_size:
                    self._bucketwise_flush_group(sub_group_id,
                                                 use_pending_buffer=True,
                                                 combined_scale=self.loss_scale,
                                                 already_locked=True)
        except Exception as e:
            self._record_bucketwise_worker_error(e)

    def _bucketwise_flush_group(self, sub_group_id, use_pending_buffer, combined_scale, already_locked=False):
        """Submit all pending ranges for one subgroup with its optimizer settings at enqueue time."""
        if self._bucketwise_update_submit is None:
            return None
        ranges = self._bucketwise_pending_ranges_by_group[sub_group_id]
        if not ranges:
            return None
        merged_ranges = reflow_utils.merge_contiguous_ranges(ranges)
        ranges.clear()
        self._bucketwise_pending_elems_by_group[sub_group_id] = 0
        limit = self._bucketwise_get_valid_partition_element_count(sub_group_id)
        submit_ranges = reflow_utils.clip_ranges_to_valid_partition(merged_ranges, limit)
        if not submit_ranges:
            return None

        last_copy_event = self._bucketwise_last_copy_event_by_group.pop(sub_group_id, None)
        # A threshold flush already holds the enqueue lock. The step thread takes it here.
        lock = nullcontext() if already_locked else self._bucketwise_enqueue_lock
        with lock:
            target_step = self._ensure_bucketwise_group_step(sub_group_id)
            if target_step is None:
                return None
            group = self.optimizer.param_groups[self.sub_group_to_group_id[sub_group_id]]
            beta1, beta2 = group["betas"]
            grad_buffer_index = self._pending_grad_buffer_idx if use_pending_buffer else self._active_grad_buffer_idx
            # A checkpoint or a replay of an edited gradient can still own unpublished weights.
            stage_copy = (sub_group_id in self._pending_checkpoint_groups
                          or (not use_pending_buffer and sub_group_id in self._staged_ready_ranges))
            self._pending_checkpoint_groups.discard(sub_group_id)
            if stage_copy:
                self._staged_ready_ranges.setdefault(sub_group_id, [])
            else:
                self._published_ready_groups.add(sub_group_id)
            work_item = {
                "stage_copy": stage_copy,
                "sub_group_id": sub_group_id,
                "ranges": submit_ranges,
                "grad_buffer_index": grad_buffer_index,
                "combined_scale": combined_scale,
                "last_copy_event": last_copy_event,
                "cancel_version": self._bucketwise_cancel_version,
                "target_step": target_step,
                "lr": group["lr"],
                "betas": (beta1, beta2),
                "weight_decay": group["weight_decay"],
                "eps": group.get("eps", 0.0),
                "bias_correction": group.get("bias_correction", True),
                "maximize": group.get("maximize", False),
            }

        future = self._bucketwise_update_submit(self._bucketwise_chunk_update_worker, work_item)
        if not self._bucketwise_inflight_futures or self._bucketwise_inflight_futures[-1] is not future:
            self._bucketwise_inflight_futures.append(future)
        return future

    def _bucketwise_flush_all_pending(self, use_pending_buffer, combined_scale):
        for sub_group_id in range(len(self._bucketwise_pending_ranges_by_group)):
            self._bucketwise_flush_group(sub_group_id,
                                         use_pending_buffer=use_pending_buffer,
                                         combined_scale=combined_scale)

    def _bucketwise_cancel_inflight_updates(self):
        # An explicit zero_grad also discards previews that are already running.
        self._bucketwise_cancel_version += 1
        for f in list(getattr(self, "_bucketwise_inflight_futures", [])):
            try:
                batch = self._ready_batches.get(f)
                if batch is not None:
                    batch.discard()
                f.cancel()
            except Exception:
                pass

    def _bucketwise_commit_group_steps(self):
        target_steps = getattr(self, "_bucketwise_group_target_step", None)
        if not target_steps:
            return
        for sub_group_id, target_step in list(target_steps.items()):
            if sub_group_id >= len(self.fp32_partitioned_groups_flat):
                continue
            p = self.fp32_partitioned_groups_flat[sub_group_id]
            if p is None:
                continue
            state = self.optimizer.state[p]
            current_step = int(state.get('step', 0))
            if current_step < target_step:
                state['step'] = target_step

    def _ensure_bucketwise_group_step(self, sub_group_id):
        """Reserve one step per subgroup, including ranks with only padding and no Ready ranges."""
        target_step = self._bucketwise_group_target_step.get(sub_group_id)
        if target_step is not None:
            return target_step
        param = self.fp32_partitioned_groups_flat[sub_group_id]
        if param is None:
            return
        state = self.optimizer._ensure_param_state(param)
        target_step = int(state.get("step", 0)) + 1
        self._bucketwise_group_target_step[sub_group_id] = target_step
        return target_step

    def _bucketwise_wait_inflight_updates(self, discard_queued=False, publish_staged=False):
        # Wait for submitted bucketwise work (optimizer + H2D submit). H2D completion itself is
        # synchronized later via CUDA events.

        futures = self._bucketwise_inflight_futures
        if not futures and not (publish_staged and self._staged_ready_ranges):
            return
        if discard_queued:
            for future in futures:
                batch = self._ready_batches.get(future)
                if batch is not None:
                    batch.discard()
                future.cancel()
        self._drain_update_futures_collecting_h2d_events(futures, publish_staged=publish_staged)
        futures.clear()

    def _nvme_optimizer_step_subgroups(self, combined_scale, timer_names):
        """NVMe optimizer offload: a swappable-optimizer subgroup's state (fp32 master + the optimizer
        moments) lives on disk, so its CPU optimizer could not run during backward. Here -- after the
        cpu subgroups drained -- swap each subgroup's state in from nvme, run its full-partition
        CPU optimizer (param + state) reading the reduced grad from the active CPU half buffer, then
        swap the updated state back out. Serialized on the main thread: the partitioned optimizer swapper
        is single-threaded and not thread-safe, so subgroups are processed one at a time."""
        for sub_group_id in range(len(self.fp16_groups)):
            if not self._swappable_optimizer_subgroup(sub_group_id):
                continue
            self._optimizer_states_and_gradient_swap_in(sub_group_id, timer_names)
            full_ranges = []
            for param in self.fp16_groups[sub_group_id]:
                if not param.requires_grad:
                    continue
                pos = self.grad_position.get(self.get_param_id(param), None)
                if pos is None:
                    continue
                dest_offset, num_elems = pos[1], pos[2]
                if num_elems > 0:
                    full_ranges.append((dest_offset, num_elems))
            if full_ranges:
                self._bucketwise_pending_ranges_by_group[sub_group_id] = sorted(full_ranges, key=lambda x: x[0])
                self._bucketwise_pending_elems_by_group[sub_group_id] = sum(sz for _, sz in full_ranges)
                future = self._bucketwise_flush_group(sub_group_id,
                                                      use_pending_buffer=False,
                                                      combined_scale=combined_scale)
                if future is not None:
                    # Join the param-update worker AND collect its H2D event, so the next forward's
                    # all-gather waits for this subgroup's updated GPU param (written on the h2d
                    # stream) -- otherwise the forward races the in-flight copy and reads stale params.
                    self._drain_update_futures_collecting_h2d_events([future])
                self._ensure_bucketwise_group_step(sub_group_id)
                # Commit the step counter into optimizer.state BEFORE the state update. step_state_halfgrad
                # reads state['step'] for bias correction and assumes the param update already advanced it
                # (in CPU mode the commit precedes the async-state loop). Without committing first,
                # state['step']=0 -> bias_correction 1-beta**0 = 0 -> divide-by-zero -> NaN weights.
                self._bucketwise_commit_group_steps()
                # Optimizer-state (momentum) update; then join both the param-update worker and the
                # state worker before swapping the now-final state out to nvme.
                self._submit_state_updates([sub_group_id], combined_scale)
                self._wait_for_pending_state_updates(sub_group_id=sub_group_id)
            self._optimizer_states_and_gradient_swap_out(sub_group_id, timer_names)

    def _bucketwise_run_full_partition_updates(self, combined_scale, param_ids=None):
        # An earlier H2D can still read the pinned output that the new Ready kernels will overwrite.
        if param_ids is None or self._reflow_modified_gradient_ids:
            for event in self._coalesce_bucketwise_h2d_events():
                event.synchronize()

        for sub_group_id in range(len(self.fp16_groups)):
            if self._swappable_optimizer_subgroup(sub_group_id):
                continue  # nvme optimizer subgroups are handled in _nvme_optimizer_step_subgroups
            full_ranges = []
            for param in self.fp16_groups[sub_group_id]:
                if not param.requires_grad:
                    continue
                param_id = self.get_param_id(param)
                if param_ids is not None and param_id not in param_ids:
                    continue
                pos = self.grad_position.get(param_id, None)
                if pos is None:
                    continue
                dest_offset, num_elems = pos[1], pos[2]
                if num_elems > 0:
                    full_ranges.append((dest_offset, num_elems))

            if not full_ranges:
                continue

            shard_ranges = []
            shard_elems = 0
            for off, sz in sorted(full_ranges, key=lambda x: x[0]):
                shard_ranges.append((off, sz))
                shard_elems += sz
                if shard_elems >= self._bucketwise_element_shard_size:
                    self._bucketwise_pending_ranges_by_group[sub_group_id] = shard_ranges
                    self._bucketwise_pending_elems_by_group[sub_group_id] = shard_elems
                    self._bucketwise_flush_group(sub_group_id, use_pending_buffer=False, combined_scale=combined_scale)
                    shard_ranges = []
                    shard_elems = 0

            if shard_ranges:
                self._bucketwise_pending_ranges_by_group[sub_group_id] = shard_ranges
                self._bucketwise_pending_elems_by_group[sub_group_id] = shard_elems
                self._bucketwise_flush_group(sub_group_id, use_pending_buffer=False, combined_scale=combined_scale)

    def backward_prologue(self):
        # The pending state commit no longer overlaps a forward, so let it use every worker core.
        self._state_commit_overlaps_forward = False
        super().backward_prologue()

    def _state_commit_cpu_affinity_mask(self):
        if self._state_commit_overlaps_forward:
            return self._state_update_cpu_affinity_mask
        return self._state_backward_cpu_affinity_mask

    @staticmethod
    def _all_reduce_norm_entries_by_group(total_norms, process_groups):
        # Sum each entry over its process group (None: skip). Groups are visited in subgroup order, so every rank
        # issues the same collectives in the same order.
        indices_by_group = {}
        for index, process_group in enumerate(process_groups):
            if process_group is None:
                continue
            if id(process_group) not in indices_by_group:
                indices_by_group[id(process_group)] = (process_group, [])
            indices_by_group[id(process_group)][1].append(index)
        for process_group, indices in indices_by_group.values():
            if len(indices) == len(process_groups):
                dist.all_reduce(total_norms, op=dist.ReduceOp.SUM, group=process_group)
                continue
            index_tensor = torch.tensor(indices, device=total_norms.device)
            entries = total_norms.index_select(0, index_tensor)
            dist.all_reduce(entries, op=dist.ReduceOp.SUM, group=process_group)
            total_norms.index_copy_(0, index_tensor, entries)

    def _get_norm_groups(self):
        # Reduce every subgroup's sum of squares in one pair of collectives instead of one pair per subgroup like
        # the base; with Reflow's many small subgroups the per-subgroup latency dominates the norm.
        if not self.offload_optimizer:
            return super()._get_norm_groups()

        partition_groups = [self._get_sub_group_process_group(i) for i in range(len(self.fp16_groups))]
        if self.model_parallel_group is None:
            counts_every_rank = [False] * len(self.fp16_groups)
        else:
            counts_every_rank = [
                self._grad_norm_counts_every_rank(group, process_group)
                for group, process_group in zip(self.fp16_groups, partition_groups)
            ]
        local_sq = [
            self._local_grad_norm_squared(group, count_every_rank)
            for group, count_every_rank in zip(self.fp16_groups, counts_every_rank)
        ]

        device = get_accelerator().current_device_name()
        zero = torch.zeros((), device=device)
        # Keep subgroup sums in their original precision until the existing FP32 collective boundary.
        norm_entries = []
        for value in local_sq:
            if torch.is_tensor(value):
                norm_entries.append(value)
            elif value == 0.0:
                norm_entries.append(zero)
            else:
                norm_entries.append(torch.tensor(value, device=device))
        total_norms = torch.stack(norm_entries).float()
        # Match complete_grad_norm_calculation_for_cpu_offload: each subgroup sums over its own partition group,
        # then the model-parallel group, then (AutoEP expert subgroups only) the expert-parallel group. Subgroups
        # sharing a group are reduced together, so the usual single-group case stays one all-reduce.
        self._all_reduce_norm_entries_by_group(total_norms, partition_groups)
        if pg_utils.is_parallel(self.model_parallel_group):
            model_groups = [self.model_parallel_group if not count else None for count in counts_every_rank]
            self._all_reduce_norm_entries_by_group(total_norms, model_groups)
        expert_groups = [self._autoep_expert_parallel_group(group) for group in self.fp16_groups]
        expert_groups = [group if pg_utils.is_parallel(group) else None for group in expert_groups]
        self._all_reduce_norm_entries_by_group(total_norms, expert_groups)

        total_norms = total_norms.sqrt()
        mask_nan_or_inf_with_val_inplace(total_norms, device=total_norms.device)
        return list(total_norms.cpu().unbind())

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    @instrument_w_nvtx
    def step(self, closure=None):
        """
            Not supporting closure.
        """
        self._wait_for_accum_gradient()
        self._wait_for_pending_state_updates()
        self._pre_step()
        self._partition_all_parameters()
        # The backward epilogue orders the reductions. Missing gradients and NVMe subgroups
        # can read the host buffers without a Ready event, so finish their D2H writes.
        if self.copy_grad_stream is not None:
            self.copy_grad_stream.synchronize()
        enqueued_ready, ready_after_norm = self._complete_pending_gradient_window()
        # With a fixed unclipped scale, even a below-threshold tail already knows its
        # Ready transform. Publish from the captured slot while the GPU finishes the norm.
        flushed_before_norm = enqueued_ready or (self.clip_grad <= 0. and not self.dynamic_loss_scale)
        self._activate_pending_cpu_grads_if_ready(flush_pending=flushed_before_norm)
        self._overflow_check_and_loss_scale_update(update_scale=False)
        scaled_global_grad_norm = None
        if not self.overflow:
            norm_groups = self._get_norm_groups()
            scaled_global_grad_norm = combine_grad_norm_groups(norm_groups)
            if is_invalid_grad_norm(scaled_global_grad_norm):
                self.overflow = True

        if self.overflow:
            # Cleanup drops the work handles, so collect real failures before restoring weights.
            self._bucketwise_wait_inflight_updates(discard_queued=True)
            self._check_bucketwise_worker_error()
        preview_scale = self.loss_scale
        # ZeRO uses the updated scale for a successful step. Keep the scaler's counters
        # as well as its scale recoverable until the final Ready succeeds on every rank.
        scaler_state = vars(self.loss_scaler).copy() if self.dynamic_loss_scale and not self.overflow else None
        if self._loss_scale_update_and_overflow_cleanup():
            if self.swap_optimizer:
                self.optimizer_swapper.log_timers()
            return

        global_grad_norm = scaled_global_grad_norm / self.loss_scale
        combined_scale = float(self._combined_grad_scale(scaled_global_grad_norm))

        if combined_scale != preview_scale:
            # Clipping and dynamic scale growth both invalidate old-scale previews. Join
            # started work before the final pass fences H2D and reuses its pinned output.
            self._bucketwise_wait_inflight_updates(discard_queued=True)
            self._bucketwise_run_full_partition_updates(combined_scale=combined_scale)
        else:
            if not flushed_before_norm:
                self._bucketwise_flush_all_pending(use_pending_buffer=False, combined_scale=self.loss_scale)
            if ready_after_norm:
                if self._reflow_modified_gradient_ids:
                    self._bucketwise_wait_inflight_updates()
                self._bucketwise_run_full_partition_updates(combined_scale=combined_scale, param_ids=ready_after_norm)

        timer_names = set()
        # Param updates already ran bucketwise during backward; here we only commit per-group
        # step counters and submit the deferred async state updates.
        self._bucketwise_wait_inflight_updates(publish_staged=True)
        try:
            self._check_bucketwise_worker_error()
        except Exception:
            if scaler_state is not None:
                vars(self.loss_scaler).update(scaler_state)
            raise
        self._global_grad_norm = global_grad_norm
        # NVMe optimizer subgroups deferred their CPU-Adam (state not resident during backward); run
        # it now with the state swapped in around it, before committing step counters / the async loop.
        if self.swap_optimizer:
            self._nvme_optimizer_step_subgroups(combined_scale, timer_names)
        if len(self._bucketwise_group_target_step) < len(self.fp16_groups):
            for sub_group_id in range(len(self.fp16_groups)):
                if not self._swappable_optimizer_subgroup(sub_group_id):
                    self._ensure_bucketwise_group_step(sub_group_id)
        self._bucketwise_commit_group_steps()
        # The commits submitted below overlap the next forward until its backward starts.
        self._state_commit_overlaps_forward = True
        resident_groups = (i for i in reversed(range(len(self.fp16_groups)))
                           if not self._swappable_optimizer_subgroup(i))
        self._submit_state_updates(resident_groups, combined_scale)
        if self._bucketwise_inflight_h2d_events:
            # Host completion covers every reader. In asynchronous mode the next forward's
            # compute and all-gather and the next gradient-slot reuse each need a stream edge.
            h2d_events = self._coalesce_bucketwise_h2d_events()
            if self._sync_bucketwise_h2d_before_forward:
                for event in h2d_events:
                    event.synchronize()
            else:
                compute_stream = get_accelerator().current_stream()
                for event in h2d_events:
                    compute_stream.wait_event(event)
                    self.copy_grad_stream.wait_event(event)
                self._get_param_coordinator().wait_for_external_allgather_dependencies(h2d_events)
        # NVMe param offload: the bucketwise workers leave the updated FP16/BF16 weights in the CPU
        # half buffer (an nvme subgroup has no GPU param to H2D into), so swap them out to nvme here,
        # on the main thread -- the partitioned param swapper holds single shared mutable state and is
        # not thread-safe. Confined to nvme subgroups (fp16 flat is None); cpu/gpu paths are untouched.
        for sub_group_id in range(len(self.fp16_groups)):
            if self.fp16_partitioned_groups_flat[sub_group_id] is None:
                ready_weights = self._cpu_grad_double_buffers[sub_group_id][self._pending_grad_buffer_idx]
                self._partitioned_params_swap_out(sub_group_id, source=ready_weights)

        self._post_step(timer_names)

        memory_stats = get_accelerator().memory_stats()
        alloc_retries = memory_stats.get("num_alloc_retries")
        if alloc_retries is None:
            alloc_retries = 0
        if alloc_retries > self.n_caching_allocator_flushes:
            if dist.get_rank() == 0:
                logger.warning(
                    "%d pytorch allocator cache flushes since last step. this happens "
                    "when there is high memory pressure and is detrimental to "
                    "performance. if this is happening frequently consider adjusting "
                    "settings to reduce memory consumption. If you are unable to "
                    "make the cache flushes go away consider adding "
                    "get_accelerator().empty_cache() calls in your training loop to ensure "
                    "that all ranks flush their caches at the same time",
                    alloc_retries - self.n_caching_allocator_flushes)
            self.n_caching_allocator_flushes = alloc_retries

    # ------------------------------------------------------------------ #
    # overflow
    # ------------------------------------------------------------------ #
    def _copy_pinned_master_to_device(self, dst, src):
        if src.numel() == 0:
            return super()._copy_pinned_master_to_device(dst, src)
        chunk_elems = min(src.numel(), (self._reflow_config.h2d_chunk_size_mib << 20) // src.element_size())
        for comm_dtype, bucket in self.ipg_buckets.items():
            if bucket.elements:
                continue
            staging = self._unused_ipg_buffer(bucket, src.dtype)
            if staging is None or staging.device != dst.device or staging.numel() < chunk_elems:
                continue
            # A logically empty bucket can still have a reduction or a D2H read in flight.
            stream = get_accelerator().current_stream()
            if self.reduce_and_partition_stream is not None and self.reduce_and_partition_stream != stream:
                stream.wait_stream(self.reduce_and_partition_stream)
            event = self._ipg_read_events.get(comm_dtype)
            if event is not None:
                stream.wait_event(event)
                del self._ipg_read_events[comm_dtype]
            break
        else:
            if src.numel() == chunk_elems:
                return super()._copy_pinned_master_to_device(dst, src)
            staging = dst.new_empty((chunk_elems, ), dtype=src.dtype)

        copies_done = get_accelerator().Event()
        try:
            for offset in range(0, src.numel(), chunk_elems):
                count = min(chunk_elems, src.numel() - offset)
                chunk = staging[:count]
                chunk.copy_(src.narrow(0, offset, count), non_blocking=True)
                dst.narrow(0, offset, count).copy_(chunk)
            copies_done.record()
        except BaseException:
            # A later chunk may fail after an earlier copy started using the borrowed storage.
            get_accelerator().current_stream().synchronize()
            raise
        copies_done.synchronize()

    def _restore_fp16_params_from_master(self, sub_group_id):
        """Undo a discarded (overflow) step: re-cast the FP32 master weights to FP16/BF16 and
        scatter them back onto the GPU params, overwriting any FP16 preview the bucketwise workers
        wrote during backward (the FP32 master itself is the unchanged pre-step weight)."""
        if sub_group_id >= len(self.fp16_partitioned_groups_flat):
            return
        fp16_flat = self.fp16_partitioned_groups_flat[sub_group_id]
        fp32_flat = self.fp32_partitioned_groups_flat[sub_group_id]
        if fp32_flat is None:
            return
        if fp16_flat is None:
            # nvme subgroup: no GPU fp16 flat to restore -- re-swap the unchanged FP32 master so the
            # on-disk weights stay the pre-step values after a discarded (overflow) step.
            self._partitioned_params_swap_out(sub_group_id)
            return
        if self._swappable_optimizer_subgroup(sub_group_id):
            # nvme optimizer subgroup: its CPU-Adam is deferred to step() (skipped entirely on
            # overflow), so the GPU param was never updated during backward and already holds the
            # pre-step weight. The FP32 master lives on nvme (swapped out), so there is nothing to
            # copy from here -- the discarded step leaves both the param and the state untouched.
            return
        self._reassign_or_swap_out_partitioned_parameters(sub_group_id)

    # ------------------------------------------------------------------ #
    # gradient access
    # ------------------------------------------------------------------ #

    def get_fp32_grad_for_param(self, param) -> Tensor:
        raise NotImplementedError(_REFLOW_FLAT_GRAD_MSG.format(api="get_fp32_grad_for_param"))

    def set_fp32_grad_for_param(self, value, param):
        raise NotImplementedError(_REFLOW_FLAT_GRAD_MSG.format(api="set_fp32_grad_for_param"))

    def _pending_gradient_partition(self, param_id):
        subgroup, offset, length = self.grad_position[param_id]
        return self._get_cpu_grad_buffer(subgroup, pending=True).narrow(0, offset, length)

    def get_local_fp32_grad_for_param(self, param) -> Tensor:
        if not param.requires_grad:
            return None
        param_id = self.get_param_id(param)
        numel = param.partition_numel()
        device = get_accelerator().current_device_name()
        has_partition = numel * self._get_param_partition_rank(param) < param.ds_numel
        if param_id not in self._reflow_window_grad_ids or not has_partition:
            return torch.zeros(numel, dtype=torch.float32, device=device)
        self._wait_for_accum_gradient(param_id)
        if self._bucketwise_worker_error is not None:
            raise self._bucketwise_worker_error
        self.copy_grad_stream.synchronize()
        return self._pending_gradient_partition(param_id).to(device=device, dtype=torch.float32)

    def set_local_grad_for_param(self, value, param):
        if not param.requires_grad:
            return
        param_id = self.get_param_id(param)
        numel = param.partition_numel()
        assert value.numel() == numel, f"Number of elements do not match: {value.numel()} != {numel}"
        if numel * self._get_param_partition_rank(param) >= param.ds_numel:
            return

        self._wait_for_gradient_edits([param_id])
        if self._bucketwise_worker_error is not None:
            raise self._bucketwise_worker_error
        gradient = self._pending_gradient_partition(param_id)
        rounded = value.detach().reshape(-1).to(dtype=gradient.dtype)
        self._write_gradient_partition(param_id, gradient, rounded)
        self._refresh_edited_gradient_overflow()

    def _wait_for_gradient_edits(self, param_ids):
        # Join readers once per API call. In FP16 the overflow rescan also reads unedited partitions.
        if self.dtype == torch.float16:
            self._wait_for_accum_gradient()
        else:
            for param_id in param_ids:
                self._wait_for_accum_gradient(param_id)
        self._bucketwise_drain_enq_executor()
        self._bucketwise_wait_inflight_updates()
        self.copy_grad_stream.synchronize()
        if self._single_grad_buffer:
            for group in {self.grad_position[param_id][0] for param_id in param_ids}:
                self._wait_for_pending_state_updates(sub_group_id=group)

    def _write_gradient_partition(self, param_id, gradient, rounded):
        # Compute the norm from the rounded values that Ready and Persist will actually read.
        norm = self._constant_buffered_norm2(rounded.to(get_accelerator().current_device_name()))
        gradient[:rounded.numel()].copy_(rounded)
        if rounded.numel() < gradient.numel():
            gradient[rounded.numel():].zero_()
        self._reflow_window_grad_ids.add(param_id)
        self._pending_grad_ready = True
        self._reflow_modified_gradient_ids.add(param_id)
        self.norm_for_param_grads[param_id] = norm

    def _refresh_edited_gradient_overflow(self):
        if self.dtype != torch.float16:
            return
        # A replacement can remove an earlier overflow. Check only gradients in this window.
        invalid = any(
            has_inf_or_nan(self._pending_gradient_partition(param_id)) for param_id in self._reflow_window_grad_ids)
        # partition_grads and has_overflow both access this flag on the reduction stream.
        with get_accelerator().stream(self.reduce_and_partition_stream):
            self.inf_or_nan_tracker.fill_(invalid)

    def _full_gradient_edit_error(self, error, group):
        if dist.get_world_size(group=group) > 1:
            flag = get_accelerator().ByteTensor([int(error is not None)])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
            if flag.item() and error is None:
                return RuntimeError("Reflow full gradient update failed on another rank")
        return error

    def _update_full_gradient_partition(self, param, group, update_func):
        param_id = self.get_param_id(param)
        gradient = self._pending_gradient_partition(param_id)
        numel = gradient.numel()
        ranks = dist.get_world_size(group=group)
        start = self._get_param_partition_rank(param) * numel
        # Gather the stored halves directly into rank order. Only the callback needs an FP32 tensor.
        gathered = torch.empty(ranks * numel, dtype=gradient.dtype, device=param.device)
        partition = gathered.narrow(0, start, numel)
        if param_id in self._reflow_window_grad_ids:
            partition.copy_(gradient, non_blocking=True)
        else:
            partition.zero_()
        if ranks > 1:
            dist.all_gather_into_tensor(gathered, partition, group=group)
        full_grad = gathered[:param.ds_numel].view(param.ds_shape).float()
        del gathered, partition

        error = None
        try:
            value = update_func(full_grad, param)
            if value.numel() != param.ds_numel:
                raise ValueError(f"Expected {param.ds_numel} gradient elements, got {value.numel()}")
            # A callback can return a non-contiguous tensor. Select only the real elements on this rank.
            shard = value.detach().reshape(-1)[start:start + numel].to(dtype=gradient.dtype)
        except Exception as exc:
            error = exc
        # A rank-local callback failure must leave this parameter intact on every owner. Vote before
        # writing its shards, so a caller can fix the callback and retry without repeating backward.
        error = self._full_gradient_edit_error(error, group)
        if error is None and start < param.ds_numel:
            self._write_gradient_partition(param_id, gradient, shard)
        elif error is not None and ranks == 1 and not get_accelerator().is_synchronized_device():
            # With no distributed vote or write-back, the H2D may still own the pinned input.
            get_accelerator().current_stream().synchronize()
        return error

    def update_fp32_grad_for_param_vectorized(self, update_func, param_list):
        params = [p for p in param_list if p.requires_grad]
        if not params:
            return
        self._wait_for_gradient_edits([self.get_param_id(p) for p in params])
        # A caller may own expert parameters in a different group. Finish each group's collectives
        # even if another group failed, and never require ranks outside the parameter's group.
        group_errors = {}
        for param in params:
            group = self._get_param_partition_group(param)
            if group not in group_errors:
                group_errors[group] = self._full_gradient_edit_error(self._bucketwise_worker_error, group)
            if group_errors[group] is None:
                group_errors[group] = self._update_full_gradient_partition(param, group, update_func)
        self._refresh_edited_gradient_overflow()
        for error in group_errors.values():
            if error is not None:
                raise error

    def get_fp32_grad_partitions(self):
        raise NotImplementedError(_REFLOW_FLAT_GRAD_MSG.format(api="get_fp32_grad_partitions"))

    def offload_states(self,
                       include: Container[OffloadStateTypeEnum] = None,
                       device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                       pin_memory: bool = True,
                       non_blocking: bool = False):
        raise NotImplementedError(_REFLOW_FLAT_GRAD_MSG.format(api="offload_states"))

    def finalize_gradient_accumulation_boundary(self):
        # Unmanaged gradient accumulation marks the boundary only after backward, but Reflow submits its CPU
        # optimizer work during backward when it sees the boundary, so the update would silently never run.
        raise NotImplementedError("Reflow requires managed gradient accumulation; set "
                                  "'managed_gradient_accumulation' to true or disable reflow.")

    # ------------------------------------------------------------------ #
    # checkpoint and hp-param access
    # ------------------------------------------------------------------ #
    # step() returns while background workers still commit the FP32 master and optimizer state, so
    # checkpoints and the safe_get/set_* APIs wait for that commit; the training loop does not.

    def checkpoint_event_prologue(self):
        self._wait_for_pending_state_updates()
        local_error = self._bucketwise_worker_error
        if dist.get_world_size() > 1:
            # Checkpoints involve every rank, including ranks outside this optimizer's DP group.
            # Reuse the training error flag, but coordinate once globally before any checkpoint file is written.
            error_flag = self._bucketwise_error_flag
            if error_flag is None:
                error_flag = get_accelerator().ByteTensor([0])
                self._bucketwise_error_flag = error_flag
            error_flag.fill_(1 if local_error is not None else 0)
            dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
            if error_flag[0].item() != 0 and local_error is None:
                raise RuntimeError("Reflow asynchronous optimizer update failed on another rank; aborting checkpoint")
        if local_error is not None:
            raise local_error
        super().checkpoint_event_prologue()

    def state_dict(self):
        self._wait_for_pending_state_updates()
        if self._bucketwise_worker_error is not None:
            raise self._bucketwise_worker_error
        return super().state_dict()

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None,
                        param_shapes=None):
        # A commit that lands after the load would overwrite the loaded state.
        self._wait_for_pending_state_updates()
        return super().load_state_dict(state_dict_list,
                                       load_optimizer_states=load_optimizer_states,
                                       load_from_fp32_weights=load_from_fp32_weights,
                                       checkpoint_folder=checkpoint_folder,
                                       load_serial=load_serial,
                                       param_shapes=param_shapes)

    def get_lean_optimizer_state(self):
        self._wait_for_pending_state_updates()
        if self._bucketwise_worker_error is not None:
            raise self._bucketwise_worker_error
        return super().get_lean_optimizer_state()

    def get_full_hp_param(self, param, optim_state_key=None) -> Tensor:
        if not param.requires_grad:
            return None
        # The inherited full-parameter getter is collective; report a failure on all
        # participating ranks before any one of them exits the following all-gather.
        self._wait_for_pending_state_updates()
        local_error = self._bucketwise_worker_error
        process_group = self._get_param_partition_group(param)
        if dist.get_world_size(group=process_group) > 1:
            error_flag = get_accelerator().ByteTensor([1 if local_error is not None else 0])
            dist.all_reduce(error_flag, op=dist.ReduceOp.MAX, group=process_group)
            if error_flag[0].item() and local_error is None:
                raise RuntimeError(
                    "Reflow asynchronous optimizer update failed on another rank; aborting state access")
        if local_error is not None:
            raise local_error
        return super().get_full_hp_param(param, optim_state_key=optim_state_key)

    def _get_fp32_opt_state_partition(self, param, release_swap_buffers, optim_state_key=None):
        # Every get/set hp-param API goes through here.
        self._wait_for_pending_state_updates()
        if self._bucketwise_worker_error is not None:
            raise self._bucketwise_worker_error
        return super()._get_fp32_opt_state_partition(param, release_swap_buffers, optim_state_key=optim_state_key)
