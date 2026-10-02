# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import gc
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Container, List

import psutil
import torch

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
from deepspeed.runtime.reflow import reflow_utils
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum
from deepspeed.runtime.zero.partition_parameters import Parameter, Tensor
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3, INIT_OPTIMIZER_TIMER
from deepspeed.runtime.utils import has_inf_or_nan, is_model_parallel_parameter, mask_nan_or_inf_with_val_inplace
from deepspeed.utils import logger
from deepspeed.utils.nvtx import instrument_w_nvtx

_REFLOW_FLAT_GRAD_FREED_MSG = ("Reflow frees the flat grad buffer; {api} is not supported under reflow.")
_DEFAULT_BUCKETWISE_SHARD_ELEMS = 50000000
_DEFAULT_BUCKETWISE_MIN_SUBMIT_ELEMS = 262144
# A power of two, so every full slice is a whole number of SIMD blocks and the kernel's scalar tail is the same
# elements as in an unsliced call; the sliced commit stays bit-exact.
_STATE_COMMIT_SLICE_ELEMS = 1 << 25


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
        # calls _setup_for_real_optimizer / initialize_optimizer_states, which are overridden
        # here and read these fields.
        self._cpu_grad_double_buffers = []
        self._active_grad_buffer_idx = 0
        self._pending_grad_ready = False
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

        self._setup_async_state_executor()
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

    def _setup_async_state_executor(self):
        self._reset_bucketwise_handles()
        self._worker_cpu_affinity_mask = None
        self._state_update_cpu_affinity_mask = None

        main_thread_cores = self.main_thread_cores
        enable_cpu_affinity = self.enable_cpu_affinity
        # Automatically find NUMA-local cores for this rank, then split them into a main-thread set
        # and per-NUMA-node optimizer-worker groups (so workers never straddle a NUMA boundary).
        total_cpus, available_cores, node_worker_groups, main_cores, worker_cores = \
            reflow_utils.plan_cpu_core_layout(main_thread_cores, self._reflow_config.main_thread_core_type,
                                             self._reflow_config.worker_core_type)
        self._worker_node_groups = node_worker_groups

        self._worker_cpu_affinity_mask = set(worker_cores)
        # While the state commit overlaps the next forward it stays on a few worker cores, away from the
        # main-thread cores at the front of the list: every busy core lowers the turbo frequency the CPU
        # allows, which slows the launch-bound forward. See _state_commit_cpu_affinity_mask.
        state_update_core_count = min(max(1, int(self.state_update_cores)), len(worker_cores))
        self._state_update_cpu_affinity_mask = set(worker_cores[-state_update_core_count:]) if worker_cores else None
        backward_count = self._reflow_config.state_update_backward_cores
        backward_cores = worker_cores[-backward_count:] if backward_count is not None else worker_cores
        self._state_backward_cpu_affinity_mask = set(backward_cores)

        original_affinity = None
        if self._reflow_config.pin_main_thread:
            if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
                raise ValueError("Reflow pin_main_thread requires OS thread-affinity support")
            original_affinity = (threading.get_native_id(), os.sched_getaffinity(0))

        if enable_cpu_affinity and available_cores:
            try:
                # Keep the main process on the whole NUMA-local slice, not only main_thread_cores: the ZeRO-3 forward
                # is bound by kernel launches from this thread. The workers still pin themselves to worker_cores.
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

        self._state_update_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reflow_state_update")

        self._setup_bucketwise_executors()

    def _setup_bucketwise_executors(self):
        bucketwise_cores_per_worker = max(1, int(self.bucketwise_cores_per_worker))

        try:
            # Build one affinity mask per worker by chunking EACH NUMA node's worker cores
            # separately, so a worker's cores never span two NUMA nodes.
            node_groups = getattr(self, "_worker_node_groups", None)
            if not node_groups:
                fallback_cores = sorted(list(
                    self._worker_cpu_affinity_mask)) or reflow_utils.resolve_available_cores()[1]
                node_groups = [fallback_cores]

            self._bucketwise_worker_cpu_affinity_masks = []
            for node_cores in node_groups:
                for start_idx in range(0, len(node_cores), bucketwise_cores_per_worker):
                    chunk = node_cores[start_idx:start_idx + bucketwise_cores_per_worker]
                    if chunk:
                        self._bucketwise_worker_cpu_affinity_masks.append(set(chunk))
            if not self._bucketwise_worker_cpu_affinity_masks:
                self._bucketwise_worker_cpu_affinity_masks = [set(sorted(self._worker_cpu_affinity_mask) or [0])]

            max_bucketwise_workers = len(self._bucketwise_worker_cpu_affinity_masks)
            total_worker_cores = sum(len(node) for node in node_groups)
            if dist.get_rank() == 0:
                logger.info(f"Bucketwise worker configuration (NUMA-aware): {max_bucketwise_workers} workers "
                            f"across {len(node_groups)} NUMA node group(s), "
                            f"{bucketwise_cores_per_worker} cores per worker, "
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
                                                              thread_name_prefix="reflow_bucketwise_update",
                                                              initializer=initializer)

        def _submit_bucketwise_task(fn, *args, **kwargs):
            if initializer is not None:
                return self._bucketwise_update_executor.submit(fn, *args, **kwargs)

            # Select the mask only here: selecting again in a caller would skip masks in an even-sized pool.
            worker_idx = self._bucketwise_worker_idx
            self._bucketwise_worker_idx = (self._bucketwise_worker_idx + 1) % len(
                self._bucketwise_worker_cpu_affinity_masks)
            affinity_mask = self._bucketwise_worker_cpu_affinity_masks[worker_idx]

            def wrapped_fn(*a, **kw):
                self._pin_current_thread(affinity_mask)
                return fn(*a, **kw)

            return self._bucketwise_update_executor.submit(wrapped_fn, *args, **kwargs)

        self._bucketwise_update_submit = _submit_bucketwise_task
        self._bucketwise_update_futures = []
        self._bucketwise_inflight_futures = []
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

        self._bucketwise_group_step_incremented = set()
        self._bucketwise_group_target_step = {}
        self._bucketwise_inflight_h2d_events = []
        self._bucketwise_cancel_version = 0
        # Cap the per-shard size at _DEFAULT_BUCKETWISE_SHARD_ELEMS (also the fallback when
        # reduce_bucket_size is unset) so each subgroup flushes to the CPU-Adam worker partway through
        # backward, overlapping the CPU optimizer, instead of waiting for the whole reduce bucket.
        self._bucketwise_element_shard_size = min(getattr(self, 'reduce_bucket_size', _DEFAULT_BUCKETWISE_SHARD_ELEMS),
                                                  _DEFAULT_BUCKETWISE_SHARD_ELEMS)
        self._bucketwise_min_submit_range_elems = _DEFAULT_BUCKETWISE_MIN_SUBMIT_ELEMS
        self._bucketwise_enqueue_lock = threading.Lock()
        self._bucketwise_pending_events = {}
        # First real (non-cancel) error raised by a bucketwise worker this iteration; step()
        # all-reduces a flag derived from it and re-raises so the step aborts on every rank.
        self._bucketwise_worker_error = None
        self._bucketwise_error_flag = None  # preallocated device flag for the per-step error all-reduce

    def _reset_bucketwise_handles(self):
        self._bucketwise_update_executor = None
        self._bucketwise_update_submit = None
        self._bucketwise_update_futures = None
        self._bucketwise_inflight_futures = None
        self._bucketwise_worker_cpu_affinity_masks = None
        self._worker_node_groups = None
        self._bucketwise_worker_idx = 0
        self._bucketwise_h2d_stream = None
        self._bucketwise_enq_executor = None
        self._bucketwise_enq_submit = None

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
        for attr in ("_bucketwise_enq_executor", "_bucketwise_update_executor", "_state_update_executor"):
            executor = getattr(self, attr, None)
            if executor is not None:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
        # The base _unpin_offload_buffers does not know Reflow's half-precision grad buffers; under the native
        # pinning backend they would otherwise stay page-locked after destroy. Safe now that no worker uses them.
        # The native unpin frees the memory, so drop the references too.
        for grad_buffers in getattr(self, "_cpu_grad_double_buffers", None) or []:
            for grad_buffer in grad_buffers or []:
                if grad_buffer is not None and grad_buffer.device.type == 'cpu':
                    get_accelerator().unpin_memory(grad_buffer)
        self._cpu_grad_double_buffers = []
        try:
            super().destroy()
        finally:
            if self._main_thread_affinity is not None:
                thread_id, cpus = self._main_thread_affinity
                self._main_thread_affinity = None
                os.sched_setaffinity(thread_id, cpus)

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

    def _setup_for_real_optimizer(self):
        # cpu_bucketwise stores gradients exclusively in _cpu_grad_double_buffers, so the
        # large flat grad buffer (grad_partitions_flat_buffer) is wasted memory here. The
        # parent setup unconditionally allocates it, so free it (and its pinned backing) to
        # reclaim ~2 bytes/param of CPU memory in the bucketwise case.
        super()._setup_for_real_optimizer()
        self.grad_partitions_flat_buffer = None
        # GPU-side inf/nan accumulator for the offloaded grads, since the flat buffer that the base
        # has_overflow scans is freed above (lazily created on the grad device in partition_grads).
        self._reflow_grad_overflow = None
        # Clear the name-mangled __param_id_to_grad_partition dict so the narrow() views into
        # the buffer freed above are dropped and its storage can actually be reclaimed.
        self._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition = {}
        # The parent pins grad_partitions_flat_buffer, so dropping the references above only returns
        # the block to torch's pinned-host caching allocator, which keeps the physical memory resident
        # (the one large block is never reused by the smaller per-subgroup bucketwise buffers). Empty
        # the host cache so the ~2 bytes/param of pinned memory is actually returned to the OS.
        gc.collect()
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()

    def _pin_current_thread(self, cpu_affinity_mask):
        if cpu_affinity_mask is None:
            return
        try:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, cpu_affinity_mask)
        except (OSError, AttributeError):
            pass

    def _drain_update_futures_collecting_h2d_events(self, futures):
        for f in futures:
            try:
                ev = f.result()
                if ev is not None:
                    self._bucketwise_inflight_h2d_events.append(ev)
            except Exception:
                pass

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

    def _get_enqueue_lock(self):
        return self._bucketwise_enqueue_lock

    def _record_bucketwise_worker_error(self, exc):
        # Record the first real (non-cancel) bucketwise worker failure under the shared enqueue
        # lock; step() reads it to abort deterministically. Keep only the first so the original
        # cause survives later cascading failures.
        with self._get_enqueue_lock():
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
    # partitioned param swap-out / CPU fp16 buffer sync
    # ------------------------------------------------------------------ #
    def _partitioned_params_swap_out(self, i, use_updated_weights=False):
        offset = 0
        fp32_param = self.fp32_partitioned_groups_flat[i]
        assert fp32_param is not None, f'fp32 parameters of sub_group {i} is None'
        # Post-step, the CPU half_params buffer holds the freshly updated FP16/BF16 weights, so swap
        # those out directly (a byte copy into the FP16 swap pool, avoiding an FP32->FP16 re-cast).
        # At init the buffer is still torch.zeros, so callers there pass use_updated_weights=False to
        # swap the pretrained FP32 master instead (cast to FP16 in the pool, like the base optimizer).
        half_param_src = None
        if use_updated_weights:
            try:
                if i < len(self._cpu_grad_double_buffers) and self._cpu_grad_double_buffers[i] is not None:
                    half_param_src = self._cpu_grad_double_buffers[i][self._pending_grad_buffer_idx]
            except Exception:
                half_param_src = None

        swap_fp16_params = []
        swap_fp32_params = []
        for param, partitioned_param in zip(self.fp16_groups[i], self.fp16_partitioned_groups[i]):
            if half_param_src is not None:
                src = half_param_src.narrow(0, offset, partitioned_param.ds_numel)
            else:
                src = fp32_param.narrow(0, offset, partitioned_param.ds_numel)
            if partitioned_param.status == PartitionedParamStatus.AVAILABLE:
                partitioned_param.data.copy_(src.data)
            else:
                swap_fp32_params.append(src)
                swap_fp16_params.append(param)
            offset += partitioned_param.ds_numel

        if len(swap_fp16_params):
            swap_fp16_params[0].nvme_swapper.swap_out_partitioned_params(dst_fp16_params=swap_fp16_params,
                                                                         src_fp32_params=swap_fp32_params)

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
                subgroup_gradient_buffer = torch.zeros(num_elements,
                                                       dtype=grad_dtype,
                                                       device=self.subgroup_to_device[i])
                if self.offload_optimizer_pin_memory and self.subgroup_to_device[i] == 'cpu':
                    subgroup_gradient_buffer = get_accelerator().pin_memory(subgroup_gradient_buffer)
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
    @instrument_w_nvtx
    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        buffers = []

        # Cache loop-invariant values (avoid repeated getattr/property/C++ binding calls).
        is_boundary = self.is_gradient_accumulation_boundary()
        micro_step_id = self.micro_step_id
        copy_grad_stream = self.copy_grad_stream
        current_stream = get_accelerator().current_stream()
        stream_needs_sync = copy_grad_stream != current_stream
        is_sync_device = get_accelerator().is_synchronized_device()
        accumulate_submit = getattr(self, "_bucketwise_update_submit", None)
        accum_futures_dict = self._bucketwise_accumulation_futures

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

            if grad_partition.dtype != self.gradient_accumulation_dtype:
                converted_grad = grad_partition.to(self.gradient_accumulation_dtype)
            else:
                converted_grad = grad_partition
            # The base flat grad buffer that has_overflow normally scans is freed on this path, so
            # accumulate any inf/nan in the offloaded grads on the GPU here (no host sync); the flag
            # is folded into has_overflow to drive FP16 dynamic-loss-scaling.
            if self.dtype == torch.float16:
                if self._reflow_grad_overflow is None:
                    self._reflow_grad_overflow = torch.zeros((), dtype=torch.uint8, device=converted_grad.device)
                self._reflow_grad_overflow.logical_or_(has_inf_or_nan(converted_grad))

            grad_numel = grad_buffer.numel()

            if micro_step_id == 0:  # don't accumulate, first micro step
                if self._single_grad_buffer:
                    # Slot 0 is also what the previous step's state commit reads, so wait for that commit before
                    # overwriting it. Usually a no-op: the commit finishes during the forward.
                    self._wait_for_pending_state_updates(sub_group_id=sub_group_id)
                pending_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=True)
                pending_slice = pending_buffer.narrow(0, dest_offset, grad_numel)

                if stream_needs_sync:
                    copy_grad_stream.wait_stream(current_stream)
                with get_accelerator().stream(copy_grad_stream):
                    pending_slice.copy_(converted_grad.view(pending_slice.shape), non_blocking=True)
                    copy_event = get_accelerator().Event()
                    copy_event.record(copy_grad_stream)
                # No gradient accumulation: this micro-step is also the boundary, so use the GPU
                # grad tensor directly for the norm/enqueue below.
                if is_boundary:
                    grad_buffer = converted_grad
                    grad_numel = grad_buffer.numel()
            else:
                if is_boundary:
                    pending_buffer = self._get_cpu_grad_buffer(sub_group_id, pending=True)
                    dst_slice = pending_buffer.narrow(0, dest_offset, grad_numel)
                    if stream_needs_sync:
                        copy_grad_stream.wait_stream(current_stream)
                    with get_accelerator().stream(copy_grad_stream):
                        accum_gpu = dst_slice.to(converted_grad.device, non_blocking=True)
                        accum_gpu.add_(converted_grad.view(accum_gpu.shape))
                        accum_event = get_accelerator().Event()
                        accum_event.record(copy_grad_stream)
                        dst_slice.copy_(accum_gpu, non_blocking=True)
                        copy_event = get_accelerator().Event()
                        copy_event.record(copy_grad_stream)
                    if stream_needs_sync:
                        current_stream.wait_event(accum_event)
                        # accum_gpu is allocated on copy_grad_stream but read by the grad-norm on
                        # current_stream below; the wait_event orders that read but does not keep the
                        # block alive, so record it against current_stream too or the allocator can
                        # recycle it (into copy_grad_stream's free list) before the norm read runs.
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
                    if stream_needs_sync:
                        copy_grad_stream.wait_stream(current_stream)
                    with get_accelerator().stream(copy_grad_stream):
                        src_slice.copy_(converted_grad.view(src_slice.shape), non_blocking=True)
                    copy_event = get_accelerator().Event()
                    copy_event.record(copy_grad_stream)
                    if accumulate_submit is not None:
                        # Capture values eagerly via default args (avoid late-binding closure bug).
                        def _accumulate_worker(_copy_event=copy_event, _dst_slice=dst_slice, _src_slice=src_slice):
                            _copy_event.synchronize()
                            ReflowCPUAdam.bf16_accumulate(_dst_slice.view(-1), _src_slice.view(-1))

                        future = accumulate_submit(_accumulate_worker)
                        if accum_futures_dict is not None:
                            accum_futures_dict.setdefault(sub_group_id, []).append(future)
                        grad_buffer = dst_slice
                    else:
                        copy_event.synchronize()
                        ReflowCPUAdam.bf16_accumulate(dst_slice.view(-1), src_slice.view(-1))
                        grad_buffer = dst_slice
                    grad_numel = grad_buffer.numel()

            if is_boundary:
                self.norm_for_param_grads[param_id] = self._constant_buffered_norm2(grad_buffer)
                # NVMe subgroups have no resident optimizer state during backward, so their CPU step waits for
                # step(); only mark the buffer ready. Other subgroups run the range during backward.
                self._pending_grad_ready = True
                if not self._swappable_optimizer_subgroup(sub_group_id):
                    self._bucketwise_schedule_enqueue_range(sub_group_id,
                                                            dest_offset,
                                                            grad_numel,
                                                            copy_event=copy_event)

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
        self._pending_grad_ready = False

        # Reset per-iteration pending/inflight state.
        self._bucketwise_group_step_incremented.clear()
        self._bucketwise_group_target_step.clear()
        for sub_group_id in range(len(self._bucketwise_pending_ranges_by_group)):
            self._bucketwise_pending_ranges_by_group[sub_group_id].clear()
            self._bucketwise_pending_elems_by_group[sub_group_id] = 0
        self._bucketwise_inflight_h2d_events.clear()
        self._bucketwise_pending_events.clear()
        if getattr(self, "_bucketwise_update_futures", None) is not None:
            self._bucketwise_update_futures.clear()
        if self._bucketwise_inflight_futures is not None:
            self._bucketwise_inflight_futures.clear()
        self._bucketwise_cancel_version = 0
        self._bucketwise_worker_error = None

    # ------------------------------------------------------------------ #
    # async state update workers
    # ------------------------------------------------------------------ #
    def _maybe_update_async_state(self, sub_group_id, combined_scale=None):
        if combined_scale is None:
            combined_scale = getattr(self, '_current_combined_scale', self.loss_scale)

        param_group_id = self.sub_group_to_group_id[sub_group_id]
        # The commit runs after step() returns, and the engine steps the LR scheduler right after step(), so capture
        # this step's hyperparameters now instead of letting the worker read the live param group later.
        live_group = self.optimizer.param_groups[param_group_id]
        group_hyperparams = {param_group_id: {key: value for key, value in live_group.items() if key != 'params'}}

        def _state_worker(pin_to_worker_cores=False, half_grad_buffer=None, combined_scale=None):
            if pin_to_worker_cores:
                self._pin_current_thread(self._state_commit_cpu_affinity_mask())

            original = self.optimizer.param_groups[param_group_id]['params']
            try:
                self.optimizer.param_groups[param_group_id]['params'] = \
                    [self.fp32_partitioned_groups_flat[sub_group_id]]
                # Align the buffer to param_groups by index, so a param group other than group 0
                # (e.g. the no-decay group of a decay/no-decay split) is updated with its buffer
                # and every other group is skipped (None).
                buffers = [None] * len(self.optimizer.param_groups)
                buffers[param_group_id] = half_grad_buffer
                numel = self.fp32_partitioned_groups_flat[sub_group_id].numel()
                for start in range(0, numel, _STATE_COMMIT_SLICE_ELEMS):
                    # Re-pick the cores for every slice so the commit widens as soon as backward starts.
                    if pin_to_worker_cores:
                        self._pin_current_thread(self._state_commit_cpu_affinity_mask())
                    slice_numel = min(_STATE_COMMIT_SLICE_ELEMS, numel - start)
                    self.optimizer.step_state_halfgrad(half_grad_buffers=buffers,
                                                       combined_scale=combined_scale,
                                                       element_range=(start, slice_numel),
                                                       group_hyperparams=group_hyperparams)
                # Flush the non-temporal (streaming) optimizer-state stores so the updated state
                # is globally visible before this worker's future completes, matching the fence the
                # foreground param path issues; otherwise a later read can see partial/stale state.
                self.optimizer.memory_fence()
            finally:
                self.optimizer.param_groups[param_group_id]['params'] = original

        # initialize_optimizer_states gives every subgroup a CPU half-precision grad buffer.
        half_grad_buffer = self._cpu_grad_double_buffers[sub_group_id][self._active_grad_buffer_idx]

        executor = getattr(self, '_state_update_executor', None)
        if executor is not None:
            previous_future = self._last_state_update_future

            def chained_worker():
                # Wait for the previous future so workers complete in submission order.
                if previous_future is not None:
                    previous_future.result()
                _state_worker(True, half_grad_buffer, combined_scale)

            try:
                future = executor.submit(chained_worker)
            except Exception as e:
                # submit() can fail (e.g. executor shut down); surface it through the all-reduced
                # bucketwise flag so every rank aborts together at the next _check, not this one alone.
                logger.warning(f"Reflow async state-update submit failed (rank {dist.get_rank()}): {e}")
                self._record_state_update_error(e)
                return
            self._last_state_update_future = future
            self._pending_state_updates_by_subgroup[sub_group_id] = future
        else:
            # Synchronous fallback (no async-state executor). Surface a failure through the same
            # all-reduced flag so every rank aborts together at the next _check, rather than this rank
            # raising alone and hanging its peers at the following collective.
            try:
                _state_worker(False, half_grad_buffer, combined_scale)
            except Exception as e:
                self._record_state_update_error(e)

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
                # The last future chains all previous ones, so waiting for it drains them all.
                future.result()
            except Exception as e:
                if dist.get_rank() == 0:
                    logger.warning(f"Error waiting for state update workers: {e}")
                self._record_state_update_error(e)
            finally:
                self._last_state_update_future = None

    def _wait_for_accum_gradient(self, sub_group_id=None):
        """Wait for pending gradient-accumulation workers (one subgroup, or all)."""
        if sub_group_id is not None:
            for f in self._bucketwise_accumulation_futures.get(sub_group_id, []):
                try:
                    f.result()
                except Exception as e:
                    self._record_bucketwise_worker_error(e)
            self._bucketwise_accumulation_futures[sub_group_id] = []
        else:
            for _sg_id, futures in list(self._bucketwise_accumulation_futures.items()):
                for f in futures:
                    try:
                        f.result()
                    except Exception as e:
                        self._record_bucketwise_worker_error(e)
            self._bucketwise_accumulation_futures.clear()

    def _get_cpu_grad_buffer(self, sub_group_id, pending=False):
        if sub_group_id >= len(self._cpu_grad_double_buffers):
            return self.fp32_partitioned_groups_flat[sub_group_id].grad
        buffers = self._cpu_grad_double_buffers[sub_group_id]
        if buffers is None:
            return self.fp32_partitioned_groups_flat[sub_group_id].grad
        index = self._pending_grad_buffer_idx if pending else self._active_grad_buffer_idx
        return buffers[index]

    def _activate_pending_cpu_grads_if_ready(self):
        if not self._pending_grad_ready:
            return False
        # Gradient buffers are written directly by the bucketwise workers; drain them and flip
        # the active/pending double buffer.
        self._bucketwise_drain_enq_executor()
        self._bucketwise_wait_inflight_updates()
        # Single-buffer mode has only slot 0: keep the active index pinned there (no flip). The
        # workers already wrote the params into fp16_flat, and Phase-2 reads the grads from slot 0.
        if not self._single_grad_buffer:
            self._active_grad_buffer_idx = 1 - self._active_grad_buffer_idx
        self._pending_grad_ready = False
        return True

    # ------------------------------------------------------------------ #
    # cpu_bucketwise pipeline
    # ------------------------------------------------------------------ #
    def _bucketwise_drain_enq_executor(self):
        """Block until all async enqueue work has been reflected in the pending ranges.

        ``_bucketwise_schedule_enqueue_range`` submits enqueue work asynchronously; before
        flushing we must ensure every gradient range has landed in the pending structures,
        otherwise a range is dropped and the parameter update is silently skipped.
        """
        enq_executor = getattr(self, "_bucketwise_enq_executor", None)
        if enq_executor is None:
            return
        try:
            sentinel = enq_executor.submit(lambda: None)
            sentinel.result(timeout=30.0)
        except Exception as e:
            self._record_bucketwise_worker_error(e)

    def _bucketwise_schedule_enqueue_range(self, sub_group_id, dest_offset, num_elements, copy_event=None):
        """Schedule a range enqueue off the main thread (falls back to sync if no executor)."""
        if not self.is_gradient_accumulation_boundary():
            return

        enq_submit = getattr(self, "_bucketwise_enq_submit", None)
        if enq_submit is None:
            self._bucketwise_enqueue_range(sub_group_id, dest_offset, num_elements, copy_event)
            return
        try:
            enq_submit(self._bucketwise_enqueue_range, sub_group_id, dest_offset, num_elements, copy_event)
        except Exception:
            self._bucketwise_enqueue_range(sub_group_id, dest_offset, num_elements, copy_event)

    def _bucketwise_get_valid_partition_element_count(self, sub_group_id: int) -> int:
        """Return the number of real (non-padding) elements this rank must update."""
        partition_element_count = int(self.fp16_partitioned_groups_flat_numel[sub_group_id])
        padding_element_count = 0
        if sub_group_id < len(self.groups_padding):
            padding_element_count = sum(self.groups_padding[sub_group_id])
        valid_partition_element_count = partition_element_count - padding_element_count
        return max(valid_partition_element_count, 0)

    def _bucketwise_clip_range_to_valid_partition(self, sub_group_id: int, dest_offset: int, num_elements: int):
        """Clamp a range to the valid partition; return None if it is entirely padding."""
        valid_partition_element_count = self._bucketwise_get_valid_partition_element_count(sub_group_id)
        return reflow_utils.clip_range_to_valid_partition(valid_partition_element_count, dest_offset, num_elements)

    def _bucketwise_chunk_update_worker(self, work_item):
        # Worker: run the partial Adam update, then async-copy only the updated ranges to GPU.
        try:
            sub_group_id = work_item["sub_group_id"]
            ranges = work_item["ranges"]
            grad_buffer_index = work_item["grad_buffer_index"]
            combined_scale = work_item["combined_scale"]
            copy_events = work_item.get("copy_events", [])
            cancel_version = work_item.get("cancel_version", 0)
            increment_step = work_item.get("increment_step", False)
            target_step = work_item.get("target_step", None)

            if cancel_version != getattr(self, "_bucketwise_cancel_version", 0):
                return None

            # Wait for the GPU->CPU grad copy (only the last event, the stream is sequential).
            last_copy_event = copy_events[-1] if copy_events else None
            if last_copy_event is not None:
                last_copy_event.synchronize()

            if cancel_version != getattr(self, "_bucketwise_cancel_version", 0):
                return None

            # Ensure all gradient accumulations for this subgroup completed.
            self._wait_for_accum_gradient(sub_group_id)

            # Make sure the previous state update for this subgroup finished.
            self._wait_for_pending_state_updates(sub_group_id=sub_group_id)

            fp32_param = self.fp32_partitioned_groups_flat[sub_group_id]
            # The kernel call below uses explicit param/grad/state slices and never reads
            # param_groups, so the worker does not touch self.optimizer.param_groups here:
            # mutating it would be a dead store that races concurrent bucketwise workers,
            # _prepare_state_and_params, and the async _state_worker.

            # grad_buffer_index holds the grads (slot 0 in single-buffer mode).
            half_grad_buffer = self._cpu_grad_double_buffers[sub_group_id][grad_buffer_index]
            if self._single_grad_buffer:
                # Single-buffer mode has no opposite slot, so the kernel writes the params straight into fp16_flat,
                # which that slot would only have duplicated. Grads are still read from slot 0.
                half_params_buffer = self.fp16_partitioned_groups_flat[sub_group_id]
            else:
                # Cross-buffer: FP16 param output goes to the slot opposite the grads.
                fp16_output_buffer_index = 1 - grad_buffer_index
                half_params_buffer = self._cpu_grad_double_buffers[sub_group_id][fp16_output_buffer_index]

            use_lion = hasattr(self.optimizer, "ds_opt_lion")
            if not use_lion and not hasattr(self.optimizer, "ds_opt_adam"):
                raise RuntimeError("cpu_bucketwise requires a Reflow CPU optimizer (ReflowCPUAdam or ReflowCPULion)")

            p = fp32_param
            state = self.optimizer.state[p]

            lr = work_item.get("lr")
            beta1 = work_item.get("beta1")
            beta2 = work_item.get("beta2")
            eps = work_item.get("eps")
            weight_decay = work_item.get("weight_decay")
            bias_correction = work_item.get("bias_correction", True)
            maximize = work_item.get("maximize", False)
            # increment_step controls the per-range skip flag only; the Python state['step']
            # is committed once at the end of step().
            if target_step is None:
                target_step = int(state.get('step', 0)) + 1

            # Lion has no second moment, epsilon, or bias correction.
            if use_lion:
                for range_idx, (offset, size) in enumerate(ranges):
                    range_increment_step = increment_step and (range_idx == 0)

                    param_slice = p.data.view(-1).narrow(0, offset, size)
                    grad_slice = half_grad_buffer.view(-1).narrow(0, offset, size)
                    exp_avg_slice = state['exp_avg'].view(-1).narrow(0, offset, size)
                    half_params_slice = half_params_buffer.view(-1).narrow(0, offset, size)

                    self.optimizer.ds_opt_lion.reflow_lion_update_params_halfgrad(
                        self.optimizer.opt_id,
                        target_step,
                        lr,
                        beta1,
                        beta2,
                        weight_decay,
                        param_slice,
                        grad_slice,
                        exp_avg_slice,
                        half_params_slice,
                        combined_scale,
                        not range_increment_step,
                    )

                self.optimizer.ds_opt_lion.reflow_lion_memory_fence()
            else:
                for range_idx, (offset, size) in enumerate(ranges):
                    range_increment_step = increment_step and (range_idx == 0)

                    param_slice = p.data.view(-1).narrow(0, offset, size)
                    grad_slice = half_grad_buffer.view(-1).narrow(0, offset, size)
                    exp_avg_slice = state['exp_avg'].view(-1).narrow(0, offset, size)
                    exp_avg_sq_slice = state['exp_avg_sq'].view(-1).narrow(0, offset, size)
                    half_params_slice = half_params_buffer.view(-1).narrow(0, offset, size)

                    self.optimizer.ds_opt_adam.reflow_adam_update_params_halfgrad(
                        self.optimizer.opt_id,
                        target_step,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        bias_correction,
                        param_slice,
                        grad_slice,
                        exp_avg_slice,
                        exp_avg_sq_slice,
                        half_params_slice,
                        combined_scale,
                        not range_increment_step,
                        maximize,
                    )

                self.optimizer.ds_opt_adam.reflow_adam_memory_fence()

            if cancel_version != getattr(self, "_bucketwise_cancel_version", 0):
                return None

            if self._single_grad_buffer:
                # Single-buffer mode already wrote the params into fp16_flat. No H2D event is needed: they are
                # CPU-resident, and step() joins this worker before the next forward reads them.
                return None

            fp16_flat = self.fp16_partitioned_groups_flat[sub_group_id]
            if fp16_flat is None:
                return None

            src = half_params_buffer.data.view(-1)
            dst = fp16_flat.data.view(-1)

            merged_ranges = reflow_utils.merge_contiguous_ranges(ranges)
            stream = getattr(self, "_bucketwise_h2d_stream", None)
            if stream is None:
                stream = get_accelerator().current_stream()

            with get_accelerator().stream(stream):
                for off, sz in merged_ranges:
                    dst.narrow(0, off, sz).copy_(src.narrow(0, off, sz), non_blocking=True)
                if not get_accelerator().is_synchronized_device():
                    h2d_event = get_accelerator().Event()
                    h2d_event.record(stream)
                else:
                    h2d_event = None

            return h2d_event
        except Exception as e:
            # A cooperative cancel (version bump) can race the kernel and surface as a transient
            # error; if this work item has been cancelled, treat it as a clean no-op like the
            # explicit cancel returns above. Otherwise this is a real kernel/OOM failure: record it
            # so step() can abort deterministically on all ranks instead of silently advancing.
            if work_item.get("cancel_version", 0) != getattr(self, "_bucketwise_cancel_version", 0):
                return None
            self._record_bucketwise_worker_error(e)
            logger.warning(f"Reflow bucketwise worker error (rank {dist.get_rank()}): {e}")
            return None

    def _bucketwise_enqueue_range(self, sub_group_id, dest_offset, num_elements, copy_event=None):
        # Accumulate a scheduled grad range; flush once the shard threshold is reached.
        try:
            clipped_range = self._bucketwise_clip_range_to_valid_partition(sub_group_id, dest_offset, num_elements)
            if clipped_range is None:
                return
            dest_offset, num_elements = clipped_range

            lock = self._get_enqueue_lock()

            with lock:
                if sub_group_id >= len(self._bucketwise_pending_ranges_by_group):
                    while len(self._bucketwise_pending_ranges_by_group) <= sub_group_id:
                        self._bucketwise_pending_ranges_by_group.append([])
                        self._bucketwise_pending_elems_by_group.append(0)

                if copy_event is not None:
                    self._bucketwise_pending_events.setdefault(sub_group_id, []).append(copy_event)

                self._bucketwise_pending_ranges_by_group[sub_group_id].append((dest_offset, num_elements))
                self._bucketwise_pending_elems_by_group[sub_group_id] += num_elements
                pending_elem_count = self._bucketwise_pending_elems_by_group[sub_group_id]

                if pending_elem_count >= self._bucketwise_element_shard_size:
                    self._bucketwise_flush_group(sub_group_id,
                                                 force=True,
                                                 use_pending_buffer=True,
                                                 combined_scale=self.loss_scale,
                                                 already_locked=True)
        except Exception as e:
            self._record_bucketwise_worker_error(e)

    def _bucketwise_flush_group(self,
                                sub_group_id,
                                force=False,
                                use_pending_buffer=True,
                                combined_scale=None,
                                already_locked=False):
        # Submit accumulated ranges as a single work item. Returns the future (or None).
        if self._bucketwise_update_submit is None:
            return None
        if sub_group_id >= len(self._bucketwise_pending_ranges_by_group):
            return None
        ranges = self._bucketwise_pending_ranges_by_group[sub_group_id]
        if not ranges:
            return None
        if not force and self._bucketwise_pending_elems_by_group[sub_group_id] < self._bucketwise_element_shard_size:
            return None

        merged_ranges = reflow_utils.merge_contiguous_ranges(ranges)

        # force=False: submit merged ranges >= min_submit_range_elems, keep the rest pending.
        min_submit_range_elems = getattr(self, "_bucketwise_min_submit_range_elems",
                                         _DEFAULT_BUCKETWISE_MIN_SUBMIT_ELEMS)
        if not force:
            submit_ranges, pending_merged, pending_elems = \
                reflow_utils.split_ranges_by_min_size(merged_ranges, min_submit_range_elems)
            self._bucketwise_pending_ranges_by_group[sub_group_id] = pending_merged
            self._bucketwise_pending_elems_by_group[sub_group_id] = pending_elems
            if not submit_ranges:
                return None
        else:
            submit_ranges = merged_ranges
            self._bucketwise_pending_ranges_by_group[sub_group_id].clear()
            self._bucketwise_pending_elems_by_group[sub_group_id] = 0

        # Final clamp/filter so submitted ranges never exceed the valid partition.
        valid_partition_element_count = self._bucketwise_get_valid_partition_element_count(sub_group_id)
        if valid_partition_element_count <= 0:
            return None
        submit_ranges = reflow_utils.clip_ranges_to_valid_partition(submit_ranges, valid_partition_element_count)
        if not submit_ranges:
            return None

        grad_buffer_index = self._pending_grad_buffer_idx if use_pending_buffer else self._active_grad_buffer_idx

        copy_events_snapshot = self._bucketwise_pending_events.get(sub_group_id, [])
        self._bucketwise_pending_events[sub_group_id] = []

        lock = self._get_enqueue_lock()

        def _prepare_state_and_params():
            increment_step = (sub_group_id not in self._bucketwise_group_step_incremented)
            if increment_step:
                self._bucketwise_group_step_incremented.add(sub_group_id)

            param_group_id = self.sub_group_to_group_id[sub_group_id]
            group = self.optimizer.param_groups[param_group_id]
            # The previous step's async state worker may still be iterating group['params'].
            # Keep its list intact while preparing the next bucket update.
            p = self.fp32_partitioned_groups_flat[sub_group_id]
            if p is None:
                return None

            # Reflow fp32 master params live on CPU, so _ensure_param_state (device=cpu) builds
            # the same lazy step/exp_avg/exp_avg_sq buffers the inline init did.
            state = self.optimizer._ensure_param_state(p)
            # Pass a fixed per-iteration target_step to the kernel; commit state['step'] once
            # at the end of step() instead of mutating it in the worker.
            current_step = int(state.get('step', 0))
            target_step = self._bucketwise_group_target_step.get(sub_group_id, None)
            if target_step is None:
                target_step = current_step + 1
                self._bucketwise_group_target_step[sub_group_id] = target_step

            beta1, beta2 = group['betas']
            lr = group['lr']
            # Lion param groups carry no 'eps'/'bias_correction'; the optimizer ignores both.
            eps = group.get('eps', 0.0)
            weight_decay = group['weight_decay']
            bias_correction = group.get('bias_correction', True)
            maximize = group.get('maximize', False)

            return increment_step, target_step, lr, beta1, beta2, eps, weight_decay, bias_correction, maximize

        if already_locked:
            result = _prepare_state_and_params()
            if result is None:
                return None
            increment_step, target_step, lr, beta1, beta2, eps, weight_decay, bias_correction, maximize = result
        else:
            with lock:
                result = _prepare_state_and_params()
                if result is None:
                    return None
                increment_step, target_step, lr, beta1, beta2, eps, weight_decay, bias_correction, maximize = result

        work_item = {
            "sub_group_id": sub_group_id,
            "ranges": submit_ranges,
            "grad_buffer_index": grad_buffer_index,
            "combined_scale": combined_scale,
            "copy_events": copy_events_snapshot,
            "cancel_version": getattr(self, "_bucketwise_cancel_version", 0),
            "increment_step": increment_step,
            "target_step": target_step,
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
            "bias_correction": bias_correction,
            "maximize": maximize,
        }

        future = self._bucketwise_update_submit(self._bucketwise_chunk_update_worker, work_item)
        if getattr(self, "_bucketwise_update_futures", None) is not None:
            self._bucketwise_update_futures.append(future)
        self._bucketwise_inflight_futures.append(future)
        return future

    def _bucketwise_flush_all_pending(self, force=False, use_pending_buffer=True, combined_scale=None):
        for sub_group_id in range(len(self._bucketwise_pending_ranges_by_group)):
            self._bucketwise_flush_group(sub_group_id,
                                         force=force,
                                         use_pending_buffer=use_pending_buffer,
                                         combined_scale=combined_scale)

    def _bucketwise_cancel_inflight_updates(self):
        # Best-effort cancel of inflight workers when clipping is needed.
        self._bucketwise_cancel_version += 1
        for f in list(getattr(self, "_bucketwise_inflight_futures", [])):
            try:
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

    def _bucketwise_wait_inflight_updates(self):
        # Wait for submitted bucketwise work (optimizer + H2D submit). H2D completion itself is
        # synchronized later via CUDA events.

        futures = list(getattr(self, "_bucketwise_update_futures", []) or [])
        if not futures:
            futures = list(getattr(self, "_bucketwise_inflight_futures", []) or [])

        self._drain_update_futures_collecting_h2d_events(futures)

        if getattr(self, "_bucketwise_update_futures", None) is not None:
            self._bucketwise_update_futures.clear()
        self._bucketwise_inflight_futures.clear()

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
                                                      force=True,
                                                      use_pending_buffer=False,
                                                      combined_scale=combined_scale)
                if future is not None:
                    # Join the param-update worker AND collect its H2D event, so the next forward's
                    # all-gather waits for this subgroup's updated GPU param (written on the h2d
                    # stream) -- otherwise the forward races the in-flight copy and reads stale params.
                    self._drain_update_futures_collecting_h2d_events([future])
                # Commit the step counter into optimizer.state BEFORE the state update. step_state_halfgrad
                # reads state['step'] for bias correction and assumes the param update already advanced it
                # (in CPU mode the commit precedes the async-state loop). Without committing first,
                # state['step']=0 -> bias_correction 1-beta**0 = 0 -> divide-by-zero -> NaN weights.
                self._bucketwise_commit_group_steps()
                # Optimizer-state (momentum) update; then join both the param-update worker and the
                # state worker before swapping the now-final state out to nvme.
                self._maybe_update_async_state(sub_group_id, combined_scale)
                self._wait_for_pending_state_updates(sub_group_id=sub_group_id)
            self._optimizer_states_and_gradient_swap_out(sub_group_id, timer_names)

    def _bucketwise_run_full_partition_updates(self, combined_scale):
        # Clipping path: re-shard and re-run each group's full range, then transfer.

        full_update_futures = []
        for sub_group_id in range(len(self.fp16_groups)):
            if self._swappable_optimizer_subgroup(sub_group_id):
                continue  # nvme optimizer subgroups are handled in _nvme_optimizer_step_subgroups
            full_ranges = []
            for param in self.fp16_groups[sub_group_id]:
                if not param.requires_grad:
                    continue
                param_id = self.get_param_id(param)
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
                    future = self._bucketwise_flush_group(sub_group_id,
                                                          force=True,
                                                          use_pending_buffer=False,
                                                          combined_scale=combined_scale)
                    if future is not None:
                        full_update_futures.append(future)
                    shard_ranges = []
                    shard_elems = 0

            if shard_ranges:
                self._bucketwise_pending_ranges_by_group[sub_group_id] = shard_ranges
                self._bucketwise_pending_elems_by_group[sub_group_id] = shard_elems
                future = self._bucketwise_flush_group(sub_group_id,
                                                      force=True,
                                                      use_pending_buffer=False,
                                                      combined_scale=combined_scale)
                if future is not None:
                    full_update_futures.append(future)

        self._drain_update_futures_collecting_h2d_events(full_update_futures)

    def backward_prologue(self):
        # The pending state commit no longer overlaps a forward, so let it use every worker core.
        self._state_commit_overlaps_forward = False
        super().backward_prologue()

    def _state_commit_cpu_affinity_mask(self):
        # A few busy cores keep the forward's core at a high turbo frequency; after the forward the commit gets every
        # worker core so a large model's commit still finishes before the next step() waits on it.
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

        local_sq = []
        for group in self.fp16_groups:
            group_sq = 0.0
            for p in group:
                if is_model_parallel_parameter(p) or (self.model_parallel_rank == 0):
                    param_id = self.get_param_id(p)
                    if param_id in self.norm_for_param_grads:
                        group_sq += self.norm_for_param_grads[param_id]**2
            local_sq.append(float(group_sq))

        total_norms = get_accelerator().FloatTensor(local_sq)
        # Match complete_grad_norm_calculation_for_cpu_offload: each subgroup sums over its own partition group,
        # then the model-parallel group, then (AutoEP expert subgroups only) the expert-parallel group. Subgroups
        # sharing a group are reduced together, so the usual single-group case stays one all-reduce.
        partition_groups = [self._get_sub_group_process_group(i) for i in range(len(self.fp16_groups))]
        self._all_reduce_norm_entries_by_group(total_norms, partition_groups)
        self._model_parallel_all_reduce(tensor=total_norms, op=dist.ReduceOp.SUM)
        expert_groups = [self._autoep_expert_parallel_group(group) for group in self.fp16_groups]
        self._all_reduce_norm_entries_by_group(total_norms, expert_groups)

        norm_groups = []
        for i in range(len(self.fp16_groups)):
            total_norm = total_norms[i]**0.5
            mask_nan_or_inf_with_val_inplace(total_norm, device=total_norm.device)
            norm_groups.append(total_norm.cpu())
        return norm_groups

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
        # This setup only touches param handles and the FP32 master, so run it before the device sync to overlap
        # the GPU drain from backward. Everything below that reads GPU results still follows the sync.
        self._pre_step()
        self._partition_all_parameters()
        get_accelerator().synchronize()
        bucketwise_drained = self._activate_pending_cpu_grads_if_ready()
        # Activation already joined the backward workers when it flipped the buffers. Otherwise
        # join them before overflow cleanup discards their handles.
        if not bucketwise_drained:
            self._bucketwise_drain_enq_executor()
            self._bucketwise_wait_inflight_updates()
        # Workers swallow their own exceptions (a cancel and a real error both return None), so
        # surface any real kernel/OOM failure here and abort deterministically on all ranks before
        # the overflow check advances the loss scale / optimizer state.
        self._check_bucketwise_worker_error()

        if self._overflow_check_and_loss_scale_update():
            if self.swap_optimizer:
                self.optimizer_swapper.log_timers()
            # Discard the step like the base optimizer. The workers already wrote new FP16/BF16 params to the GPU
            # during backward, so restore them from the untouched FP32 master.
            self._bucketwise_cancel_version += 1
            if not get_accelerator().is_synchronized_device():
                get_accelerator().synchronize()
            for sub_group_id in range(len(self.fp16_groups)):
                self._restore_fp16_params_from_master(sub_group_id)
            return

        norm_groups = self._get_norm_groups()
        scaled_global_grad_norm = torch.linalg.vector_norm(torch.stack(norm_groups))

        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale

        combined_scale = self.loss_scale
        needs_clipping = False
        if self.clip_grad > 0.:
            # norm is in fact norm*scale, so unscale.
            clip = ((scaled_global_grad_norm / self.loss_scale) + 1e-6) / self.clip_grad
            # Re-run the update exactly when the clip factor changes the scale, like the base optimizer. A plain
            # norm > clip_grad test misses norms just below the threshold, which the base still clips, and would
            # commit the clipped state under params that were computed unclipped during backward.
            needs_clipping = clip > 1.0
            clip = max(clip, 1.0)
            combined_scale = clip * self.loss_scale
        self._current_combined_scale = combined_scale

        self._bucketwise_flush_all_pending(force=True, use_pending_buffer=False, combined_scale=self.loss_scale)
        if needs_clipping:
            # cancel() does not wait. Join the cancelled workers first so their unclipped H2D writes are enqueued
            # before the clipped re-run's and cannot land last.
            self._bucketwise_cancel_inflight_updates()
            self._bucketwise_wait_inflight_updates()
            self._bucketwise_run_full_partition_updates(combined_scale=combined_scale)

        timer_names = set()
        # Param updates already ran bucketwise during backward; here we only commit per-group
        # step counters and submit the deferred async state updates.
        self._bucketwise_wait_inflight_updates()
        # NVMe optimizer subgroups deferred their CPU-Adam (state not resident during backward); run
        # it now with the state swapped in around it, before committing step counters / the async loop.
        if self.swap_optimizer:
            self._nvme_optimizer_step_subgroups(combined_scale, timer_names)
        self._bucketwise_commit_group_steps()
        # The commits submitted below overlap the next forward until its backward starts.
        self._state_commit_overlaps_forward = True
        for sub_group_id in reversed(range(len(self.fp16_groups))):
            if self._swappable_optimizer_subgroup(sub_group_id):
                continue  # nvme: already updated above with its state swapped in
            self._maybe_update_async_state(sub_group_id, combined_scale)
        if self._bucketwise_inflight_h2d_events:
            # Instead of blocking on the workers' H2D copies, make every stream that reads the params wait on them.
            # With overlap_comm the all-gather has its own stream, so a compute-stream wait alone is not enough.
            h2d_events = self._coalesce_bucketwise_h2d_events()
            compute_stream = get_accelerator().current_stream()
            for h2d_event in h2d_events:
                if h2d_event is not None:
                    if self._sync_bucketwise_h2d_before_forward:
                        h2d_event.synchronize()
                    compute_stream.wait_event(h2d_event)
                    self.copy_grad_stream.wait_event(h2d_event)
            self._get_param_coordinator().wait_for_external_allgather_dependencies(h2d_events)
        # NVMe param offload: the bucketwise workers leave the updated FP16/BF16 weights in the CPU
        # half buffer (an nvme subgroup has no GPU param to H2D into), so swap them out to nvme here,
        # on the main thread -- the partitioned param swapper holds single shared mutable state and is
        # not thread-safe. Confined to nvme subgroups (fp16 flat is None); cpu/gpu paths are untouched.
        for sub_group_id in range(len(self.fp16_groups)):
            if self.fp16_partitioned_groups_flat[sub_group_id] is None:
                self._partitioned_params_swap_out(sub_group_id, use_updated_weights=True)

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
        fp16_flat.data.copy_(fp32_flat.data)
        self._unflatten_partitioned_parameters(sub_group_id)

    @instrument_w_nvtx
    def has_overflow(self, partition_gradients=True):
        if partition_gradients:
            with get_accelerator().stream(self.reduce_and_partition_stream):
                if self.grad_partitions_flat_buffer is None:
                    # No flat grad buffer is allocated in bucketwise mode, so there is nothing to
                    # scan for inf/nan here.
                    pass
                elif hasattr(self.inf_or_nan_tracker, "logical_or_"):
                    self.inf_or_nan_tracker.logical_or_(torch.isinf(self.grad_partitions_flat_buffer).any())
                    self.inf_or_nan_tracker.logical_or_(torch.isnan(self.grad_partitions_flat_buffer).any())
                else:
                    # logical_or_ not available in older versions of pytorch
                    self.inf_or_nan_tracker += torch.isinf(self.grad_partitions_flat_buffer).any()
                    self.inf_or_nan_tracker += torch.isnan(self.grad_partitions_flat_buffer).any()
                    self.inf_or_nan_tracker = self.inf_or_nan_tracker > 0

                overflow_gpu = self.inf_or_nan_tracker.clone().to(get_accelerator().current_device_name()).to(
                    torch.uint8)
                self.inf_or_nan_tracker.zero_()
                # Fold in the inf/nan seen while offloading grads in partition_grads (this path
                # freed the flat buffer the branch above would otherwise scan).
                if self._reflow_grad_overflow is not None:
                    overflow_gpu.logical_or_(self._reflow_grad_overflow.to(overflow_gpu.device))
                    self._reflow_grad_overflow.zero_()

            if not get_accelerator().resolves_data_dependency():
                get_accelerator().default_stream().wait_stream(self.reduce_and_partition_stream)
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.dp_process_group)

        else:
            params = []
            for group in self.fp16_groups:
                for param in group:
                    params.append(param)

            overflow = self.has_overflow_serial(params, is_grad_list=partition_gradients)
            overflow_gpu = get_accelerator().ByteTensor([overflow])
        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        self._model_parallel_all_reduce(tensor=overflow_gpu, op=dist.ReduceOp.MAX)

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # ------------------------------------------------------------------ #
    # auxiliary APIs unsupported under reflow
    # ------------------------------------------------------------------ #
    # Reflow frees the flat grad buffer and leaves the FP32 .grad unset, so these APIs would fail with an
    # opaque AttributeError; raise a clear NotImplementedError instead.

    def get_fp32_grad_for_param(self, param) -> Tensor:
        raise NotImplementedError(_REFLOW_FLAT_GRAD_FREED_MSG.format(api="get_fp32_grad_for_param"))

    def set_fp32_grad_for_param(self, value, param):
        raise NotImplementedError(_REFLOW_FLAT_GRAD_FREED_MSG.format(api="set_fp32_grad_for_param"))

    def offload_states(self,
                       include: Container[OffloadStateTypeEnum] = None,
                       device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
                       pin_memory: bool = True,
                       non_blocking: bool = False):
        raise NotImplementedError(_REFLOW_FLAT_GRAD_FREED_MSG.format(api="offload_states"))

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

    def _get_fp32_opt_state_partition(self, param, release_swap_buffers, optim_state_key=None):
        # Every get/set hp-param API goes through here.
        self._wait_for_pending_state_updates()
        return super()._get_fp32_opt_state_partition(param, release_swap_buffers, optim_state_key=optim_state_key)
