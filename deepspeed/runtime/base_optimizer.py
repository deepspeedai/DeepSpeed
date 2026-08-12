# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
from functools import wraps
from typing import Any
from weakref import ref

from deepspeed.utils import logger
from deepspeed.utils.tensor_fragment import map_to_flat_opt_states
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank, see_memory_usage
from deepspeed.runtime.torch_autocast import get_comm_dtype, is_autocast_initialized
from deepspeed.runtime.utils import maybe_loss_for_backward


class DeepSpeedOptimizer(object):
    pass


class _IPGBucketParameterList(list):
    """Record when a non-empty IPG bucket is drained during an active backward."""

    def __init__(self, values, optimizer):
        super().__init__(values)
        self._optimizer_ref = ref(optimizer)

    def clear(self):
        optimizer = self._optimizer_ref()
        if (self and optimizer is not None and optimizer._backward_active_depth > 0
                and not optimizer._clearing_aborted_backward):
            optimizer._backward_reduction_observed = True
        super().clear()


def _get_universal_checkpoint_ep_info() -> tuple[int, int]:
    # Universal checkpoints use EP slicing only when an expert group exists.
    try:
        from deepspeed.utils import groups
        expert_groups = groups._get_expert_parallel_group_dict()
        if not expert_groups:
            return 0, 1
        max_ep_name = groups._get_max_expert_size_name()
        if max_ep_name not in expert_groups:
            return 0, 1
        return groups._get_expert_parallel_rank(max_ep_name), groups._get_expert_parallel_world_size(max_ep_name)
    except (RuntimeError, AttributeError, KeyError):
        return 0, 1


class BackwardHookStateManager:
    """Manages backward pass state for ZeRO optimizers.

    This class handles the complex state management needed for gradient accumulation hooks
    to work correctly with:

    1. **Reentrant Gradient Checkpointing** (use_reentrant=True):
       With reentrant checkpointing, gradient hooks fire in multiple phases within a
       single backward() call. For example, with model: linear1 (checkpointed) -> linear2:
         - Phase 1: Hooks for linear2 fire (non-checkpointed params)
         - Checkpoint recomputes linear1's forward
         - Phase 2: Hooks for linear1 fire (checkpointed params)

       The challenge is that `count_used_parameters_in_backward()` only sees params
       currently in the backward graph. During Phase 1, it returns 2 (linear2's params),
       but after checkpoint recomputation, it returns 4 (all params). We must NOT run
       the epilogue prematurely after Phase 1.

       Solution: Queue a post-backward callback on the autograd engine at the start of
       backward and run the epilogue when the graph task completes. This avoids premature
       epilogues across reentrant phases. The `_max_expected_hooks_seen` counter remains
       as a fallback when the callback API is unavailable.

    2. **TiledFusedLogitsLoss and Similar Custom Autograd Functions**:
       Some custom autograd functions call `torch.autograd.backward()` from their
       forward pass BEFORE the user calls `engine.backward(loss)`. These internal
       backward calls trigger ZeRO's gradient hooks, but we must NOT run the epilogue
       until the user's actual backward pass.

       Solution: Track `_backward_active_depth` which is only incremented when
       `enter_backward()` is called (from engine.backward or user code). Hooks check
       this depth before running the epilogue.

    3. **Multiple Backward Phases with Exit/Re-entry**:
       When the epilogue runs after Phase 1 (with reentrant checkpointing), it calls
       `exit_backward()`, setting `_backward_active_depth` to 0. When Phase 2's hooks
       fire, we need to re-enter the backward context.

       Solution: `_backward_seen_this_step` flag tracks if backward was ever active
       this step. Combined with `_backward_active_depth == 0`, this detects Phase 2
       and calls `enter_backward()` again.

    Attributes:
        remaining_grad_acc_hooks: Count of hooks remaining before epilogue should run
        backward_active_depth: Nesting depth of backward() calls (0 = not in backward)
        backward_seen_this_step: True if enter_backward() was called this step
        epilogue_ran_this_backward: True if epilogue ran (for micro_step_id management)
        hooks_fired_this_backward: Count of gradient hooks that have fired
        max_expected_hooks_seen: Maximum expected hook count seen (grows with reentrant)
        post_backward_callback_queued: True if a post-backward callback is queued
        post_backward_callback_graph_task_id: Graph task id for the queued callback
    """

    def __init__(self):
        self.remaining_grad_acc_hooks = 0
        self._grad_acc_post_hooks = []
        self.backward_active_depth = 0
        self.backward_seen_this_step = False
        self.epilogue_ran_this_backward = False
        self.hooks_fired_this_backward = 0
        self.max_expected_hooks_seen = 0
        self.post_backward_callback_queued = False
        self.post_backward_callback_graph_task_id = None

    def register_grad_acc_post_hook(self, hook):
        """Register a callback to run when all gradient hooks have fired."""
        self._grad_acc_post_hooks.append(hook)

    def unregister_grad_acc_post_hooks(self):
        """Remove all registered gradient accumulation post hooks."""
        self._grad_acc_post_hooks = []

    def run_grad_acc_post_hooks(self):
        """Run all registered post hooks if backward is active.

        Custom autograd Functions (e.g., TiledFusedLogitsLoss) can invoke
        `torch.autograd.backward()` from their *forward* pass before the user
        ever calls `engine.backward(loss)`. Those early backward calls still
        trigger ZeRO's grad hooks, but we must not run the engine's
        post-backward logic (which reduces/clears grads) until the outer/user
        backward is active. The depth guard filters out only those pre-user
        invocations while still allowing backward calls that happen during
        the real user backward.
        """
        if self.backward_active_depth == 0:
            return
        for hook in self._grad_acc_post_hooks:
            hook()

    def enter_backward(self):
        """Enter backward context. Call at the start of backward pass."""
        # On first real backward entry of a step, reset counters that may have been
        # polluted by pre-user-backward hooks (e.g. TiledFusedLogitsLoss calling
        # torch.autograd.backward() from forward). Do NOT reset on reentrant
        # phase re-entry (backward_seen_this_step == True) so phase-to-phase
        # state remains intact.
        if self.backward_active_depth == 0 and not self.backward_seen_this_step:
            self.hooks_fired_this_backward = 0
            self.max_expected_hooks_seen = 0
            self.remaining_grad_acc_hooks = 0
            self.post_backward_callback_queued = False
            self.post_backward_callback_graph_task_id = None
        self.backward_active_depth += 1
        # Track that backward has been active at some point in this step.
        # This is used to detect subsequent gradient hook phases with reentrant checkpointing.
        self.backward_seen_this_step = True

    def exit_backward(self):
        """Exit backward context. Call at the end of backward pass."""
        if self.backward_active_depth > 0:
            self.backward_active_depth -= 1

    def reset_for_new_step(self):
        """Reset state at the start of each forward/backward step.

        Returns whether the previous backward was still active. This can happen
        when autograd exits through an exception before the backward epilogue.
        """
        incomplete_backward = self.backward_active_depth > 0
        self.remaining_grad_acc_hooks = 0
        self.backward_active_depth = 0
        self.backward_seen_this_step = False
        self.hooks_fired_this_backward = 0
        self.max_expected_hooks_seen = 0
        self.epilogue_ran_this_backward = False
        self.post_backward_callback_queued = False
        self.post_backward_callback_graph_task_id = None
        return incomplete_backward

    def should_refresh_expected_hook_count(self):
        """Return True when count_used_parameters_in_backward() should be re-evaluated.

        Refresh is needed in two cases:
        1. First hook of a backward (or backward phase): hooks_fired == 0.
        2. A new reentrant phase started: remaining hooks exhausted, we exited
           backward, but backward was active earlier this step.

        The predicate must be evaluated BEFORE reenter_backward_if_needed()
        because re-entering changes backward_active_depth and hides the
        phase-boundary signal.
        """
        return (self.hooks_fired_this_backward == 0
                or (self.remaining_grad_acc_hooks == 0 and self.backward_active_depth == 0
                    and self.backward_seen_this_step))

    def reenter_backward_if_needed(self):
        """Re-enter backward context for subsequent phases in reentrant checkpointing.

        With reentrant gradient checkpointing, gradient hooks can fire in multiple phases
        within a single backward call. When the epilogue runs after a phase, it calls
        exit_backward(), setting backward_active_depth to 0. When the next phase starts,
        we need to re-enter backward.

        We detect subsequent phases by checking:
        1. remaining_grad_acc_hooks == 0 (epilogue ran or new backward)
        2. backward_active_depth == 0 (we've exited from previous phase)
        3. backward_seen_this_step == True (backward was active earlier)

        This distinguishes from TiledFusedLogitsLoss which calls backward() during forward -
        in that case backward_seen_this_step is False because enter_backward() was never called.
        """
        if self.remaining_grad_acc_hooks == 0:
            if self.backward_active_depth == 0 and self.backward_seen_this_step:
                self.enter_backward()

    def queue_post_backward_callback(self):
        """Queue post-backward hooks to run after the current graph finishes."""
        if self.post_backward_callback_queued:
            return True
        if self.backward_active_depth == 0:
            return False

        engine = getattr(torch.autograd.Variable, "_execution_engine", None)
        if engine is None or not hasattr(engine, "queue_callback"):
            return False
        if not hasattr(torch._C, "_current_graph_task_id"):
            return False

        graph_task_id = torch._C._current_graph_task_id()
        if graph_task_id == -1:
            return False

        def _run_post_backward():
            self.run_grad_acc_post_hooks()

        engine.queue_callback(_run_post_backward)
        self.post_backward_callback_queued = True
        self.post_backward_callback_graph_task_id = graph_task_id
        return True

    def update_hook_state_and_maybe_run_epilogue(self, current_expected_count):
        """Update hook state after a gradient hook fires and run epilogue if all hooks have fired.

        With reentrant gradient checkpointing, count_used_parameters_in_backward() returns the
        count of params that will execute in the current backward graph. This count grows as
        checkpointed regions are recomputed. We track the MAXIMUM count seen to ensure we don't
        run the epilogue until all params that will ever participate have been processed.
        Counters are reset at forward() time via reset_for_new_step().

        Args:
            current_expected_count: The current expected number of hooks, from
                                   count_used_parameters_in_backward() plus any leaf modules.
        """
        self.hooks_fired_this_backward += 1
        self.max_expected_hooks_seen = max(self.max_expected_hooks_seen, current_expected_count)

        # Prefer running post-backward hooks via autograd engine callback when available.
        # This avoids premature epilogues with reentrant checkpointing.
        if self.queue_post_backward_callback():
            self.remaining_grad_acc_hooks = max(self.max_expected_hooks_seen - self.hooks_fired_this_backward, 0)
            return

        # Fallback: Run epilogue only when we've processed ALL params that will participate.
        # This is the maximum count we've seen (accounts for late-joining params
        # from reentrant checkpointing) and also excludes unused params.
        if self.hooks_fired_this_backward >= self.max_expected_hooks_seen:
            self.remaining_grad_acc_hooks = 0
            self.run_grad_acc_post_hooks()
        else:
            self.remaining_grad_acc_hooks = self.max_expected_hooks_seen - self.hooks_fired_this_backward


class ZeROOptimizer(DeepSpeedOptimizer):
    """Base class for ZeRO optimizer implementations (stages 1, 2, and 3)."""

    def __init_subclass__(cls, **kwargs):
        """Require every ZeRO step implementation to reject an incomplete backward."""
        super().__init_subclass__(**kwargs)
        step = cls.__dict__.get("step")
        if step is None or getattr(step, "_deepspeed_aborted_backward_guard", False):
            return

        @wraps(step)
        def guarded_step(self, *args, **kwargs):
            self._ensure_backward_complete_before_step()
            result = step(self, *args, **kwargs)
            self._backward_completed_since_step = False
            return result

        guarded_step._deepspeed_aborted_backward_guard = True
        cls.step = guarded_step

    def __setattr__(self, name, value):
        """Preserve the aborted-backward guard across runtime step replacement."""
        if (name == "step" and callable(value) and "_backward_hook_state" in self.__dict__
                and not getattr(value, "_deepspeed_aborted_backward_guard", False)):
            step = value

            @wraps(step)
            def guarded_step(*args, **kwargs):
                self._ensure_backward_complete_before_step()
                result = step(*args, **kwargs)
                self._backward_completed_since_step = False
                return result

            guarded_step._deepspeed_aborted_backward_guard = True
            value = guarded_step
        object.__setattr__(self, name, value)

    def __init__(self):
        self._backward_hook_state = BackwardHookStateManager()
        self._aborted_backward_requires_restart = False
        self._backward_completed_since_step = False
        self._backward_reduction_observed = False
        self._clearing_aborted_backward = False
        # Mirrored copy of the engine GAS boundary for managed reduce/offload paths.
        # Engine owns the source of truth (micro-step / step() / set_*); ZeRO reads this
        # during backward. Prefer get/set methods over touching the private field.
        self._is_gradient_accumulation_boundary = True

    def is_gradient_accumulation_boundary(self) -> bool:
        """Whether the current micro-batch is a gradient accumulation boundary.

        Used by managed ZeRO reduce/partition/offload logic. Unmanaged mode still
        mirrors True while ``engine.step()`` runs so late readers stay consistent;
        deferred offload finalize does not branch on this flag.
        """
        return self._is_gradient_accumulation_boundary

    def set_gradient_accumulation_boundary(self, is_boundary: bool) -> None:
        """Mirror the engine's gradient accumulation boundary into this optimizer."""
        self._is_gradient_accumulation_boundary = bool(is_boundary)

    # Delegate backward hook state management to the manager.
    # These properties provide backward compatibility with code that accesses
    # these attributes directly (e.g., in stage3.py and stage_1_and_2.py).
    @property
    def _remaining_grad_acc_hooks(self):
        return self._backward_hook_state.remaining_grad_acc_hooks

    @_remaining_grad_acc_hooks.setter
    def _remaining_grad_acc_hooks(self, value):
        self._backward_hook_state.remaining_grad_acc_hooks = value

    @property
    def _backward_active_depth(self):
        return self._backward_hook_state.backward_active_depth

    @_backward_active_depth.setter
    def _backward_active_depth(self, value):
        self._backward_hook_state.backward_active_depth = value

    @property
    def _backward_seen_this_step(self):
        return self._backward_hook_state.backward_seen_this_step

    @_backward_seen_this_step.setter
    def _backward_seen_this_step(self, value):
        self._backward_hook_state.backward_seen_this_step = value

    @property
    def _epilogue_ran_this_backward(self):
        return self._backward_hook_state.epilogue_ran_this_backward

    @_epilogue_ran_this_backward.setter
    def _epilogue_ran_this_backward(self, value):
        self._backward_hook_state.epilogue_ran_this_backward = value

    @property
    def _hooks_fired_this_backward(self):
        return self._backward_hook_state.hooks_fired_this_backward

    @_hooks_fired_this_backward.setter
    def _hooks_fired_this_backward(self, value):
        self._backward_hook_state.hooks_fired_this_backward = value

    @property
    def _max_expected_hooks_seen(self):
        return self._backward_hook_state.max_expected_hooks_seen

    @_max_expected_hooks_seen.setter
    def _max_expected_hooks_seen(self, value):
        self._backward_hook_state.max_expected_hooks_seen = value

    @property
    def _grad_acc_post_hooks(self):
        return self._backward_hook_state._grad_acc_post_hooks

    @_grad_acc_post_hooks.setter
    def _grad_acc_post_hooks(self, value):
        self._backward_hook_state._grad_acc_post_hooks = value

    def load_hp_checkpoint_state_from_checkpoint_dir(self, lp_groups_name: str, checkpoint_dir: str) -> None:
        checkpoint_dir = os.path.join(checkpoint_dir, "zero")
        optim_state_path = os.path.join(checkpoint_dir, "optimizer_state.pt")
        assert os.path.isfile(
            optim_state_path), f'{optim_state_path} containing optimizer global state is missing! Cannot proceed.'
        optim_sd = torch.load(optim_state_path, weights_only=False)

        self._load_global_state(optim_sd)

        tp_rank = bwc_tensor_model_parallel_rank(mpu=self.mpu)
        if self.mpu is None:
            logger.warning("MPU is not provided, setting tp size to 1 in checkpoint loading.")
            tp_world_size = 1
        else:
            tp_world_size = self.mpu.get_slice_parallel_world_size() if hasattr(self.mpu, "get_slice_parallel_world_size") \
                else self.mpu.get_tensor_model_parallel_world_size()

        ep_rank, ep_size = _get_universal_checkpoint_ep_info()

        for i, (param_group,
                loaded_param_group) in enumerate(zip(self.optimizer.param_groups, optim_sd['param_groups'])):
            # We have an assumption that all params in the same param_group have the same keys
            opt_keys = set()
            steps = []

            lp_groups = getattr(self, lp_groups_name)
            for lp in lp_groups[i]:
                if lp._hp_mapping is not None:
                    #print(f"Loading {self.param_names[lp]} {tp_rank=} {tp_world_size=}")
                    step = lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]),
                                                       tp_rank,
                                                       tp_world_size,
                                                       ep_rank=ep_rank,
                                                       ep_size=ep_size)
                    for key in lp._hp_mapping.get_optim_state_keys():
                        opt_keys.add(key)
                    steps.append(step)

            hp_param = param_group['params'][0]
            assert all(step == steps[0] for step in steps), f"Steps {steps} are not equal"
            if steps[0] is not None:
                self.optimizer.state[hp_param]['step'] = steps[0]

            map_to_flat_opt_states(hp_param, lp_groups[i], self.optimizer.state, opt_keys)

            for key, value in loaded_param_group.items():
                if key == 'params':
                    continue
                param_group[key] = value

    def report_ipg_memory_usage(self, tag, param_elems, dtype=None):
        dtypes = self.ipg_buckets.keys() if dtype is None else [dtype]

        for dt in dtypes:
            bucket = self.ipg_buckets[dt]
            elem_count = bucket.elements + param_elems
            percent_of_bucket_size = (100.0 * elem_count) // self.reduce_bucket_size
            see_memory_usage(
                f"{tag}: elems in_bucket {dt} {bucket.elements} param {param_elems} max_percent {percent_of_bucket_size}"
            )

    def get_param_comm_dtype(self, param):
        if is_autocast_initialized():
            return get_comm_dtype(param)
        else:
            return self.communication_data_type

    def needs_scaler(self) -> bool:
        """
        Check if this optimizer requires loss scaling for correct backward pass.

        Returns True if any of the following conditions are met:
        - Custom loss scaler is enabled
        - torch.autocast gradient scaler is active (fp16 only)
        - Dynamic loss scaling is enabled (fp16 with DeepSpeed's loss scaler)

        Returns False for bf16 or fp32, which don't require gradient scaling.
        """
        return (self.custom_loss_scaler or self.torch_autocast_gradscaler is not None
                or (hasattr(self, 'dynamic_loss_scale') and self.dynamic_loss_scale))

    def scale_if_loss(self, value: Any) -> Any:
        """
        Applies loss scaling to the input value if it is a loss tensor.
        """
        if maybe_loss_for_backward(value):
            if self.custom_loss_scaler:
                return self.external_loss_scale * value
            if self.torch_autocast_gradscaler:
                return self.torch_autocast_gradscaler.scale(value)
            # Only call loss_scaler if it exists (not present in BF16_Optimizer)
            if hasattr(self, 'loss_scaler') and self.loss_scaler is not None:
                return self.loss_scaler.scale_loss(value)

        return value

    def backward_prologue(self):
        pass

    def backward_epilogue(self, **kwargs):
        pass

    def backward(self, loss, **kwargs):
        assert maybe_loss_for_backward(loss), "Optimizer's backward() only accepts a scalar tensor"

        scaled_loss = self.backward_prologue(loss)
        retain_graph = kwargs.pop('retain_graph', False)
        self.enter_backward()
        scaled_loss.backward(retain_graph=retain_graph)
        self.backward_epilogue()
        self.exit_backward()

    def register_grad_acc_post_hook(self, hook):
        """Register a callback to run when all gradient hooks have fired."""
        self._backward_hook_state.register_grad_acc_post_hook(hook)

    def unregister_grad_acc_post_hooks(self):
        """Remove all registered gradient accumulation post hooks."""
        self._backward_hook_state.unregister_grad_acc_post_hooks()

    def run_grad_acc_post_hooks(self):
        """Run all registered post hooks if backward is active."""
        self._backward_hook_state.run_grad_acc_post_hooks()

    def enter_backward(self):
        """Enter backward context. Call at the start of backward pass."""
        self._backward_hook_state.enter_backward()

    def exit_backward(self):
        """Exit backward context. Call at the end of backward pass."""
        self._backward_hook_state.exit_backward()
        if self._backward_active_depth == 0:
            self._backward_completed_since_step = True

    @staticmethod
    def _aborted_backward_error():
        return RuntimeError("An aborted backward already contributed gradients; discard this engine and restart the "
                            "accumulation window")

    def _ensure_backward_complete_before_step(self):
        if self._aborted_backward_requires_restart or self._backward_active_depth > 0:
            self._aborted_backward_requires_restart = True
            raise self._aborted_backward_error()

    def _track_ipg_bucket_reductions(self, buckets):
        for bucket in buckets.values():
            if not isinstance(bucket.params, _IPGBucketParameterList):
                bucket.params = _IPGBucketParameterList(bucket.params, self)

    def _clear_unreduced_parameter_gradients(self):
        groups = getattr(self, "bit16_groups", None)
        if groups is None:
            groups = getattr(self, "fp16_groups", None)
        if groups is None:
            groups = getattr(self, "bf16_groups", ())
        for group in groups:
            for param in group:
                param.grad = None
                if hasattr(param, "grad_accum"):
                    param.grad_accum = None

    def clear_backward_seen_flag(self):
        """Reset hook state and discard partial reduction state from an aborted backward."""
        buckets = getattr(self, "ipg_buckets", {})
        self._track_ipg_bucket_reductions(buckets)
        if self._aborted_backward_requires_restart:
            raise self._aborted_backward_error()

        hooks_fired = self._backward_hook_state.hooks_fired_this_backward
        params_already_reduced = getattr(self, "params_already_reduced", None)
        if isinstance(params_already_reduced, dict):
            any_param_reduced = any(params_already_reduced.values())
        elif params_already_reduced is not None:
            any_param_reduced = any(params_already_reduced)
        else:
            any_param_reduced = False
        bf16_grad_contributed = any(any(group) for group in getattr(self, "fp32_groups_has_gradients", ()))
        zero1_grad_contributed = (hasattr(self, "partition_gradients") and not self.partition_gradients
                                  and self._backward_completed_since_step)
        coalesced_grad_contributed = (getattr(self, "_coalesce_grad_reduction", False)
                                      and self._backward_completed_since_step)
        stage12_buckets_were_ready = getattr(self, "ready_for_gradients", False)

        incomplete_backward = self._backward_hook_state.reset_for_new_step()
        if not incomplete_backward:
            self._backward_reduction_observed = False
            return

        # Bucket parameter lists are instrumented before every backward. Clearing
        # a non-empty list while backward is active means a reduction crossed into
        # persistent accumulation state. ZeRO-1's separate grad_accum path is also
        # irreversible once any hook has contributed to the shared accumulator.
        partial_reduction = (self._backward_reduction_observed or any_param_reduced or zero1_grad_contributed
                             or coalesced_grad_contributed
                             or (getattr(self, "use_grad_accum_attribute", False) and hooks_fired > 0)
                             or bf16_grad_contributed)

        # Autograd exceptions bypass the backward epilogue, so a partially filled
        # IPG bucket and its reduced flags must not flow into the next root forward.
        self._clearing_aborted_backward = True
        try:
            for bucket in buckets.values():
                if hasattr(bucket, "clear_params"):
                    bucket.clear_params()
                else:
                    bucket.clear()
        finally:
            self._clearing_aborted_backward = False

        self._clear_unreduced_parameter_gradients()

        extra_large_params = getattr(self, "extra_large_param_to_reduce", None)
        if extra_large_params is not None:
            extra_large_params.clear()

        if isinstance(params_already_reduced, dict):
            for param_id in params_already_reduced:
                params_already_reduced[param_id] = False
        elif params_already_reduced is not None:
            for param_id in range(len(params_already_reduced)):
                params_already_reduced[param_id] = False

        if hasattr(self, "reset_partition_gradient_structures"):
            self.reset_partition_gradient_structures()
        if hasattr(self, "ready_for_gradients"):
            self.ready_for_gradients = False
        if hasattr(self, "grads_in_partition_offset"):
            self.grads_in_partition_offset = 0

        if partial_reduction:
            self._aborted_backward_requires_restart = True
            raise self._aborted_backward_error()

        # Stage 1/2 advances micro_step_id when its first gradient hook sets up
        # the IPG buckets. A safe abort before any bucket drain must undo that
        # advance so CPU-offload retry does not import the previous step's
        # accumulated_grads_in_cpu as a current-window contribution.
        if stage12_buckets_were_ready and hasattr(self, "micro_step_id"):
            self.micro_step_id -= 1

        self._backward_reduction_observed = False

    def should_refresh_expected_hook_count(self):
        """Return True when count_used_parameters_in_backward() should be re-evaluated."""
        return self._backward_hook_state.should_refresh_expected_hook_count()

    def reenter_backward_if_needed(self):
        """Re-enter backward context for subsequent phases in reentrant checkpointing."""
        self._backward_hook_state.reenter_backward_if_needed()

    def update_hook_state_and_maybe_run_epilogue(self, current_expected_count):
        """Update hook state after a gradient hook fires and run epilogue if all hooks have fired."""
        self._backward_hook_state.update_hook_state_and_maybe_run_epilogue(current_expected_count)

    def queue_post_backward_callback(self):
        """Queue post-backward hooks to run after autograd completes."""
        return self._backward_hook_state.queue_post_backward_callback()

    def _configure_master_weights(self,
                                  fp16_master_weights_and_gradients=False,
                                  bf16_master_weights_and_gradients=False,
                                  bf16_optimizer_states=False,
                                  offload_enabled=False,
                                  fp16_offload_validator=None,
                                  bf16_offload_validator=None):
        """
        Common validation and dtype selection for ZeRO optimizer master-weight settings.
        Optionally accepts callables that enforce backend-specific offload requirements.
        ``offload_enabled`` tells this method whether optimizer-state offload is configured,
        so the offload requirement is also enforced for the bf16-optimizer-states + offload case.
        """
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients
        self.bf16_master_weights_and_gradients = bf16_master_weights_and_gradients
        assert not (self.fp16_master_weights_and_gradients and self.bf16_master_weights_and_gradients), \
            "fp16_master_weights_and_gradients and bf16_master_weights_and_gradients are mutually exclusive."

        self.bf16_optimizer_states = bf16_optimizer_states
        if self.bf16_optimizer_states:
            assert self.bf16_master_weights_and_gradients, \
                "bf16_optimizer_states requires bf16_master_weights_and_gradients."

        # bf16 master weights require ZeRO-Offload + DeepSpeedCPUAdam whenever the optimizer states
        # cannot stay on the GPU: either because they remain fp32 (bf16_optimizer_states disabled),
        # or because CPU offload is explicitly requested alongside bf16 optimizer states.
        if (self.bf16_master_weights_and_gradients and bf16_offload_validator is not None
                and (not self.bf16_optimizer_states or offload_enabled)):
            bf16_offload_validator()
            # Offloaded bf16 optimizer states need the CPU optimizer to store moments in the
            # parameter (bf16) precision; otherwise they would silently expand back to fp32.
            if self.bf16_optimizer_states:
                assert not getattr(self.optimizer, 'fp32_optimizer_states', True), \
                    "bf16_optimizer_states with ZeRO-Offload requires DeepSpeedCPUAdam constructed " \
                    "with fp32_optimizer_states=False so optimizer moments are stored in bf16."

        if self.fp16_master_weights_and_gradients and fp16_offload_validator is not None:
            fp16_offload_validator()

        if self.fp16_master_weights_and_gradients:
            return torch.float16
        elif self.bf16_master_weights_and_gradients:
            return torch.bfloat16
        else:
            return torch.float32
