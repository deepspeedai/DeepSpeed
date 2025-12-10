# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
from typing import Any

from deepspeed.utils import logger
from deepspeed.utils.tensor_fragment import map_to_flat_opt_states
from deepspeed.runtime.utils import bwc_tensor_model_parallel_rank, see_memory_usage
from deepspeed.runtime.torch_autocast import get_comm_dtype, is_autocast_initialized
from deepspeed.runtime.utils import maybe_loss_for_backward


class DeepSpeedOptimizer(object):
    pass


class ZeROOptimizer(DeepSpeedOptimizer):

    def __init__(self):
        self._remaining_grad_acc_hooks = 0
        self._grad_acc_post_hooks = []
        self._backward_active_depth = 0

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

        for i, (param_group,
                loaded_param_group) in enumerate(zip(self.optimizer.param_groups, optim_sd['param_groups'])):
            # We have an assumption that all params in the same param_group have the same keys
            opt_keys = set()
            steps = []

            lp_groups = getattr(self, lp_groups_name)
            for lp in lp_groups[i]:
                if lp._hp_mapping is not None:
                    #print(f"Loading {self.param_names[lp]} {tp_rank=} {tp_world_size=}")
                    step = lp.load_hp_checkpoint_state(os.path.join(checkpoint_dir, self.param_names[lp]), tp_rank,
                                                       tp_world_size)
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
        self._grad_acc_post_hooks.append(hook)

    def unregister_grad_acc_post_hooks(self):
        self._grad_acc_post_hooks = []

    def run_grad_acc_post_hooks(self):
        # Custom autograd Functions (e.g., TiledFusedLogitsLoss) can invoke
        # `torch.autograd.backward()` from their *forward* pass before the user
        # ever calls `engine.backward(loss)`. Those early backward calls still
        # trigger ZeRO's grad hooks, but we must not run the engine's
        # post-backward logic (which reduces/clears grads) until the outer/user
        # backward is active. The depth guard filters out only those pre-user
        # invocations while still allowing backward calls that happen during
        # the real user backward.
        if self._backward_active_depth == 0:
            return
        for hook in self._grad_acc_post_hooks:
            hook()

    def enter_backward(self):
        self._backward_active_depth += 1

    def exit_backward(self):
        if self._backward_active_depth > 0:
            self._backward_active_depth -= 1

    def _configure_master_weights(self,
                                  fp16_master_weights_and_gradients=False,
                                  bf16_master_weights_and_gradients=False,
                                  bf16_optimizer_states=False,
                                  fp16_offload_validator=None,
                                  bf16_fp32_offload_validator=None):
        """
        Common validation and dtype selection for ZeRO optimizer master-weight settings.
        Optionally accepts callables that enforce backend-specific offload requirements.
        """
        self.fp16_master_weights_and_gradients = fp16_master_weights_and_gradients
        self.bf16_master_weights_and_gradients = bf16_master_weights_and_gradients
        assert not (self.fp16_master_weights_and_gradients and self.bf16_master_weights_and_gradients), \
            "fp16_master_weights_and_gradients and bf16_master_weights_and_gradients are mutually exclusive."

        self.bf16_optimizer_states = bf16_optimizer_states
        if self.bf16_optimizer_states:
            assert self.bf16_master_weights_and_gradients, \
                "bf16_optimizer_states requires bf16_master_weights_and_gradients."

        if (self.bf16_master_weights_and_gradients and not self.bf16_optimizer_states
                and bf16_fp32_offload_validator is not None):
            bf16_fp32_offload_validator()

        if self.fp16_master_weights_and_gradients and fp16_offload_validator is not None:
            fp16_offload_validator()

        if self.fp16_master_weights_and_gradients:
            return torch.float16
        elif self.bf16_master_weights_and_gradients:
            return torch.bfloat16
        else:
            return torch.float32
