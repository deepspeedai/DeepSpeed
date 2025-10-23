# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import cast, List, Optional, Tuple, Union
from torch import Tensor

from deepspeed.utils.torch import required_torch_version

# Check if we have PyTorch >= 2.0 for ZenFlow features
_ZENFLOW_AVAILABLE = required_torch_version(min_version=2.1)

if _ZENFLOW_AVAILABLE:
    try:
        from torch.optim.optimizer import (
            _default_to_fused_or_foreach,
            _disable_dynamo_if_unsupported,
            _get_capturable_supported_devices,
            _get_value,
            _stack_if_compiling,
            _view_as_real,
            DeviceDict,
            Optimizer,
        )
    except ImportError as e:
        # print(f"[WARNING] ZenFlow disabled: torch internal optimizer symbols could not be imported: {e}")
        _ZENFLOW_AVAILABLE = False

if not _ZENFLOW_AVAILABLE:
    # safe disable dynamo if unsupported
    def _disable_dynamo_if_unsupported(**kwargs):  # noqa

        def wrapper(fn):
            return fn

        return wrapper

    _ZENFLOW_AVAILABLE = False


class ZenFlowSelectiveAdamW(torch.optim.AdamW):

    def __init__(self, *args, offload=False, bucket_size=5e8, **kwargs):
        if not _ZENFLOW_AVAILABLE:
            raise RuntimeError("ZenFlow features are not available with PyTorch < 2.0. "
                               "Please upgrade to PyTorch 2.0+ to use ZenFlow, or omit 'zenflow' "
                               "from your DeepSpeed configuration to use the default ZeRO-Offload optimizer.")
        super(ZenFlowSelectiveAdamW, self).__init__(*args, **kwargs)

        self.offload = offload

        if offload:
            self.step = self._step_with_offload
            self.bucket_size = bucket_size
        else:
            self.step = self._step_without_offload

    def temp_copy_param(self, group_to_paramlist):
        for group_id, params in group_to_paramlist.items():
            for param in params:
                if hasattr(param, "selected_grad"):
                    temp_selected_param = param.data[:, param.selected_indices].clone().detach() if len(
                        param.shape) != 1 else param.data.clone().detach()
                    if self.offload:
                        param.temp_selected_param = temp_selected_param.cpu()
                    else:
                        param.temp_selected_param = temp_selected_param

    def copy_mv_from_cpu(self, params):
        for param in params:
            param.exp_avg = param.exp_avg_cpu_data.to(param.device, non_blocking=True)
            param.exp_avg_sq = param.exp_avg_sq_cpu_data.to(param.device, non_blocking=True)

    def copy_mv_to_cpu(self, params):
        for param in params:
            param.exp_avg_cpu_data.copy_(param.exp_avg.data, non_blocking=True)
            param.exp_avg_sq_cpu_data.copy_(param.exp_avg_sq.data, non_blocking=True)
            param.exp_avg = None
            param.exp_avg_sq = None

    def clear_selected_mv(self):
        print("Zenflow: clearing selective optimizer states...")
        for group in self.param_groups:
            for param in group['params']:
                state = self.state.setdefault(param, {})
                if len(state) == 0:
                    continue
                if self.offload:
                    param.exp_avg_cpu_data.zero_()
                    param.exp_avg_sq_cpu_data.zero_()
                else:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

    @torch.no_grad()
    def _step_without_offload(self):
        for group in self.param_groups:

            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            for param in group["params"]:
                if hasattr(param, "selected_grad"):
                    selected_param = param.data[:, param.selected_indices] if len(param.shape) != 1 else param.data
                    if hasattr(param, 'temp_selected_param') and param.temp_selected_param is not None:
                        selected_param.copy_(param.temp_selected_param)

                    state = self.state.setdefault(param, {})
                    if len(state) == 0:
                        state["step"] = torch.zeros((), dtype=param.dtype, device=selected_param.device)
                        state["exp_avg"] = torch.zeros_like(selected_param)
                        state["exp_avg_sq"] = torch.zeros_like(selected_param)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(selected_param)

                    params_with_grad.append(selected_param)
                    grads.append(param.selected_grad)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if amsgrad:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state_steps.append(state["step"])

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )

            for i, param in enumerate(group["params"]):
                if hasattr(param, "selected_grad"):
                    if len(param.shape) != 1:
                        param.data[:, param.selected_indices] = params_with_grad[i]

            for param in group["params"]:
                if hasattr(param, "temp_selected_param"):
                    param.temp_selected_param = None
                    param.selected_grad = None

    @torch.no_grad()
    def _step_with_offload(self):
        """
        Performs parameter updates in offload mode.

        In this mode, group_step() calls adamw() on each pre-partitioned param bucket,
        so memory can be released after each bucket update to reduce GPU overhead.
        Without offload, adamw() is called directly for speed.
        """
        for group_id, group in enumerate(self.param_groups):
            params = group["params"]

            bucket = []
            bucket_numel = 0

            def flush_bucket():
                if not bucket:
                    return
                for param in bucket:
                    if hasattr(param, "temp_selected_param") and param.temp_selected_param is not None:
                        selected_param = param.data[:, param.selected_indices] if len(param.shape) != 1 else param.data
                        temp_selected_param = param.temp_selected_param.to(param.device, non_blocking=True)
                        selected_param.copy_(temp_selected_param)
                        param.temp_selected_param = None

                self.group_step({group_id: bucket})
                bucket.clear()

            for param in params:
                if hasattr(param, "selected_grad"):
                    bucket.append(param)
                    bucket_numel += param.numel()
                    if bucket_numel >= self.bucket_size:
                        flush_bucket()
                        bucket_numel = 0

            flush_bucket()

    @torch.no_grad()
    def group_step(self, group_to_paramlist):
        for group_id, params in group_to_paramlist.items():
            group = self.param_groups[group_id]

            if self.offload:
                self.copy_mv_from_cpu(params)

            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []

            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            for param in params:
                if hasattr(param, "selected_grad"):
                    is_2d = (len(param.shape) != 1)
                    selected_param = param.data[:, param.selected_indices] if is_2d else param.data

                    state = self.state.setdefault(param, {})
                    if len(state) == 0:
                        state["step"] = torch.zeros((), dtype=param.dtype, device=selected_param.device)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(selected_param)
                        if not self.offload:
                            state["exp_avg"] = torch.zeros_like(selected_param)
                            state["exp_avg_sq"] = torch.zeros_like(selected_param)

                    if self.offload:
                        exp_avg_t = param.exp_avg.view_as(selected_param)
                        exp_avg_sq_t = param.exp_avg_sq.view_as(selected_param)
                    else:
                        exp_avg_t = state["exp_avg"]
                        exp_avg_sq_t = state["exp_avg_sq"]

                    params_with_grad.append(selected_param)
                    grads.append(param.selected_grad)
                    exp_avgs.append(exp_avg_t)
                    exp_avg_sqs.append(exp_avg_sq_t)
                    if amsgrad:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state_steps.append(state["step"])

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )

            for i, param in enumerate(params):
                if hasattr(param, "selected_grad") and len(param.shape) != 1:
                    param.data[:, param.selected_indices] = params_with_grad[i]

            if self.offload:
                self.copy_mv_to_cpu(params)

            for param in params:
                param.selected_grad = None


class ZenFlowSelectiveAdamW_stage3(torch.optim.AdamW):

    def __init__(self, *args, offload=False, bucket_size=5e8, **kwargs):
        super(ZenFlowSelectiveAdamW_stage3, self).__init__(*args, **kwargs)
        self.offload = offload

        if offload:
            self.step = self._step_with_offload
            self.bucket_size = bucket_size
        else:
            self.step = self._step_without_offload

    @torch.no_grad()
    def temp_copy_param(self, paramlist):
        for param in paramlist:
            if hasattr(param, "selected_grad"):
                num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)

                if num_row != 1:
                    param_2d = param.ds_tensor.data.narrow(0, param.complete_column_offset, param.complete_numel).view(
                        param.complete_numel // num_row, num_row)
                    temp_selected_param = param_2d[param.selected_indices, :].clone().detach()
                else:
                    temp_selected_param = param.ds_tensor.data.clone().detach()

                if self.offload:
                    param.temp_selected_param = temp_selected_param.cpu()
                else:
                    param.temp_selected_param = temp_selected_param

    def clear_selected_mv(self):
        print("Zenflow: clearing selective optimizer states...")
        for group in self.param_groups:
            for param in group['params']:
                state = self.state.setdefault(param, {})
                if len(state) == 0:
                    continue
                if self.offload:
                    param.exp_avg_cpu_data.zero_()
                    param.exp_avg_sq_cpu_data.zero_()
                else:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

    @torch.no_grad()
    def _step_without_offload(self):
        for group in self.param_groups:

            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])
            for param in group["params"]:
                if hasattr(param, "selected_grad"):
                    num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)
                    if num_row != 1:
                        param_2d = param.ds_tensor.data.narrow(0, param.complete_column_offset,
                                                               param.complete_numel).view(
                                                                   param.complete_numel // num_row, num_row)
                        selected_param = param_2d[param.selected_indices, :]
                    else:
                        selected_param = param.ds_tensor.data
                    if hasattr(param, 'temp_selected_param') and param.temp_selected_param is not None:
                        selected_param.copy_(param.temp_selected_param)

                    state = self.state.setdefault(param, {})
                    if len(state) == 0:
                        state["step"] = torch.zeros((), dtype=param.dtype, device=selected_param.device)
                        state["exp_avg"] = torch.zeros_like(selected_param)
                        state["exp_avg_sq"] = torch.zeros_like(selected_param)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(selected_param)

                    params_with_grad.append(selected_param)
                    grads.append(param.selected_grad)
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    if amsgrad:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state_steps.append(state["step"])
            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )
            for i, param in enumerate(group["params"]):
                if hasattr(param, "selected_grad"):
                    num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)
                    if num_row != 1:
                        param_2d = param.ds_tensor.data.narrow(0, param.complete_column_offset,
                                                               param.complete_numel).view(
                                                                   param.complete_numel // num_row, num_row)
                        param_2d[param.selected_indices, :] = params_with_grad[i]

            for param in group["params"]:
                if hasattr(param, "temp_selected_param"):
                    param.temp_selected_param = None
                    param.selected_grad = None

    def copy_mv_from_cpu(self, params):
        for param in params:
            param.exp_avg = param.exp_avg_cpu_data.to(param.device, non_blocking=True)
            param.exp_avg_sq = param.exp_avg_sq_cpu_data.to(param.device, non_blocking=True)

    def copy_mv_to_cpu(self, params):
        for param in params:
            param.exp_avg_cpu_data.copy_(param.exp_avg.data, non_blocking=True)
            param.exp_avg_sq_cpu_data.copy_(param.exp_avg_sq.data, non_blocking=True)
            param.exp_avg = None
            param.exp_avg_sq = None

    @torch.no_grad()
    def group_step(self, paramlist):

        group_to_paramlist = {}
        for param in paramlist:
            group_id = param.group_id
            if group_id not in group_to_paramlist:
                group_to_paramlist[group_id] = []
            group_to_paramlist[group_id].append(param)

        for group_id in sorted(group_to_paramlist.keys()):
            params = group_to_paramlist[group_id]
            group = self.param_groups[group_id]

            if self.offload:
                self.copy_mv_from_cpu(params)

            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []

            amsgrad: bool = group["amsgrad"]
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            for param in params:
                if hasattr(param, "selected_grad"):
                    num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)

                    if num_row != 1:
                        param_2d = param.ds_tensor.data.narrow(0, param.complete_column_offset,
                                                               param.complete_numel).view(
                                                                   param.complete_numel // num_row, num_row)
                        selected_param = param_2d[param.selected_indices, :]
                    else:
                        selected_param = param.ds_tensor.data

                    state = self.state.setdefault(param, {})
                    if len(state) == 0:
                        state["step"] = torch.zeros((), dtype=param.dtype, device=selected_param.device)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(selected_param)
                        if not self.offload:
                            state["exp_avg"] = torch.zeros_like(selected_param)
                            state["exp_avg_sq"] = torch.zeros_like(selected_param)

                    if self.offload:
                        exp_avg_t = param.exp_avg.view_as(selected_param)
                        exp_avg_sq_t = param.exp_avg_sq.view_as(selected_param)
                    else:
                        exp_avg_t = state["exp_avg"]
                        exp_avg_sq_t = state["exp_avg_sq"]

                    params_with_grad.append(selected_param)
                    grads.append(param.selected_grad)
                    exp_avgs.append(exp_avg_t)
                    exp_avg_sqs.append(exp_avg_sq_t)
                    if amsgrad:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    state_steps.append(state["step"])

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )

            for i, param in enumerate(params):
                if hasattr(param, "selected_grad"):
                    num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)
                    if num_row != 1:
                        param_2d = param.ds_tensor.data.narrow(0, param.complete_column_offset,
                                                               param.complete_numel).view(
                                                                   param.complete_numel // num_row, num_row)
                        param_2d[param.selected_indices, :] = params_with_grad[i]

            if self.offload:
                self.copy_mv_to_cpu(params)

            for param in params:
                param.selected_grad = None

    @torch.no_grad()
    def _step_with_offload(self):
        """
        Performs parameter updates in offload mode.

        In this mode, group_step() calls adamw() on each pre-partitioned param bucket,
        so memory can be released after each bucket update to reduce GPU overhead.
        Without offload, adamw() is called directly for speed.
        """

        for group_id, group in enumerate(self.param_groups):
            params = group["params"]

            bucket = []
            bucket_numel = 0

            def flush_bucket():
                if not bucket:
                    return
                for param in bucket:
                    if hasattr(param, "temp_selected_param") and param.temp_selected_param is not None:
                        temp_selected_param = param.temp_selected_param.to(param.device, non_blocking=True)
                        num_column, num_row = param.ds_shape if len(param.ds_shape) != 1 else (param.ds_shape[0], 1)
                        if num_row != 1:
                            param_2d = param.ds_tensor.data.narrow(0, param.complete_column_offset,
                                                                   param.complete_numel).view(
                                                                       param.complete_numel // num_row, num_row)
                            param_2d[param.selected_indices, :] = temp_selected_param
                        else:
                            param.ds_tensor.data.copy_(temp_selected_param)
                        param.temp_selected_param = None

                self.group_step(bucket)
                bucket.clear()

            for param in params:
                if hasattr(param, "selected_grad"):
                    bucket.append(param)
                    bucket_numel += param.numel()
                    if bucket_numel >= self.bucket_size:
                        flush_bucket()
                        bucket_numel = 0

            flush_bucket()


def _single_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(supports_xla=False)
        assert all(
            p.device.type == step.device.type and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs_,
            device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                )
            else:
                _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch._utils.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0)
        else:
            torch._foreach_add_(device_state_steps, 1)

        # Perform stepweight decay
        if weight_decay != 0:
            torch._foreach_mul_(device_params, 1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        bias_correction1: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2: Union[Tuple[Tensor, ...], List[Tensor]]
        bias_correction2_sqrt: Union[Tuple[Tensor, ...], List[Tensor]]

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1**_get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2**_get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [
                bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
            ]

            if amsgrad:
                device_max_exp_avg_sqs = cast(List[Tensor], device_max_exp_avg_sqs_)

                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(
                device_params,
                device_exp_avgs,
                exp_avg_sq_sqrt,
                step_size,  # type: ignore[arg-type]
            )


def _fused_adamw(
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor],
        grad_scale: Optional[Tensor],
        found_inf: Optional[Tensor],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: Union[Tensor, float],
        weight_decay: float,
        eps: float,
        maximize: bool,
        capturable: bool,  # Needed for consistency.
        differentiable: bool,
        has_complex: bool,  # Needed for consistency.
) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    grad_scale_dict: DeviceDict = ({grad_scale.device: grad_scale} if grad_scale is not None else {})
    found_inf_dict: DeviceDict = ({found_inf.device: found_inf} if found_inf is not None else {})

    # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    lr_dict: Optional[DeviceDict] = ({lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs,
            device_state_steps_,
        ),
            _,
    ) in grouped_tensors.items():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        if device.type == "mps":  # type: ignore[union-attr]
            assert found_inf is None and grad_scale is None

        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(device, grad_scale.to(device, non_blocking=True))
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(device, found_inf.to(device, non_blocking=True))
        if lr_dict is not None and device not in lr_dict:
            lr = lr_dict.setdefault(
                device,
                lr.to(device=device, non_blocking=True)  # type: ignore[union-attr]
            )
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adamw_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,  # type: ignore[arg-type]
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,  # type: ignore[arg-type]
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(device_state_steps, [device_found_inf] * len(device_state_steps))


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamw)
def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not _ZENFLOW_AVAILABLE:
        raise RuntimeError("ZenFlow adamw function is not available with PyTorch < 2.0. "
                           "Please upgrade to PyTorch 2.0+ to use ZenFlow, or omit 'zenflow' "
                           "from your DeepSpeed configuration to use the default ZeRO-Offload optimizer.")

    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adamw
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
    )
