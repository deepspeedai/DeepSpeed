# Copyright (c) 2025 Peng Du and Zhipeng Wang
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import logger
try:
    from deepspeed.runtime.zero.muon.original_muon import MuonWithAuxAdam as BaseMuonWithAuxAdam
    from deepspeed.runtime.zero.muon.original_muon import adam_update
except ImportError:
    pass


class MuonWithAuxAdam(BaseMuonWithAuxAdam):

    def __init__(self,
                 param_groups,
                 adam_optimizer=None,
                 adam_optimizer_kwargs=None,
                 adam_w_mode=True,
                 fallback_to_inline=False):
        super().__init__(param_groups)
        self.aux_optimizer = None
        self.adam_w_mode = adam_w_mode
        self._aux_param_groups = [group for group in self.param_groups if not group["use_muon"]]
        aux_param_groups = self._aux_param_groups
        if aux_param_groups and adam_optimizer is not None:
            try:
                self.aux_optimizer = adam_optimizer(aux_param_groups, **(adam_optimizer_kwargs or {}))
            except RuntimeError as error:
                if not fallback_to_inline:
                    raise
                logger.warning(f"FusedAdam initialization failed; falling back to Muon's inline Adam update: {error}")
            if self.aux_optimizer is not None:
                for group, aux_group in zip(aux_param_groups, self.aux_optimizer.param_groups):
                    for key, value in aux_group.items():
                        group.setdefault(key, value)
                self.aux_optimizer.param_groups = aux_param_groups
                self.aux_optimizer.state = self.state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.aux_optimizer is None:
            return

        aux_param_groups = [group for group in self.param_groups if not group["use_muon"]]
        for group in aux_param_groups:
            for key, value in self.aux_optimizer.defaults.items():
                group.setdefault(key, value)
        self._aux_param_groups = aux_param_groups
        self.aux_optimizer.param_groups = aux_param_groups
        self.aux_optimizer.state = self.state

    @torch.no_grad()
    def step(self, closure=None, step_id=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                # we move the muon update part to the deepspeed's optimizer since the parameter here is a flat version
                # thus not suitable for muon update
                for p in group["params"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(p.grad.reshape(p.shape), alpha=-group["lr"])

        aux_param_groups = [group for group in self.param_groups if not group["use_muon"]]
        if self.aux_optimizer is not None:
            if (len(aux_param_groups) != len(self._aux_param_groups)
                    or any(current is not cached
                           for current, cached in zip(aux_param_groups, self._aux_param_groups))):
                self._aux_param_groups = aux_param_groups
                self.aux_optimizer.param_groups = aux_param_groups
            for group in aux_param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
            if self.aux_optimizer.state is not self.state:
                self.aux_optimizer.state = self.state
            if step_id is not None and getattr(self.aux_optimizer, "overlap_step", None) is False:
                self.aux_optimizer.step(step_id=step_id)
            else:
                self.aux_optimizer.step()
        else:
            for group in aux_param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    grad = p.grad
                    if not self.adam_w_mode:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"],
                                         group["eps"])
                    if self.adam_w_mode:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
