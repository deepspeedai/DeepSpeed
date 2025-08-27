# Copyright (c) 2025 Peng Du and Zhipeng Wang
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
try:
    from deepspeed.runtime.zero.muon.original_muon import MuonWithAuxAdam as BaseMuonWithAuxAdam
    from deepspeed.runtime.zero.muon.original_muon import adam_update
except ImportError:
    pass


class MuonWithAuxAdam(BaseMuonWithAuxAdam):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
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
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"],
                                         group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
