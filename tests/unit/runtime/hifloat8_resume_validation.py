# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Opt-in NPU DP2 ZeRO-2 save/load validation for Dense and grouped HiFloat8.

Run with a valid two-device Ascend rank table and torchao_npu installed::

    torchrun --nproc_per_node=2 tests/unit/runtime/hifloat8_resume_validation.py \
        --output /tmp/hifloat8-resume

GroupedExperts are replicated for this DP2 lifecycle check. AutoEP expert
parallel routing is validated separately with a model integration run.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import deepspeed
import deepspeed.comm as dist
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from torchao_npu.hifloat8 import HiFloat8Config


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(128, 128, bias=False)
        self.experts = GroupedExperts(128, 128, 2, use_grouped_mm=False)
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(std=0.02)

    def forward(self, x):
        counts = torch.tensor((0, x.shape[0]), device=x.device, dtype=torch.int64)
        return self.experts(self.dense(x), counts)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    return copy.deepcopy(value)


def compare_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, LossScaler):
        assert type(a) is type(b)
        compare_nested(vars(a), vars(b))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_nested(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    deepspeed.init_distributed(dist_backend='hccl')
    torch.manual_seed(42)
    initial = Model().bfloat16().state_dict()
    config = {
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 2,
        'bf16': {
            'enabled': True
        },
        'zero_optimization': {
            'stage': 2,
            'overlap_comm': False
        },
        'hifloat8': {
            'enabled': True,
            'backend': 'torchao_npu',
            'module_name_patterns': ['dense', 'experts'],
            'expected_module_count': 2
        },
        'steps_per_print': 10000,
    }
    generator = torch.Generator().manual_seed(123 + rank)
    batches = [torch.randn(32, 128, generator=generator).bfloat16().to('npu') for _ in range(8)]
    targets = [torch.randn(32, 128, generator=generator).bfloat16().to('npu') * 0.01 for _ in range(8)]

    def create():
        model = Model().bfloat16()
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        assert engine.module.dense.config is engine.module.experts.hifloat8_config
        assert isinstance(engine.module.dense.config, HiFloat8Config)
        return engine

    def step(engine, i):
        loss = (engine(batches[i]).float() - targets[i].float()).square().mean()
        engine.backward(loss)
        engine.step()
        return float(loss.detach())

    engine = create()
    losses = []
    for i in range(8):
        losses.append(step(engine, i))
        if i == 3:
            assert engine.global_steps == 2
            engine.save_checkpoint(str(args.output / 'checkpoint'), tag='step2', client_state={'marker': 2})
    final = snapshot(engine.module.state_dict())
    optimizer_final = snapshot(engine.optimizer.state_dict())
    resumed = create()
    path, state = resumed.load_checkpoint(str(args.output / 'checkpoint'), tag='step2')
    assert path and state['marker'] == 2 and resumed.global_steps == 2
    resumed_losses = [step(resumed, i) for i in range(4, 8)]
    assert resumed_losses == losses[4:]
    compare_nested(final, resumed.module.state_dict())
    compare_nested(optimizer_final, resumed.optimizer.state_dict())
    assert resumed.global_steps == 4
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'rank': rank,
        'world_size': dist.get_world_size(),
        'zero_stage': 2,
        'gradient_accumulation_steps': 2,
        'optimizer_steps': 4,
        'losses': losses,
        'resume_losses': resumed_losses,
        'parameters_exact': True,
        'optimizer_exact': True,
        'same_policy_object': True
    }
    (args.output / f'result_rank{rank}.json').write_text(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
