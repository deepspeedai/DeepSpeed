# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP gradient parity paths."""

import deepspeed
import deepspeed.comm as dist
import torch
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoETransformer,
    engine_input_dtype as _engine_input_dtype,
    mixed_precision_config as _mixed_precision_config,
    seed_everything as _seed_everything,
)


def _make_model():
    return MockMoETransformer(num_layers=1, num_experts=4, hidden_size=128, intermediate_size=256)


def _make_zero2_config():
    return {
        **_mixed_precision_config(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
        },
    }


def _make_autoep_zero2_config(ep_size):
    config = _make_zero2_config()
    config["expert_parallel"] = {
        "enabled": True,
        "autoep_size": ep_size,
        "preset_model": "mixtral",
        "load_balance_coeff": None,
        "use_grouped_mm": False,
    }
    return config


def _make_autoep_zero3_config(ep_size):
    config = _make_autoep_zero2_config(ep_size)
    config["zero_optimization"] = {
        "stage": 3,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
    }
    return config


def _make_local_batches(*, logical_dp_world_size, logical_dp_rank, grad_accum, seed, seq_len, micro_batch_size,
                        hidden_size, device, dtype):
    batches = []
    for accum_idx in range(grad_accum):
        batch_idx = accum_idx * logical_dp_world_size + logical_dp_rank
        generator = torch.Generator().manual_seed(seed + batch_idx)
        batches.append(
            torch.randn((micro_batch_size, seq_len, hidden_size), generator=generator, dtype=dtype).to(device))
    return batches


def _run_until_boundary(engine, *, logical_dp_world_size, logical_dp_rank, grad_accum, seed):
    batches = _make_local_batches(
        logical_dp_world_size=logical_dp_world_size,
        logical_dp_rank=logical_dp_rank,
        grad_accum=grad_accum,
        seed=seed,
        seq_len=16,
        micro_batch_size=1,
        hidden_size=128,
        device=engine.device,
        dtype=_engine_input_dtype(engine),
    )
    for batch_idx, batch in enumerate(batches):
        loss = engine(batch).mean()
        engine.backward(loss)
        if batch_idx + 1 < len(batches):
            engine.step()


def _gather_autoep_expert_grad(param, group):
    grad = safe_get_full_grad(param)
    assert grad is not None, "Expected full expert grad"
    group_size = dist.get_world_size(group=group)
    shards = [torch.zeros_like(grad) for _ in range(group_size)]
    dist.all_gather(shards, grad.detach(), group=group)
    # The gather reconstructs expert shards; gradient reduction has already
    # applied the data-parallel normalization, so do not average by EP size.
    return torch.cat([shard.float().cpu() for shard in shards], dim=0)


def _collect_autoep_expert_grads(engine):
    from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer

    grads = {}
    for module_name, module in engine.module.named_modules():
        if not isinstance(module, AutoEPMoELayer):
            continue
        prefix = f"{module_name}.experts"
        w1 = _gather_autoep_expert_grad(module.experts.w1, module.ep_group)
        w2 = _gather_autoep_expert_grad(module.experts.w2, module.ep_group)
        w3 = _gather_autoep_expert_grad(module.experts.w3, module.ep_group)
        grads[f"{prefix}.gate_up_proj"] = torch.cat([w1, w3], dim=1)
        grads[f"{prefix}.down_proj"] = w2
    return grads


def _collect_zero2_expert_grads(engine):
    grads = {}
    for name, param in engine.module.named_parameters():
        if name.endswith(".experts.gate_up_proj") or name.endswith(".experts.down_proj"):
            grad = safe_get_full_grad(param)
            assert grad is not None, f"Expected full grad for {name}"
            grads[name] = grad.detach().float().cpu().clone()
    return grads


def _assert_grad_maps_close(actual, expected, *, lhs_name, rhs_name):
    for name in sorted(expected):
        assert name in actual, f"Missing {lhs_name} param snapshot for {name}"
        diff = (actual[name] - expected[name]).abs()
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   atol=1e-1,
                                   rtol=5e-3,
                                   msg=(f"Gradient mismatch for {name} between {lhs_name} and {rhs_name}; "
                                        f"max_diff={diff.max().item()} "
                                        f"actual_norm={actual[name].norm().item()} "
                                        f"expected_norm={expected[name].norm().item()}"))


class TestAutoEPGradParity(DistributedTest):
    world_size = 4

    def test_zero2_autoep_matches_zero2_after_one_update(self):
        ep_size = 2
        seed = 1234

        _seed_everything(seed)
        reference_state = _make_model().state_dict()

        autoep_model = _make_model()
        zero2_model = _make_model()
        autoep_model.load_state_dict(reference_state)
        zero2_model.load_state_dict(reference_state)

        autoep_engine, _, _, _ = deepspeed.initialize(model=autoep_model, config=_make_autoep_zero2_config(ep_size))
        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_zero2_config())

        autoep_rank = dist.get_rank() // ep_size
        _run_until_boundary(autoep_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)

        autoep_expert = _collect_autoep_expert_grads(autoep_engine)
        zero2_expert = _collect_zero2_expert_grads(zero2_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(autoep_expert, zero2_expert, lhs_name="AutoEP expert", rhs_name="ZeRO-2 expert")

    def test_zero3_autoep_expert_grads_match_zero2_autoep(self):
        ep_size = 2
        seed = 2345

        _seed_everything(seed)
        reference_state = _make_model().state_dict()

        zero2_model = _make_model()
        zero3_model = _make_model()
        zero2_model.load_state_dict(reference_state)
        zero3_model.load_state_dict(reference_state)

        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_autoep_zero2_config(ep_size))
        zero3_engine, _, _, _ = deepspeed.initialize(model=zero3_model, config=_make_autoep_zero3_config(ep_size))

        logical_rank = dist.get_rank() // ep_size
        logical_world_size = self.world_size // ep_size
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=logical_world_size,
                            logical_dp_rank=logical_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero3_engine,
                            logical_dp_world_size=logical_world_size,
                            logical_dp_rank=logical_rank,
                            grad_accum=2,
                            seed=seed)

        zero2_expert = _collect_autoep_expert_grads(zero2_engine)
        zero3_expert = _collect_autoep_expert_grads(zero3_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(zero3_expert,
                                zero2_expert,
                                lhs_name="ZeRO-3 AutoEP expert",
                                rhs_name="ZeRO-2 AutoEP expert")
