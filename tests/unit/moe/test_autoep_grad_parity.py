# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP vs ZeRO-2 parity checks for mixed logical-DP / EP training."""

import copy

import deepspeed
import deepspeed.comm as dist
import pytest
import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import safe_get_full_grad
from transformers import AutoModelForCausalLM, MixtralConfig
from unit.common import DistributedTest


def _mixed_precision_config():
    accelerator = get_accelerator()
    if accelerator.is_bf16_supported():
        return {"bf16": {"enabled": True}}
    if accelerator.is_fp16_supported() and accelerator.device_name() != "cpu":
        return {
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8,
            },
        }
    if accelerator.is_fp16_supported():
        return {
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8,
            },
        }
    pytest.skip("AutoEP grad parity tests require fp16 or bf16 support")


def _make_model_config():
    return MixtralConfig(
        num_hidden_layers=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=512,
        output_router_logits=False,
        router_jitter_noise=0.0,
        tie_word_embeddings=False,
    )


def _make_zero2_config(clip_grad):
    return {
        **_mixed_precision_config(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": clip_grad,
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


def _make_autoep_zero2_config(clip_grad, ep_size):
    config = _make_zero2_config(clip_grad)
    config["gradient_accumulation_steps"] = 2
    config["expert_parallel"] = {
        "enabled": True,
        "autoep_size": ep_size,
        "preset_model": "mixtral",
        "load_balance_coeff": None,
    }
    return config


def _seed_everything(seed=1234):
    torch.manual_seed(seed)
    get_accelerator().manual_seed(seed)
    get_accelerator().manual_seed_all(seed)


def _make_local_batches(*, logical_dp_world_size, logical_dp_rank, grad_accum, seed, seq_len, micro_batch_size,
                        vocab_size, device):
    batches = []
    for accum_idx in range(grad_accum):
        batch_idx = accum_idx * logical_dp_world_size + logical_dp_rank
        generator = torch.Generator().manual_seed(seed + batch_idx)
        input_ids = torch.randint(
            0,
            vocab_size,
            (micro_batch_size, seq_len),
            generator=generator,
            dtype=torch.long,
        ).to(device)
        attention_mask = torch.ones_like(input_ids)
        batches.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        })
    return batches


def _run_until_boundary(engine, *, logical_dp_world_size, logical_dp_rank, grad_accum, seed, use_manual_scale=False):
    batches = _make_local_batches(
        logical_dp_world_size=logical_dp_world_size,
        logical_dp_rank=logical_dp_rank,
        grad_accum=grad_accum,
        seed=seed,
        seq_len=16,
        micro_batch_size=1,
        vocab_size=512,
        device=engine.device,
    )
    for batch_idx, batch in enumerate(batches):
        outputs = engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if use_manual_scale:
            scaled_loss = engine.scale(outputs.loss)
            scaled_loss.backward()
        else:
            engine.backward(outputs.loss)
        if batch_idx + 1 < len(batches):
            engine.step()


def _normalize_autoep_name(name):
    return name.replace(".mlp.router.gate.", ".mlp.gate.")


def _collect_nonexpert_grads(engine):
    grads = {}
    for name, param in engine.module.named_parameters():
        if ".experts." in name:
            continue
        grad = safe_get_full_grad(param)
        assert grad is not None, f"Expected full grad for {name}"
        grads[_normalize_autoep_name(name)] = grad.detach().float().cpu().clone()
    return grads


def _gather_autoep_expert_grad(param, group):
    grad = safe_get_full_grad(param)
    assert grad is not None, "Expected full expert grad"
    shards = [torch.zeros_like(grad) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(shards, grad.detach(), group=group)
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


def _assert_grad_maps_close(actual, expected, *, lhs_name, rhs_name, clip_grad):
    for name in sorted(expected):
        assert name in actual, f"Missing {lhs_name} param snapshot for {name}"
        torch.testing.assert_close(actual[name],
                                   expected[name],
                                   atol=5e-3,
                                   rtol=5e-3,
                                   msg=(f"Gradient mismatch for {name} between {lhs_name} and {rhs_name} "
                                        f"with clip_grad={clip_grad}"))


class TestAutoEPGradParity(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("clip_grad", [0.0, 1.0])
    def test_zero2_autoep_matches_zero2_after_one_update(self, clip_grad):
        ep_size = 2
        seed = 1234

        _seed_everything(seed)
        model_config = _make_model_config()
        reference_state = AutoModelForCausalLM.from_config(model_config).state_dict()

        autoep_model = AutoModelForCausalLM.from_config(model_config)
        zero2_model = AutoModelForCausalLM.from_config(model_config)
        autoep_model.load_state_dict(copy.deepcopy(reference_state))
        zero2_model.load_state_dict(copy.deepcopy(reference_state))

        autoep_engine, _, _, _ = deepspeed.initialize(model=autoep_model,
                                                      config=_make_autoep_zero2_config(clip_grad, ep_size))
        zero2_engine, _, _, _ = deepspeed.initialize(model=zero2_model, config=_make_zero2_config(clip_grad))

        autoep_rank = dist.get_rank() // ep_size
        _run_until_boundary(autoep_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(zero2_engine,
                            logical_dp_world_size=self.world_size,
                            logical_dp_rank=dist.get_rank(),
                            grad_accum=1,
                            seed=seed)

        autoep_nonexpert = _collect_nonexpert_grads(autoep_engine)
        autoep_expert = _collect_autoep_expert_grads(autoep_engine)
        zero2_nonexpert = _collect_nonexpert_grads(zero2_engine)
        zero2_expert = _collect_zero2_expert_grads(zero2_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(autoep_nonexpert,
                                zero2_nonexpert,
                                lhs_name="AutoEP",
                                rhs_name="ZeRO-2",
                                clip_grad=clip_grad)
        _assert_grad_maps_close(autoep_expert,
                                zero2_expert,
                                lhs_name="AutoEP expert",
                                rhs_name="ZeRO-2 expert",
                                clip_grad=clip_grad)

    @pytest.mark.parametrize("clip_grad", [0.0, 1.0])
    def test_zero2_autoep_scale_matches_engine_backward(self, clip_grad):
        ep_size = 2
        seed = 1234

        _seed_everything(seed)
        model_config = _make_model_config()
        reference_state = AutoModelForCausalLM.from_config(model_config).state_dict()

        autoep_backward_model = AutoModelForCausalLM.from_config(model_config)
        autoep_manual_model = AutoModelForCausalLM.from_config(model_config)
        autoep_backward_model.load_state_dict(copy.deepcopy(reference_state))
        autoep_manual_model.load_state_dict(copy.deepcopy(reference_state))

        autoep_backward_engine, _, _, _ = deepspeed.initialize(model=autoep_backward_model,
                                                               config=_make_autoep_zero2_config(clip_grad, ep_size))
        autoep_manual_engine, _, _, _ = deepspeed.initialize(model=autoep_manual_model,
                                                             config=_make_autoep_zero2_config(clip_grad, ep_size))

        autoep_rank = dist.get_rank() // ep_size
        _run_until_boundary(autoep_backward_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed)
        _run_until_boundary(autoep_manual_engine,
                            logical_dp_world_size=self.world_size // ep_size,
                            logical_dp_rank=autoep_rank,
                            grad_accum=2,
                            seed=seed,
                            use_manual_scale=True)

        autoep_backward_nonexpert = _collect_nonexpert_grads(autoep_backward_engine)
        autoep_backward_expert = _collect_autoep_expert_grads(autoep_backward_engine)
        autoep_manual_nonexpert = _collect_nonexpert_grads(autoep_manual_engine)
        autoep_manual_expert = _collect_autoep_expert_grads(autoep_manual_engine)

        dist.barrier()
        if dist.get_rank() != 0:
            return

        _assert_grad_maps_close(autoep_manual_nonexpert,
                                autoep_backward_nonexpert,
                                lhs_name="AutoEP manual backward",
                                rhs_name="AutoEP engine.backward",
                                clip_grad=clip_grad)
        _assert_grad_maps_close(autoep_manual_expert,
                                autoep_backward_expert,
                                lhs_name="AutoEP manual expert backward",
                                rhs_name="AutoEP engine.backward expert",
                                clip_grad=clip_grad)
