# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP + AutoTP folding with tensor-parallel attention, against the unmodified model.

The model is a small transformers ``glm4_moe`` (the GLM-4.5-Air architecture): attention with
biases, one dense MLP layer, then MoE layers with a sigmoid router and a shared expert. AutoTP
shards only the attention projections and AutoEP replaces the MoE blocks, under ZeRO-2 and ZeRO-3
(ZeRO-3 also with the model built under ``deepspeed.zero.Init``). Under folding every parameter
family reaches the optimizer by its own reduction: TP-sharded attention over the data-parallel
group, replicated norms, router, shared expert and embeddings averaged over the TP group, routed
experts divided by the TP size and reduced over the expert-data-parallel group. The reference is
the same model with plain data parallelism at ZeRO-0. Both runs see the same micro-batches, so
every gradient and the gradient norm must agree. The attention layers below each MoE layer are what
the other folding tests leave out: their weights are sharded, so they need the MoE layer's input
gradient to be the same on every TP peer.
"""

import copy

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, TensorParallel_Layer
from deepspeed.utils import groups, safe_get_full_grad
from unit.common import DistributedTest

transformers = pytest.importorskip("transformers")

GRAD_ACCUM = 2
SEQ_LEN = 16
ATTENTION_ONLY_TP_SPECS = [
    {
        "patterns": [r".*\.self_attn\.o_proj\.weight$"],
        "partition_type": "row"
    },
    {
        "patterns": [r".*\.self_attn\.[qkv]_proj\.weight$"],
        "partition_type": "column"
    },
]


def _tiny_glm4_moe_config():
    config = transformers.Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        partial_rotary_factor=0.5,
        attention_bias=True,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    # The reference runs the experts in float32; transformers' grouped kernels are bf16 only.
    config._experts_implementation = "eager"
    return config


def _common_config():
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": GRAD_ACCUM,
        # Large enough never to clip, so both engines compute the norm without changing the step.
        "gradient_clipping": 1e9,
        "communication_data_type": "fp32",
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1
            }
        },
        "zero_allow_untested_optimizer": True,
    }


def _reference_config():
    config = _common_config()
    config["zero_optimization"] = {"stage": 0}
    return config


def _folded_config(tp_size, ep_size, zero_stage):
    config = _common_config()
    config["zero_optimization"] = {"stage": zero_stage, "overlap_comm": True, "reduce_scatter": True}
    config["expert_parallel"] = {
        "enabled": True,
        "autoep_size": ep_size,
        "preset_model": "deepseek_v3",
        "use_grouped_mm": False,
    }
    config["tensor_parallel"] = {
        "autotp_size": tp_size,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": ATTENTION_ONLY_TP_SPECS
        },
    }
    return config


def _batch(logical_rank, logical_world_size, micro_step, device):
    generator = torch.Generator().manual_seed(1000 + micro_step * logical_world_size + logical_rank)
    return torch.randint(0, 128, (1, SEQ_LEN), generator=generator).to(device)


def _run_to_boundary(engine, logical_rank, logical_world_size):
    """Forward and backward over one accumulation window; the last backward reduces the gradients."""
    losses = []
    for micro_step in range(GRAD_ACCUM):
        input_ids = _batch(logical_rank, logical_world_size, micro_step, engine.device)
        loss = engine(input_ids=input_ids, labels=input_ids).loss
        engine.backward(loss)
        losses.append(loss.detach().float().item())
        if micro_step + 1 < GRAD_ACCUM:
            engine.step()
    return losses


def _gather_tp_shards(tensor, dim):
    group = groups.get_tensor_model_parallel_group()
    shards = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


def _gather_expert_grads(layer, param):
    grad = safe_get_full_grad(param)
    assert grad is not None
    shards = [torch.empty_like(grad) for _ in range(dist.get_world_size(group=layer.ep_group))]
    dist.all_gather(shards, grad.contiguous(), group=layer.ep_group)
    return torch.cat(shards, dim=0)


def _folded_grads(model):
    """Every gradient of the folded model, rebuilt at the reference model's names and shapes."""
    grads = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, AutoEPMoELayer):
            w1 = _gather_expert_grads(module, module.experts.w1)
            w2 = _gather_expert_grads(module, module.experts.w2)
            w3 = _gather_expert_grads(module, module.experts.w3)
            grads[f"{prefix}experts.gate_up_proj"] = torch.cat([w1, w3], dim=1)
            grads[f"{prefix}experts.down_proj"] = w2
            grads[f"{prefix}gate.weight"] = safe_get_full_grad(module.router.gate.weight)
            continue
        for name, param in module.named_parameters(recurse=False):
            if any(isinstance(owner, AutoEPMoELayer) for owner in _owners(model, module_name)):
                if ".shared_experts." not in f".{prefix}":
                    continue
            grad = safe_get_full_grad(param)
            assert grad is not None, f"{prefix}{name} has no gradient"
            if isinstance(module, LinearAllreduce) and name == "weight":
                grad = _gather_tp_shards(grad, dim=1)
            elif isinstance(module, LinearLayer):
                grad = _gather_tp_shards(grad, dim=0)
            grads[f"{prefix}{name}"] = grad
    return {name: grad.detach().float().cpu() for name, grad in grads.items()}


def _owners(model, module_name):
    """The modules on the path from ``model`` to ``module_name``, excluding the module itself."""
    owners = []
    parts = module_name.split(".") if module_name else []
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
        owners.append(module)
    return owners


def _reference_grads(model):
    return {name: param.grad.detach().float().cpu() for name, param in model.named_parameters()}


def _assert_structure(model):
    """AutoTP shards the attention projections only; AutoEP owns every MoE block."""
    for name, module in model.named_modules():
        is_tp = isinstance(module, TensorParallel_Layer)
        is_attention_projection = ".self_attn." in f".{name}" and name.rsplit(
            ".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
        assert is_tp == is_attention_projection, f"{name}: tensor parallel {is_tp}"
    moe_layers = [module for module in model.modules() if isinstance(module, AutoEPMoELayer)]
    assert len(moe_layers) == model.config.num_hidden_layers - model.config.first_k_dense_replace


def _grad_mismatches(folded, reference):
    """Name, relative error and scale of every gradient that differs from the reference."""
    mismatches = []
    for name, want in reference.items():
        got = folded[name]
        tolerance = 1e-5 * max(want.abs().max().item(), 1e-6)
        if torch.allclose(got, want, rtol=1e-4, atol=tolerance):
            continue
        want_norm_sq = want.square().sum().item()
        scale = got.mul(want).sum().item() / want_norm_sq if want_norm_sq else float("nan")
        relative = (got - want).norm().item() / max(want.norm().item(), 1e-12)
        mismatches.append(f"{name}: relative error {relative:.3e}, scale {scale:.4f}")
    return mismatches


def _check_folded_matches_reference(tp_size, ep_size, zero_stage, zero_init):
    torch.manual_seed(1234)
    config = _tiny_glm4_moe_config()
    reference_model = transformers.Glm4MoeForCausalLM(config).float()
    reference_state = copy.deepcopy(reference_model.state_dict())
    folded_config = _folded_config(tp_size, ep_size, zero_stage)

    if zero_init:
        # How a large model is built: every parameter partitioned over all ranks from the start.
        with deepspeed.zero.Init(config_dict_or_path=folded_config):
            folded_model = transformers.Glm4MoeForCausalLM(config)
        params = list(folded_model.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if dist.get_rank() == 0:
                folded_model.load_state_dict(reference_state)
    else:
        folded_model = transformers.Glm4MoeForCausalLM(config).float()
        folded_model.load_state_dict(reference_state)

    reference_engine, _, _, _ = deepspeed.initialize(model=reference_model, config=_reference_config())
    folded_engine, _, _, _ = deepspeed.initialize(model=folded_model, config=folded_config)
    _assert_structure(folded_engine.module)

    # The reference sees the same micro-batches: rank r takes the batch of the folded run's
    # data-parallel rank r // tp_size, so its average over all ranks is the folded average.
    logical_world_size = dist.get_world_size() // tp_size
    logical_rank = dist.get_rank() // tp_size
    reference_losses = _run_to_boundary(reference_engine, logical_rank, logical_world_size)
    folded_losses = _run_to_boundary(folded_engine, logical_rank, logical_world_size)
    torch.testing.assert_close(folded_losses, reference_losses, rtol=1e-5, atol=1e-5)

    reference = _reference_grads(reference_engine.module)
    folded = _folded_grads(folded_engine.module)
    assert folded.keys() == reference.keys(), sorted(set(folded) ^ set(reference))
    mismatches = _grad_mismatches(folded, reference)
    assert not mismatches, f"{len(mismatches)} of {len(reference)} gradients differ:\n" + "\n".join(mismatches)

    # The optimizer computes the norm of the gradients it holds during step(); the reference norm
    # is taken directly from the full, data-parallel averaged reference gradients.
    reference_norm = torch.sqrt(sum(grad.double().square().sum() for grad in reference.values())).item()
    folded_engine.step()
    folded_norm = float(folded_engine.get_global_grad_norm())
    if dist.get_rank() == 0:
        print(f"GRAD_NORM zero_stage={zero_stage} zero_init={zero_init} tp={tp_size} ep={ep_size} "
              f"folded={folded_norm:.6f} reference={reference_norm:.6f}")
    # ZeRO-2 takes a different, approximate norm for MoE parameter groups: it averages each
    # expert group's norm over the expert-data-parallel group (_average_expert_grad_norms)
    # instead of adding the squared norms across expert-parallel ranks, with or without folding.
    if zero_stage == 3:
        assert folded_norm == pytest.approx(reference_norm, rel=1e-4), (folded_norm, reference_norm)


STAGES = [(2, False), (3, False), (3, True)]  # (ZeRO stage, model built under zero.Init)

# Folding only needs TP and EP to divide the world size: dense layers use TP x (world / TP), experts
# use EP x (world / EP), so TP x EP may be smaller or larger than the world size. The model has 4
# key-value heads (TP <= 4) and 8 routed experts.
SHAPES = [
    (4, 2, 2),  # TP x EP = world
    (4, 2, 4),  # each EP group spans both TP groups; expert data parallel 1
    (4, 4, 2),  # all ranks are one TP group (dense data parallel 1)
    (4, 4, 4),  # the EP group is the TP group
    (8, 2, 2),  # TP x EP < world: dense data parallel 4, expert data parallel 4
    (8, 2, 4),  # TP x EP = world
    (8, 4, 2),  # TP x EP = world with the larger TP
    (8, 4, 4),  # TP x EP = 2 x world
    (8, 2, 8),  # one expert per rank
]


@pytest.mark.parametrize("world_size, tp_size, ep_size", SHAPES)
@pytest.mark.parametrize("zero_stage, zero_init", STAGES)
class TestShardedAttentionFolding(DistributedTest):

    def test_grads_and_norm_match_reference(self, world_size, tp_size, ep_size, zero_stage, zero_init):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoEP folding runs on an accelerator")
        assert dist.get_world_size() == world_size
        _check_folded_matches_reference(tp_size=tp_size, ep_size=ep_size, zero_stage=zero_stage, zero_init=zero_init)
