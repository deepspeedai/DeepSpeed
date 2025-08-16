# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed
import pytest
import gc
import random
from unit.common import DistributedTest
from unit.simple_model import SimplePRMoEModel, SimpleMoEModel, sequence_dataloader
import deepspeed.comm as dist
from deepspeed import get_accelerator
from deepspeed.moe.sharded_moe import top1gating, topkgating
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer, is_moe_param
from deepspeed.utils.torch import required_torch_version


@pytest.mark.parametrize("zero_stage", [0, 1, 2])
class TestSimpleMoE(DistributedTest):
    world_size = 2

    def test(self, zero_stage):
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        # should automatically create moe param groups in deepspeed backend
        hidden_dim = 16
        model = SimpleMoEModel(hidden_dim=hidden_dim, ep_size=1)
        model, optimizer, _, _ = deepspeed.initialize(config=config_dict, model=model)
        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device,
                                          dtype=torch.float16)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("zero_stage", [0, 1, 2])
@pytest.mark.parametrize("use_residual", [True, False])
class TestMoE(DistributedTest):
    world_size = 4

    def test(self, ep_size, zero_stage, use_residual):
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 16

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimpleMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        param_group = {'params': [p for p in model.parameters()], 'name': 'random-unique-name'}
        params = split_params_into_different_moe_groups_for_optimizer(param_group)
        optimizer = torch.optim.AdamW(params=params)
        model, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      dist_init_required=False)
        #dist_init_required=False -- parameterize to True/False?

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device,
                                          dtype=torch.float16)

        def strict_average_tensor(tensor, communication_data_type: torch.dtype):
            process_group = optimizer.dp_process_group
            curr_size = 0
            pg_offsets = []

            ipg_bucket = optimizer.ipg_buckets[communication_data_type]
            for i, param_idx, param_id in ipg_bucket.params:
                param = optimizer.bit16_groups[i][param_idx]
                process_group = optimizer.dp_process_group
                if ipg_bucket.has_moe_params:
                    process_group = optimizer.expert_dp_process_group[param.group_name] if is_moe_param(
                        param) else optimizer.dp_process_group
                partition_ids = optimizer.param_to_partition_ids[i][param_id]
                # Get all partition ids + their offsets
                partition_offsets = []
                for partition_id in partition_ids:
                    offset = optimizer.grad_start_offset[i][partition_id][param_id]
                    partition_offsets.append(offset)
                partition_offsets.sort()
                # Calculate rank and offsets for grad slices
                for idx, offset in enumerate(partition_offsets):
                    # Calculate numel for grad slice depending on partition location
                    if idx == len(partition_offsets) - 1:
                        # Last partition_id uses its own offset
                        numel = param.numel() - offset
                    else:
                        # Set numel to next partition's offset
                        numel = partition_offsets[idx + 1] - offset
                    pg_offsets.append((curr_size, process_group))
                    curr_size += numel

            def strict_narrow(dim, start, length):
                lo, hi = 0, len(pg_offsets) - 1
                while lo < hi:
                    mi = lo + (hi - lo) // 2
                    if pg_offsets[mi][0] >= start:
                        hi = mi
                    else:
                        lo = mi + 1
                curr_slice, reduce_process_group = lo, pg_offsets[lo][1]
                while curr_slice < len(pg_offsets) and start + length > pg_offsets[curr_slice][0]:
                    assert reduce_process_group == pg_offsets[curr_slice][
                        1], "reduce process_group does not match the parameter's process_group"
                    curr_slice += 1
                return orig_narrow(dim, start, length)  # real call

            orig_narrow, tensor.narrow = tensor.narrow, strict_narrow
            type(optimizer).average_tensor(optimizer, tensor, communication_data_type)  # real call
            tensor.narrow = orig_narrow

        if "average_tensor" in dir(optimizer):
            optimizer.average_tensor = strict_average_tensor

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            gc.collect()  # Must do this or we get a memory leak in this test


@pytest.mark.parametrize("ep_size, use_residual", [(2, True), (2, False)])
class TestPRMoE(DistributedTest):
    world_size = 4

    def test(self, ep_size, use_residual):
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 8, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 16

        # E+D -- ep_size = 2
        # E only -- ep_size = 4
        model = SimplePRMoEModel(hidden_dim, ep_size=ep_size, use_residual=use_residual)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer,
                                              dist_init_required=False)

        data_loader = sequence_dataloader(model=model,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=model.device,
                                          dtype=torch.float16)

        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestTopk(DistributedTest):
    world_size = 2

    def test(self):
        device = get_accelerator().current_device_name()
        if dist.get_rank() == 0:
            logits = torch.rand(2, 2, device=device)
        elif dist.get_rank() == 1:
            logits = torch.rand(10, 2, device=device)

        output = top1gating(logits=logits,
                            capacity_factor=1,
                            min_capacity=0,
                            used_token=None,
                            noisy_gate_policy=None,
                            drop_tokens=False,
                            use_rts=True,
                            use_tutel=False)


class TestTopkGate(DistributedTest):

    def test(self):

        def check_equal(logits, cap, sparse_truth, res):
            m, n = logits.shape
            dispatch_mask_truth = torch.zeros(m, n, cap)
            i, j, k = sparse_truth.t()
            dispatch_mask_truth[i, j, k] = 1
            assert (torch.equal(dispatch_mask_truth, res))

        #s=4   e=4  topk=2   cap=2(s*topk/e)
        logits = torch.tensor([[0.11, 0.2, 0.1, 0.3], [0.3, 0.4, 0.11, 0.1], [0.11, 0.1, 0.6, 0.5],
                               [0.1, 0.11, 0.7, 0.8]])
        logits *= dist.get_rank() + 1
        probs_dispatch_res = topkgating(logits, 2, 1, min_capacity=1, drop_policy='probs')[2]
        probs_sec_sparse = torch.tensor([[0, 1, 0], [1, 0, 0], [1, 1, 1], [2, 2, 0], [2, 3, 0], [3, 2, 1], [3, 3, 1]])
        check_equal(logits, 2, probs_sec_sparse, probs_dispatch_res)

        position_sec_sparse = torch.tensor([[0, 1, 0], [0, 3, 0], [1, 0, 0], [1, 1, 1], [2, 2, 0], [2, 3, 1],
                                            [3, 2, 1]])
        position_dispatch_res = topkgating(logits, 2, 1, min_capacity=1, drop_policy='position')[2]
        check_equal(logits, 2, position_sec_sparse, position_dispatch_res)

        #s=4   e=6  topk=3   cap=2(s*topk/e)
        logits2 = torch.tensor([[0.5858, 0.4801, 0.6269, 0.5397, 0.9722, 0.7034],
                                [0.5445, 0.6332, 0.4519, 0.6308, 0.0519, 0.6450],
                                [0.4874, 0.8110, 0.7467, 0.8474, 0.0277, 0.3068],
                                [0.8570, 0.6714, 0.5310, 0.3274, 0.4836, 0.9892]])
        logits2 *= dist.get_rank() + 1

        #top3 full mask     #prob_mask          #postion_mask
        #0 0 1 0 1 1        #0 0 1 0 1 1        #0 0 1 0 1 1
        #0 1 0 1 0 1        #0 0 0 1 0 0        #0 1 0 1 0 1
        #0 1 1 1 0 0        #0 1 1 1 0 0        #0 1 1 1 0 0
        #1 1 0 0 0 1        #1 1 0 0 0 1        #1 0 0 0 0 0
        probs_dispatch_res = topkgating(logits2, 3, 1, min_capacity=1, drop_policy='probs')[2]
        probs_sec_sparse = torch.tensor([[0, 2, 0], [0, 4, 0], [0, 5, 0], [1, 3, 0], [2, 1, 0], [2, 2, 1], [2, 3, 1],
                                         [3, 0, 0], [3, 1, 1], [3, 5, 1]])
        check_equal(logits2, 2, probs_sec_sparse, probs_dispatch_res)

        position_sec_sparse = torch.tensor([[0, 2, 0], [0, 4, 0], [0, 5, 0], [1, 1, 0], [1, 3, 0], [1, 5, 1],
                                            [2, 1, 1], [2, 2, 1], [2, 3, 1], [3, 0, 0]])
        position_dispatch_res = topkgating(logits2, 3, 1, min_capacity=1, drop_policy='position')[2]
        check_equal(logits2, 2, position_sec_sparse, position_dispatch_res)

        #s=4   e=4  topk=2   drop_tokens=False
        logits3 = torch.tensor([[0.95, 0.85, 0.90, 0.80], [0.70, 0.65, 0.75, 0.60], [0.50, 0.55, 0.45, 0.40],
                                [0.35, 0.30, 0.25, 0.20]])
        logits3 *= dist.get_rank() + 1
        dispatch_res = topkgating(logits3, 2, 1, min_capacity=1, drop_tokens=False)[2]
        sec_sparse = torch.tensor([[0, 0, 0], [0, 2, 0], [1, 0, 1], [1, 2, 1], [2, 0, 2], [2, 1, 0], [3, 0, 3],
                                   [3, 1, 1]])
        check_equal(logits3, 4, sec_sparse, dispatch_res)


class TestExpertWeightGradWithZero(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("zero_stage", [0, 1, 2])
    def test(self, zero_stage):

        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        def seed_everything(seed=11):
            random.seed(seed)
            torch.manual_seed(seed)
            get_accelerator().manual_seed(seed)
            get_accelerator().manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        def get_state_dict_ep2(state_dict):
            """
            convert state_dict from EP=1 to EP=2
            """
            rank = int(deepspeed.comm.get_rank())
            ep_state_dict = dict()
            dst_sub_key = "deepspeed_moe.experts.deepspeed_experts.0"
            src_sub_key = f"deepspeed_moe.experts.deepspeed_experts.{rank}"
            for moe_layer in ["moe_1", "moe_2"]:
                for mlp_in_moe in [0, 1]:
                    dst_key = f"{moe_layer}.{dst_sub_key}.{mlp_in_moe}"
                    src_key = f"{moe_layer}.{src_sub_key}.{mlp_in_moe}"
                    ep_state_dict[f"{dst_key}.weight"] = state_dict[f"{src_key}.weight"].detach().clone()
                    ep_state_dict[f"{dst_key}.bias"] = state_dict[f"{src_key}.bias"].detach().clone()

            for key in state_dict.keys():
                if "deepspeed_moe.experts.deepspeed_experts" not in key:
                    ep_state_dict[key] = state_dict[key].detach().clone()
            return ep_state_dict

        def get_models(hidden_dim):
            model_ep1 = SimpleMoEModel(hidden_dim=hidden_dim, num_experts=2, ep_size=1, use_rts=False)
            model_ep2 = SimpleMoEModel(hidden_dim=hidden_dim, num_experts=2, ep_size=2, use_rts=False)

            state_dict_ep1 = model_ep1.state_dict()
            state_dict_ep2 = get_state_dict_ep2(state_dict_ep1)
            model_ep2.load_state_dict(state_dict_ep2)

            model_ep1, _, _, _ = deepspeed.initialize(config=config_dict, model=model_ep1)
            model_ep2, _, _, _ = deepspeed.initialize(config=config_dict, model=model_ep2)

            return model_ep1, model_ep2

        def extract_expert_grad(model, expert_id):

            def _get_weight_bias(experts):
                return ([deepspeed.utils.safe_get_full_grad(expert[0].weight)
                         for expert in experts][expert_id].detach().clone(),
                        [deepspeed.utils.safe_get_full_grad(expert[0].bias)
                         for expert in experts][expert_id].detach().clone(),
                        [deepspeed.utils.safe_get_full_grad(expert[1].weight)
                         for expert in experts][expert_id].detach().clone(),
                        [deepspeed.utils.safe_get_full_grad(expert[1].bias)
                         for expert in experts][expert_id].detach().clone())

            return (*_get_weight_bias(model.moe_1.deepspeed_moe.experts.deepspeed_experts),
                    *_get_weight_bias(model.moe_2.deepspeed_moe.experts.deepspeed_experts))

        seed_everything()

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.1,
                }
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }

        hidden_dim = 4
        total_samples = 2
        rank = deepspeed.comm.get_rank()
        model_ep1, model_ep2 = get_models(hidden_dim)

        data_loader = sequence_dataloader(model=model_ep1,
                                          total_samples=total_samples,
                                          hidden_dim=hidden_dim,
                                          device=model_ep1.device,
                                          dtype=torch.float32)
        expert_weight_grad_ep1 = []
        expert_weight_grad_ep2 = []
        for batch in data_loader:
            loss_ep1 = model_ep1(batch[0], batch[1])
            loss_ep2 = model_ep2(batch[0], batch[1])

            model_ep1.backward(loss_ep1)
            model_ep2.backward(loss_ep2)

            expert_weight_grad_ep1.extend(extract_expert_grad(model_ep1, rank))
            expert_weight_grad_ep2.extend(extract_expert_grad(model_ep2, 0))

            model_ep1.step()
            model_ep2.step()

        assert len(expert_weight_grad_ep1) == len(expert_weight_grad_ep2)
        for grad_from_ep1, grad_from_ep2 in zip(expert_weight_grad_ep1, expert_weight_grad_ep2):
            assert torch.allclose(grad_from_ep1, grad_from_ep2, atol=0, rtol=1e-4)
