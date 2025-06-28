# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
UlyssesPlus: UlyssesSPHF tests
"""

from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF, UlyssesSPDataLoaderAdapter
from deepspeed.runtime.utils import move_to_device
from deepspeed.utils import groups
from deepspeed.utils import safe_get_full_grad
from torch import tensor
from transformers import AutoModelForCausalLM
from unit.common import DistributedTest, preferred_dtype
from unit.util import torch_assert_equal, torch_assert_close, torch_assert_dicts_of_tensors_equal
import deepspeed
import deepspeed.comm as dist
import pytest
import torch


def get_grad(param, zero_stage):
    return safe_get_full_grad(param)
    # z1 now has contiguous_gradients enabled by default so `param.grad is None` even under z1
    # if zero_stage == 1:
    #     return param.grad
    # else:
    #     return safe_get_full_grad(param)


@pytest.mark.parametrize("zero_stage", [1, 3])
class TestUlyssesSPHF(DistributedTest):
    world_size = 2

    def test_ulysses_sp_hf(self, zero_stage):
        model_name_or_path = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
        #model_name_or_path = 'Felladrin/Llama-160M-Chat-v1'
        max_length = 64
        sequence_parallel_size = self.world_size
        micro_batch_size = 1

        rank = dist.get_rank()

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "zero_optimization": {
                "stage": zero_stage,
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "sequence_parallel_size": sequence_parallel_size,
        }

        dtype = preferred_dtype()
        if dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "loss_scale": 1.0}

        # Part 1. Baseline: Setup
        def collate_fn(batch):
            input_ids, position_ids = batch[0]
            #print(f"{batch=}")
            return dict(input_ids=input_ids.unsqueeze(0),
                        position_ids=position_ids.unsqueeze(0),
                        labels=input_ids.unsqueeze(0))

        input_ids = tensor([[1, 10, 10, 10, 2, 2], [1, 20, 20, 20, 2, 2]], )
        position_ids = tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        ds = torch.utils.data.TensorDataset(input_ids, position_ids)

        # 1. Baseline: DataLoader calibration
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        batch_a = next(iter(dl_a))
        #print(f"{rank=} {batch_a=}")
        expected_batch_a = {
            'input_ids': tensor([[1, 10, 10, 10, 2, 2]]),
            'position_ids': tensor([[0, 1, 2, 3, 4, 5]]),
            'labels': tensor([[1, 10, 10, 10, 2, 2]])
        }
        torch_assert_dicts_of_tensors_equal(batch_a, expected_batch_a)

        # 2. Baseline: Attention
        model_a = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model_a, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_a,
                                                model_parameters=model_a.parameters(),
                                                mpu=None)
        batch_a = move_to_device(batch_a, model_a.device)
        loss_a = model_a(**batch_a).loss
        model_a.backward(loss_a)
        #print(f"{loss_a=}")

        grad_a = get_grad(model_a.module.model.layers[0].self_attn.q_proj.weight, zero_stage)
        assert grad_a is not None
        #print(f"{grad_a}")

        # Part 2. Ulysses: Setup
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=model_name_or_path,
            core_attn_implementation="sdpa",
            sequence_parallel_size=sequence_parallel_size,
            max_length=max_length,
            micro_batch_size=micro_batch_size,
            seq_length_is_variable=True,
        )

        model_b = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model_b, _, _, _ = deepspeed.initialize(config=config_dict,
                                                model=model_b,
                                                model_parameters=model_b.parameters(),
                                                mpu=mpu)

        # 3. Ulysses: UlyssesSPDataLoaderAdapter test
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()
        sp_rank = groups._get_sequence_parallel_rank()
        dl_a = torch.utils.data.DataLoader(ds, batch_size=micro_batch_size, collate_fn=collate_fn)
        dl_b = UlyssesSPDataLoaderAdapter(
            dl_a,
            sp_rank=sp_rank,
            sp_group=sp_group,
            sp_world_size=sp_world_size,
            device=model_b.device,
        )
        batch_b = next(iter(dl_b))

        expected_batch_b = [
            {
                'input_ids': tensor([[1, 10, 10]]),
                'position_ids': tensor([[0, 1, 2]]),
                'shift_labels': tensor([[10, 10, 10]]),
            },
            {
                'input_ids': tensor([[10, 2, 2]]),
                'position_ids': tensor([[3, 4, 5]]),
                'shift_labels': tensor([[2, 2, -100]]),
            },
        ]

        # here we expect each sample to be sharded in half, rank0 getting the first half and rank1 the other half
        #print(f"{sp_rank=} {batch_b=}")
        torch_assert_dicts_of_tensors_equal(batch_b, expected_batch_b[sp_rank])

        # 4. UlyssesSPAttentionHF test
        batch_b = move_to_device(batch_b, model_b.device)
        outputs = model_b(**batch_b)
        # HF doesn't calculate loss with shift_labels yet and requires us to do it manually (liger does that)
        shift_labels = batch_b["shift_labels"]
        loss_b = model_b.module.loss_function(
            logits=outputs.logits,
            labels=None,
            shift_labels=shift_labels,
            vocab_size=model_b.module.config.vocab_size,
        )
        # print(f"{sp_rank=} {loss_b=}")

        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss_b, group=sp_group)
        good_tokens = sum((shift_labels != -100).view(-1))
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss_b = total_loss / total_good_tokens
        # print(f"{sp_rank=} gathered {loss_b=}")
        model_b.backward(loss_b)

        grad_b = get_grad(model_b.module.model.layers[0].self_attn.q_proj.weight, zero_stage)
        assert grad_b is not None
        #print(f"{grad_b}")

        # compare loss of A (non-Ulysses Attention) and B (Ulyssses Attention)
        torch_assert_equal(loss_a, loss_b)

        # - we are feeding the exact same sample to each rank of A
        # - for B we feed half the sample to each rank, but in total it's the same sample as each rank of A sees
        # thus we expect very similar grads (but not exact)
        if zero_stage in [1, 2]:
            # possibly some issue with z1/z2 as it requires higher tolerance than z3?
            torch_assert_close(grad_a, grad_b, rtol=1.6e-02, atol=1e-03)
        else:
            torch_assert_close(grad_a, grad_b)
