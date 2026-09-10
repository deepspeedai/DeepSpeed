# Copyright (c) The DeepSpeed Contributors
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

import deepspeed.comm as dist
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
from unit.common import DistributedTest
from unit.util import torch_assert_close, torch_assert_equal


def _qwen35_classes(model_family):
    pytest.importorskip("fla")
    if model_family == "dense":
        configuration = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
        modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        return configuration.Qwen3_5TextConfig, modeling.Qwen3_5ForCausalLM, modeling
    configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
    modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    return configuration.Qwen3_5MoeTextConfig, modeling.Qwen3_5MoeForCausalLM, modeling


def _make_model(device, model_family="dense"):
    Qwen3_5TextConfig, Qwen3_5ForCausalLM, modeling = _qwen35_classes(model_family)
    config_kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=256,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
    )
    if model_family == "dense":
        config_kwargs["intermediate_size"] = 128
    else:
        config_kwargs.update(
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts_per_tok=2,
            num_experts=4,
        )
    config = Qwen3_5TextConfig(**config_kwargs)
    config._attn_implementation = "flash_attention_2"
    torch.manual_seed(1234)
    return Qwen3_5ForCausalLM(config).to(device=device, dtype=torch.bfloat16).train(), modeling


def _run_unmodified_qwen_with_fla(model, modeling, input_ids, position_ids, packed):
    from fla.modules.conv import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    original_conv = modeling.causal_conv1d_fn
    original_chunk = modeling.torch_chunk_gated_delta_rule

    def reference_conv(hidden_states, weight, bias=None, activation=None, **kwargs):
        output, _ = causal_conv1d(
            hidden_states.transpose(1, 2),
            weight,
            bias,
            activation=activation,
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
            seq_idx=kwargs.get("seq_idx"),
        )
        return output.transpose(1, 2)

    modeling.causal_conv1d_fn = reference_conv
    modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    try:
        kwargs = {}
        if packed:
            cu_seqlens = torch.tensor([0, 80, 128], device=input_ids.device, dtype=torch.int32)
            kwargs.update(
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=80,
                max_length_k=80,
                seq_idx=torch.cat((torch.zeros(80), torch.ones(48))).to(device=input_ids.device,
                                                                        dtype=torch.int32).unsqueeze(0),
            )
        return model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        ).logits
    finally:
        modeling.causal_conv1d_fn = original_conv
        modeling.torch_chunk_gated_delta_rule = original_chunk


def _selected_grads(model):
    return (
        model.model.layers[0].linear_attn.in_proj_qkv.weight.grad.detach().float().clone(),
        model.model.layers[-1].self_attn.q_proj.weight.grad.detach().float().clone(),
        model.lm_head.weight.grad.detach().float().clone(),
    )


class TestUlyssesSPQwen35LinearAttention(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("model_family", ["dense", "moe"])
    @pytest.mark.parametrize("packed", [False, True])
    def test_output_and_gradient_equivalence(self, packed, model_family):
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        world_size = dist.get_world_size()

        generator = torch.Generator(device="cpu").manual_seed(9876 + int(packed))
        input_ids = torch.randint(2, 128, (1, 128), generator=generator)
        position_ids = (torch.cat(
            (torch.arange(80), torch.arange(48))).unsqueeze(0) if packed else torch.arange(128).unsqueeze(0))

        baseline, baseline_modeling = _make_model(device, model_family)
        baseline_logits = _run_unmodified_qwen_with_fla(
            baseline,
            baseline_modeling,
            input_ids.to(device),
            position_ids.to(device),
            packed,
        )
        baseline_logits.float().square().mean().backward()
        baseline_grads = _selected_grads(baseline)

        candidate, _ = _make_model(device, model_family)
        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=candidate,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=world_size,
            micro_batch_size=1,
            seq_length=128,
            seq_length_is_variable=False,
        )
        local_input_ids = input_ids.chunk(world_size, dim=1)[rank].to(device)
        local_position_ids = position_ids.chunk(world_size, dim=1)[rank].to(device)
        candidate_logits = candidate(
            input_ids=local_input_ids,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        (candidate_logits.float().square().sum() / baseline_logits.numel()).backward()
        candidate_grads = _selected_grads(candidate)
        for gradient in candidate_grads:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)

        expected_logits = baseline_logits.chunk(world_size, dim=1)[rank]
        torch_assert_close(expected_logits, candidate_logits, atol=2e-2, rtol=2e-2)
        for expected_gradient, candidate_gradient in zip(baseline_grads, candidate_grads):
            torch_assert_close(expected_gradient, candidate_gradient, atol=2e-2, rtol=2e-2)

        UlyssesSPAttentionHF.unregister_from_transformers("flash_attention_2")

    def test_packed_documents_do_not_leak_across_rank_boundary(self):
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        position_ids = torch.cat((torch.arange(80), torch.arange(48))).unsqueeze(0)
        generator = torch.Generator(device="cpu").manual_seed(1122)
        first_input = torch.randint(2, 128, (1, 128), generator=generator)
        second_input = first_input.clone()
        second_input[:, :80] = torch.randint(2, 128, (1, 80), generator=generator)

        candidate, _ = _make_model(device)
        candidate.eval()
        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=candidate,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=self.world_size,
            micro_batch_size=1,
            seq_length=128,
            seq_length_is_variable=False,
        )
        local_position_ids = position_ids.chunk(self.world_size, dim=1)[rank].to(device)
        with torch.no_grad():
            first_logits = candidate(
                input_ids=first_input.chunk(self.world_size, dim=1)[rank].to(device),
                position_ids=local_position_ids,
                use_cache=False,
            ).logits
            second_logits = candidate(
                input_ids=second_input.chunk(self.world_size, dim=1)[rank].to(device),
                position_ids=local_position_ids,
                use_cache=False,
            ).logits

        if rank == 1:
            # The second document begins at global token 80, i.e. local token 16 on rank 1.
            torch_assert_close(first_logits[:, 16:], second_logits[:, 16:], atol=1e-5, rtol=1e-5)
        UlyssesSPAttentionHF.unregister_from_transformers("flash_attention_2")

    def test_disable_in_eval_uses_unsharded_original_forward(self):
        rank = dist.get_rank()
        device = torch.device("cuda", rank)
        generator = torch.Generator(device="cpu").manual_seed(4567)
        input_ids = torch.randint(2, 128, (1, 32), generator=generator).to(device)
        position_ids = torch.arange(32, device=device).unsqueeze(0)
        baseline, _ = _make_model(device)
        candidate, _ = _make_model(device)
        baseline.eval()
        candidate.eval()

        with torch.no_grad():
            expected = baseline(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits
        UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=candidate,
            core_attn_implementation="flash_attention_2",
            sequence_parallel_size=self.world_size,
            micro_batch_size=1,
            seq_length_is_variable=True,
            disable_in_eval=True,
        )
        with torch.no_grad():
            actual = candidate(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits

        torch_assert_equal(expected, actual)
        UlyssesSPAttentionHF.unregister_from_transformers("flash_attention_2")
