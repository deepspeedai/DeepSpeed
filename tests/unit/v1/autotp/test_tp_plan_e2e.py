# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint.constants import (ORIGINAL_VOCAB_SIZE, PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
                                            TP_REPLICATED_PARAMETER_PATTERNS, UNIVERSAL_CHECKPOINT_INFO,
                                            VOCABULARY_PARAMETER_PATTERNS)
from deepspeed.utils import groups
from deepspeed.runtime.utils import is_model_parallel_parameter
from unit.common import DistributedTest, preferred_dtype


def skip_on_device():
    return


class TestTPPlanEndToEnd(DistributedTest):
    world_size = 2

    class SimpleHFModel(torch.nn.Module):

        class Block(torch.nn.Module):

            def __init__(self, hidden_size):
                super().__init__()
                self.q_proj = torch.nn.Linear(hidden_size, hidden_size * 2)
                self.o_proj = torch.nn.Linear(hidden_size * 2, hidden_size)

            def forward(self, x):
                return self.o_proj(self.q_proj(x))

        def __init__(self, hidden_size=64):
            super().__init__()
            self.hidden_size = hidden_size
            self.config = type(
                "Config",
                (),
                {"base_model_tp_plan": {
                    "*.q_proj": "colwise",
                    "*.o_proj": "rowwise",
                }},
            )()
            self.layers = torch.nn.ModuleList([self.Block(hidden_size)])

        def forward(self, x):
            return self.layers[0](x)

    def _setup_baseline_linears(self, model):
        torch_q = torch.nn.Linear(model.hidden_size, model.hidden_size * 2)
        torch_o = torch.nn.Linear(model.hidden_size * 2, model.hidden_size)
        torch_q.load_state_dict(model.layers[0].q_proj.state_dict(), strict=True)
        torch_o.load_state_dict(model.layers[0].o_proj.state_dict(), strict=True)

        if preferred_dtype() == torch.float16:
            torch_q = torch_q.half()
            torch_o = torch_o.half()
        elif preferred_dtype() == torch.bfloat16:
            torch_q = torch_q.bfloat16()
            torch_o = torch_o.bfloat16()

        device = get_accelerator().current_device_name()
        torch_q = torch_q.to(device)
        torch_o = torch_o.to(device)
        return torch_q, torch_o

    def _compare_tp_gradients(self, model, torch_q, torch_o, input_tensor, engine):

        def _get_grad(param):
            if param.grad is not None:
                return param.grad
            return getattr(param, "grad_accum", None)

        torch_q.zero_grad(set_to_none=True)
        torch_o.zero_grad(set_to_none=True)
        torch_q_out = torch_q(input_tensor)
        torch_o_out = torch_o(torch_q_out)
        torch_loss = torch_o_out.sum()
        torch_loss.backward()

        output = engine(input_tensor)
        loss = output.sum()
        engine.backward(loss)

        tp_rank = groups.get_tensor_model_parallel_rank()
        tp_size = engine.autotp_size()
        q_proj = model.layers[0].q_proj
        o_proj = model.layers[0].o_proj

        torch_q_grad = torch.chunk(torch_q.weight.grad, tp_size, dim=0)[tp_rank]
        torch_q_bias_grad = torch.chunk(torch_q.bias.grad, tp_size, dim=0)[tp_rank]
        torch_o_grad = torch.chunk(torch_o.weight.grad, tp_size, dim=1)[tp_rank]

        q_weight_grad = _get_grad(q_proj.weight)
        q_bias_grad = _get_grad(q_proj.bias) if q_proj.bias is not None else None
        o_weight_grad = _get_grad(o_proj.weight)

        torch.testing.assert_close(q_weight_grad, torch_q_grad, atol=2e-2, rtol=2e-2)
        if q_bias_grad is not None:
            torch.testing.assert_close(q_bias_grad, torch_q_bias_grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(o_weight_grad, torch_o_grad, atol=2e-2, rtol=2e-2)

    def _gather_and_compare_params(self, model, torch_q, torch_o, compare_values=True):
        q_proj = model.layers[0].q_proj
        o_proj = model.layers[0].o_proj

        original_shards = []
        for _, param in q_proj.named_parameters(recurse=False):
            if is_model_parallel_parameter(param):
                original_shards.append((param, param.data.detach().clone()))

        for _, param in o_proj.named_parameters(recurse=False):
            if is_model_parallel_parameter(param):
                original_shards.append((param, param.data.detach().clone()))

        for param, _ in original_shards:
            param.gather_params([param])

        if compare_values:
            torch.testing.assert_close(q_proj.weight, torch_q.weight, atol=2e-2, rtol=2e-2)
            if q_proj.bias is not None:
                torch.testing.assert_close(q_proj.bias, torch_q.bias, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(o_proj.weight, torch_o.weight, atol=2e-2, rtol=2e-2)
            if o_proj.bias is not None:
                torch.testing.assert_close(o_proj.bias, torch_o.bias, atol=2e-2, rtol=2e-2)

        for param, original in original_shards:
            param._tp_partition([param])
            torch.testing.assert_close(param.data, original, atol=2e-2, rtol=2e-2)

    def test_tp_plan_basic_training(self):
        skip_on_device()

        model = self.SimpleHFModel()
        if preferred_dtype() == torch.float16:
            model = model.half()
        elif preferred_dtype() == torch.bfloat16:
            model = model.bfloat16()

        torch_q, torch_o = self._setup_baseline_linears(model)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": 0
            },
            "steps_per_print": 1,
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        input_tensor = torch.randn(2, 4, 64, dtype=preferred_dtype()).to(get_accelerator().current_device_name())
        dist.broadcast(
            input_tensor,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        if preferred_dtype() == torch.float16:
            torch_q = torch_q.half()
            torch_o = torch_o.half()
        elif preferred_dtype() == torch.bfloat16:
            torch_q = torch_q.bfloat16()
            torch_o = torch_o.bfloat16()

        self._compare_tp_gradients(model, torch_q, torch_o, input_tensor, engine)

    def test_tp_plan_attaches_universal_checkpoint_info(self):
        model = self.SimpleHFModel()
        model.lm_head = torch.nn.Linear(model.hidden_size, 64)
        model.config.base_model_tp_plan["lm_head"] = "colwise_gather_output"
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "zero_optimization": {
                "stage": 0
            },
        }

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2
        assert model.lm_head.gather_output
        uc_info = getattr(model, UNIVERSAL_CHECKPOINT_INFO)
        row_parallel_patterns = uc_info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS]
        replicated_patterns = uc_info[TP_REPLICATED_PARAMETER_PATTERNS]
        assert r"^layers\.0\.o_proj\.weight$" in row_parallel_patterns
        assert r"^layers\.0\.q_proj\.weight$" not in row_parallel_patterns
        assert r"^layers\.0\.o_proj\.bias$" in replicated_patterns
        assert r"^layers\.0\.q_proj\.bias$" not in replicated_patterns
        assert r"^lm_head\.weight$" in uc_info[VOCABULARY_PARAMETER_PATTERNS]
        assert uc_info[ORIGINAL_VOCAB_SIZE] == 64

    def test_tp_plan_with_zero1(self):
        skip_on_device()

        model = self.SimpleHFModel()
        torch_q, torch_o = self._setup_baseline_linears(model)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": 1
            },
            "steps_per_print": 1,
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        for _ in range(1):
            input_tensor = torch.randn(2, 4, 64, dtype=preferred_dtype()).to(get_accelerator().current_device_name())
            dist.broadcast(
                input_tensor,
                src=groups.get_tensor_model_parallel_src_rank(),
                group=groups.get_tensor_model_parallel_group(),
            )
            self._gather_and_compare_params(model, torch_q, torch_o, compare_values=False)
            output = engine(input_tensor)
            loss = output.mean()
            engine.backward(loss)
            engine.step()

        for p in engine.parameters():
            assert not torch.isnan(p).any()

    def test_tp_plan_with_zero2(self):
        skip_on_device()

        model = self.SimpleHFModel()
        torch_q, torch_o = self._setup_baseline_linears(model)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "tensor_parallel": {
                "autotp_size": 2
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": 2
            },
            "steps_per_print": 1,
        }

        if preferred_dtype() == torch.float16:
            ds_config["fp16"] = {"enabled": True}
        elif preferred_dtype() == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}

        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        assert engine.autotp_size() == 2

        input_tensor = torch.randn(2, 4, 64, dtype=preferred_dtype()).to(get_accelerator().current_device_name())
        dist.broadcast(
            input_tensor,
            src=groups.get_tensor_model_parallel_src_rank(),
            group=groups.get_tensor_model_parallel_group(),
        )
        self._gather_and_compare_params(model, torch_q, torch_o, compare_values=False)
        output = engine(input_tensor)
        loss = output.mean()
        engine.backward(loss)
        engine.step()


class TestReplicatedGradAllreduce(DistributedTest):
    """Replicated parameters marked replicated_with_grad_allreduce must not drift across TP ranks.

    Qwen3's q_norm/k_norm normalize the locally sharded attention heads, so each rank computes only
    its shard's contribution to their gradient. Before the style was implemented the whole tp_plan
    was discarded and the partial gradients were silently left unsummed: training ran, the loss
    went down, and the ranks held different weights after the first optimizer step.
    """

    world_size = 2
    non_daemonic_procs = True

    def test_qwen3_norm_grads_match_across_ranks(self):
        if get_accelerator().device_name() == "cpu":
            import pytest
            pytest.skip("CPU does not support this test yet")
        import pytest
        transformers = pytest.importorskip("transformers")
        if not hasattr(transformers, "Qwen3ForCausalLM"):
            pytest.skip("transformers build has no Qwen3")

        config = transformers.Qwen3Config(vocab_size=256,
                                          hidden_size=64,
                                          intermediate_size=128,
                                          num_hidden_layers=2,
                                          num_attention_heads=8,
                                          num_key_value_heads=8,
                                          head_dim=8,
                                          max_position_embeddings=64,
                                          use_cache=False,
                                          tie_word_embeddings=False)
        config._attn_implementation = "sdpa"
        torch.manual_seed(42)
        model = transformers.Qwen3ForCausalLM(config)

        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-6
                }
            },
            "tensor_parallel": {
                "autotp_size": self.world_size
            },
            "zero_optimization": {
                "stage": 0
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        norm_params = [name for name, _ in engine.module.named_parameters() if "q_norm" in name or "k_norm" in name]
        assert norm_params, "the model should expose q_norm/k_norm parameters"
        registered = [
            name for name, param in engine.module.named_parameters()
            if getattr(param, "_ds_grad_allreduce_registered", False)
        ]
        assert set(norm_params) <= set(registered), \
            f"grad all-reduce hooks missing for {sorted(set(norm_params) - set(registered))}"

        device = torch.device(get_accelerator().current_device_name())
        torch.manual_seed(1234)
        input_ids = torch.randint(0, 256, (1, 16), device=device)
        logits = engine(input_ids=input_ids, use_cache=False).logits
        engine.backward(logits.float().pow(2).mean())

        tp_group = groups.get_tensor_model_parallel_group()
        tp_world = dist.get_world_size(group=tp_group)
        for name, param in engine.module.named_parameters():
            if name not in norm_params:
                continue
            gathered = [torch.empty_like(param.grad) for _ in range(tp_world)]
            dist.all_gather(gathered, param.grad.contiguous(), group=tp_group)
            for other in gathered[1:]:
                assert torch.allclose(gathered[0], other, atol=1e-6), \
                    f"gradient of {name} differs across tensor-parallel ranks"
