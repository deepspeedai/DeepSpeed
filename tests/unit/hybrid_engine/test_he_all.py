# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import os
import weakref
import torch
import pytest
import deepspeed
from packaging.version import Version
from deepspeed.ops.op_builder import OpBuilder
from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.module_inject import ReplaceWithTensorSlicing
from deepspeed.module_inject.containers.gptneox import DS_GPTNEOXContainer, GPTNEOXLayerPolicy
from deepspeed.module_inject.containers.llama import DS_LLAMAContainer, LLAMALayerPolicy

from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, __version__)
from deepspeed.ops.op_builder import InferenceBuilder

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)

rocm_version = OpBuilder.installed_rocm_version()
if rocm_version != (0, 0):
    pytest.skip("skip inference tests on rocm for now", allow_module_level=True)


@pytest.mark.seq_inference
@pytest.mark.parametrize("batch_size", [1, 2], ids=["bsz=1", "bsz=2"])
@pytest.mark.parametrize("model_name", ["EleutherAI/gpt-neo-1.3B", "facebook/opt-1.3b"])
class TestHybridEngineTextGen(DistributedTest):
    world_size = 1

    def _generate(self, model, tokenizer, prompt):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        tokens = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
        for t in tokens:
            if torch.is_tensor(tokens[t]):
                tokens[t] = tokens[t].to(f'{get_accelerator().device_name()}:{local_rank}')
        output = model.generate(**tokens, do_sample=False, max_length=100)
        outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
        return outputs

    def get_model(self, model_name):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.dropout = 0.0
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config)
        model = model.half()
        model = model.to(f'{get_accelerator().device_name()}:{local_rank}')
        return model

    def get_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_prompt(self, batch_size):
        if batch_size == 1:
            prompt = ["Microsoft is in Washington"]
        elif batch_size == 2:
            prompt = ["DeepSpeed is", "Microsoft is in Washington"]
        else:
            raise NotImplementedError(f"batch_size {batch_size} not implemented")
        return prompt

    def test_correctness(self, batch_size, model_name):
        pytest.skip("skip test for now, will fix in follow-up PR")
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        prompt = self.get_prompt(batch_size)

        base_out = self._generate(model, tokenizer, prompt)

        ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
        model, *_ = deepspeed.initialize(model=model, config=ds_config)

        model.eval()
        ds1_out = self._generate(model, tokenizer, prompt)
        assert base_out == ds1_out, f"base_out: {base_out}, ds1_out: {ds1_out}"

        model.train()
        model.eval()
        ds2_out = self._generate(model, tokenizer, prompt)
        assert base_out == ds2_out

    def test_functionality(self, batch_size, model_name):
        model = self.get_model(model_name)
        tokenizer = self.get_tokenizer(model_name)
        prompt = self.get_prompt(batch_size)

        ds_config = {"train_batch_size": 1, "fp16": {"enabled": True}, "hybrid_engine": {"enabled": True}}
        model, *_ = deepspeed.initialize(model=model, config=ds_config)

        model.eval()
        ds1_out = self._generate(model, tokenizer, prompt)

        model.train()
        model.eval()
        ds2_out = self._generate(model, tokenizer, prompt)

        assert ds1_out == ds2_out, f"ds1_out: {ds1_out}, ds2_out: {ds2_out}"


class CaptureScores(LogitsProcessor):

    def __init__(self) -> None:
        self.scores: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        self.scores.append(scores.detach().clone())
        return scores


class FailGeneration(LogitsProcessor):

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Generation failed in a logits processor")


@pytest.mark.seq_inference
@pytest.mark.skipif(Version(__version__) >= Version("4.44"),
                    reason="These injected models use the legacy cache contract")
@pytest.mark.parametrize("model_type", ["bloom", "opt"])
@pytest.mark.parametrize("world_size", [1, 2])
class TestHybridEngineTensorParallel(DistributedTest):
    world_size = 2

    def test_generation_preserves_training_weights(self, model_type: str, world_size: int) -> None:
        torch.manual_seed(1234)
        config = AutoConfig.for_model(model_type=model_type,
                                      vocab_size=128,
                                      hidden_size=128,
                                      num_hidden_layers=1,
                                      num_attention_heads=4,
                                      ffn_dim=512,
                                      max_position_embeddings=64,
                                      dropout=0.0,
                                      hidden_dropout=0.0,
                                      attention_dropout=0.0,
                                      activation_dropout=0.0,
                                      bos_token_id=1,
                                      eos_token_id=127,
                                      pad_token_id=0)
        model = AutoModelForCausalLM.from_config(config=config, attn_implementation="eager")
        oracle = copy.deepcopy(model).to(device=get_accelerator().current_device_name(), dtype=torch.float16).eval()
        oracle_parameters: dict[str, torch.nn.Parameter] = dict(oracle.named_parameters())
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               optimizer=optimizer,
                                               config={
                                                   "train_micro_batch_size_per_gpu": 1,
                                                   "fp16": {
                                                       "enabled": True,
                                                       "loss_scale": 1.0
                                                   },
                                                   "zero_optimization": {
                                                       "stage": 3,
                                                       "reduce_bucket_size": 1000,
                                                       "stage3_prefetch_bucket_size": 1000,
                                                       "stage3_param_persistence_threshold": 0,
                                                   },
                                                   "hybrid_engine": {
                                                       "enabled": True,
                                                       "inference_tp_size": world_size,
                                                       "max_out_tokens": 16,
                                                       "pin_parameters": True,
                                                       "tp_gather_partition_size": 1,
                                                       "release_inference_cache": False,
                                                   },
                                               })
        rank = deepspeed.comm.get_rank()
        training_parameters: set[int] = {id(parameter) for parameter in model.parameters()}
        for step, prompt in enumerate(([2, 3, 4, 5], [6, 7, 8], [2, 3, 4, 5])):
            inputs = torch.tensor([prompt], device=engine.device, dtype=torch.long) + rank * 8
            if step == 2:
                engine.train()
                loss = engine(input_ids=inputs, labels=inputs).loss
                assert torch.isfinite(loss)
                engine.backward(loss)
                engine.step()
                with deepspeed.zero.GatheredParameters(list(model.parameters())):
                    assert any(not torch.equal(parameter, oracle_parameters[name])
                               for name, parameter in model.named_parameters() if parameter.requires_grad)
                    oracle.load_state_dict(model.state_dict())
            engine.eval()
            oracle.eval()
            actual_scores = CaptureScores()
            expected_scores = CaptureScores()
            inference_parameters: list[weakref.ReferenceType[torch.Tensor]] = []

            def capture_inference_parameters(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
                for container in engine._inference_containers:
                    inference_parameters.extend(
                        weakref.ref(parameter) for parameter in container.module.parameters()
                        if id(parameter) not in training_parameters)

            if world_size > 1 and step == 0:
                with torch.no_grad(), model.register_forward_pre_hook(capture_inference_parameters):
                    with pytest.raises(RuntimeError, match="Generation failed in a logits processor"):
                        model.generate(input_ids=inputs, max_new_tokens=2, logits_processor=[FailGeneration()])
                assert inference_parameters
                assert all(parameter() is None for parameter in inference_parameters)
                inference_parameters.clear()

            with torch.no_grad(), model.register_forward_pre_hook(capture_inference_parameters):
                expected = oracle.generate(input_ids=inputs,
                                           do_sample=False,
                                           min_new_tokens=2,
                                           max_new_tokens=2,
                                           logits_processor=[expected_scores])
                actual = engine.module.generate(input_ids=inputs,
                                                do_sample=False,
                                                min_new_tokens=2,
                                                max_new_tokens=2,
                                                logits_processor=[actual_scores])
            if world_size > 1:
                assert inference_parameters
                assert all(parameter() is None
                           for parameter in inference_parameters), "Inference shards remain live after generation"
            torch.testing.assert_close(actual=actual, expected=expected, atol=0, rtol=0)
            assert len(actual_scores.scores) == len(expected_scores.scores) == 2
            for actual_score, expected_score in zip(actual_scores.scores, expected_scores.scores):
                if world_size > 1:
                    actual_score = actual_score[rank:rank + 1]
                torch.testing.assert_close(actual=actual_score, expected=expected_score, atol=1e-3, rtol=1e-2)
            with deepspeed.zero.GatheredParameters(list(model.parameters())):
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(actual=parameter, expected=oracle_parameters[name], atol=0, rtol=0)


@pytest.mark.seq_inference
@pytest.mark.parametrize("model_type", ["gpt_neox", "llama"])
@pytest.mark.parametrize("reversed_dim", [False, True])
class TestHybridEngineProjectionShards(DistributedTest):
    world_size = 2

    def test_projection_matches_unsharded_model(self, model_type: str, reversed_dim: bool) -> None:
        torch.manual_seed(1234)
        config = AutoConfig.for_model(model_type=model_type,
                                      vocab_size=128,
                                      hidden_size=128,
                                      intermediate_size=512,
                                      num_hidden_layers=1,
                                      num_attention_heads=4,
                                      num_key_value_heads=4)
        model = AutoModelForCausalLM.from_config(config=config, attn_implementation="eager")
        model = model.to(device=get_accelerator().current_device_name(), dtype=torch.float16).eval()
        if model_type == "gpt_neox":
            layer = model.gpt_neox.layers[0]
            container_type = DS_GPTNEOXContainer
            policy = GPTNEOXLayerPolicy(client_module=layer, inference=True)
        else:
            layer = model.model.layers[0]
            container_type = DS_LLAMAContainer
            policy = LLAMALayerPolicy(client_module=layer, inference=True)
        oracle = copy.deepcopy(layer)
        container = container_type(policy=policy,
                                   config=DeepSpeedInferenceConfig(dtype=torch.float16,
                                                                   set_empty_params=reversed_dim,
                                                                   transposed_mode=reversed_dim),
                                   model_config=config,
                                   layer_id=0,
                                   child=layer)
        group = deepspeed.comm.get_world_group()
        container.set_tensor_parallel_config(mp_size=2, mp_group=group)
        container.initialize_tensors(enable_training=True)
        container.create_ds_model_config()
        container.create_module()
        if reversed_dim:
            container.set_params_wo_copy(Z3_enabled=True)
            container.transform_for_inference()
        else:
            container.transpose()
        before: dict[str, torch.Tensor] = {name: value.detach().clone() for name, value in layer.named_parameters()}
        slicer = ReplaceWithTensorSlicing(mp_group=group,
                                          mp_size=2,
                                          out_dim=0 if reversed_dim else 1,
                                          in_dim=1 if reversed_dim else 0)
        inputs = torch.randn(1, 4, 128, device=get_accelerator().current_device_name(), dtype=torch.float16)
        rank = deepspeed.comm.get_rank()
        for _ in range(2):
            container.apply_tensor_parallelism(mp_replace=slicer, reversed_dim=reversed_dim)
            with torch.no_grad():
                if model_type == "gpt_neox":
                    attention = container.module.attention
                    actual = torch.nn.functional.linear(
                        input=inputs,
                        weight=attention.attn_qkvw if reversed_dim else attention.attn_qkvw.t(),
                        bias=attention.attn_qkvb)
                    expected = oracle.attention.query_key_value(inputs).view(1, 4, 4, 3, 32)
                    expected = expected[:, :, rank * 2:(rank + 1) * 2].permute(0, 1, 3, 2, 4).reshape(1, 4, 192)
                else:
                    mlp = container.module.mlp
                    if reversed_dim:
                        up = torch.nn.functional.linear(input=inputs, weight=mlp.inter_up_w)
                        gate = torch.nn.functional.linear(input=inputs, weight=mlp.inter_gate_w)
                    else:
                        up, gate = torch.nn.functional.linear(input=inputs, weight=mlp.inter_w.t()).chunk(2, dim=-1)
                    actual = torch.nn.functional.linear(input=up * torch.nn.functional.silu(gate),
                                                        weight=mlp.output_w if reversed_dim else mlp.output_w.t())
                    deepspeed.comm.all_reduce(actual)
                    expected = oracle.mlp(inputs)
                torch.testing.assert_close(actual=actual, expected=expected, atol=1e-3, rtol=1e-2)
                for name, parameter in layer.named_parameters():
                    torch.testing.assert_close(actual=parameter, expected=before[name], atol=0, rtol=0)
