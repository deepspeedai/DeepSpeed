# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
import gc
import weakref

import pytest
import torch
import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import groups
from unit.common import DistributedTest


def skip_on_device():
    if get_accelerator().device_name() == "xpu":
        pytest.skip("XPU requires a higher version for test")


def build_qwen3_model():
    from transformers import AutoModelForCausalLM, Qwen3Config

    # Construct locally so the test does not download model configuration or weights. q_norm/k_norm act on the
    # sharded heads and the decoder norms on the split sequence, so both gradient reductions are exercised.
    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return config, AutoModelForCausalLM.from_config(config)


# plain: separate collectives; overlap: one autograd node per block with the backward reduce-scatter overlapped.
MODES = {
    "plain": {
        "sequence_sharding_overlap": False
    },
    "overlap": {
        "sequence_sharding_overlap": True
    },
}


def build_config(zero_stage, mode):
    return {
        "train_micro_batch_size_per_gpu": 2,
        "tensor_parallel": {
            "autotp_size": 2,
            "sequence_sharding": True,
            **MODES[mode],
        },
        # Plain SGD keeps a weight difference proportional to the gradient difference; Adam's normalized update
        # turns rounding-level gradient differences into 1e-5 weight differences.
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1
            },
        },
        "zero_allow_untested_optimizer": True,
        # The reference does not clip; DeepSpeed clips the global gradient norm to 1.0 by default.
        "gradient_clipping": 0.0,
        "zero_optimization": {
            "stage": zero_stage
        },
    }


def tp_shared_input_ids(vocab_size, seq_len, step):
    generator = torch.Generator().manual_seed(step)
    input_ids = torch.randint(0, vocab_size, (2, seq_len), generator=generator)
    input_ids = input_ids.to(get_accelerator().current_device_name())
    dist.broadcast(input_ids,
                   src=groups.get_tensor_model_parallel_src_rank(),
                   group=groups.get_tensor_model_parallel_group())
    return input_ids


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("zero_stage", [1, 2, 3])
class TestAutoTPSequenceSharding(DistributedTest):
    world_size = 2

    def test_matches_single_device_training(self, zero_stage, mode, checkpointing):
        """AutoTP with sequence sharding trains like the unsharded model under plain PyTorch SGD, also when each
        decoder layer is recomputed in backward (the recomputation must save what the forward pass saved)."""
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        reference = copy.deepcopy(model).to(get_accelerator().current_device_name())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        if checkpointing:
            model.gradient_checkpointing_enable()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage, mode))
        # The decoder norms must run on this rank's half of the sequence, not on the whole of it.
        norm_lengths = []
        engine.module.model.layers[0].input_layernorm.register_forward_hook(
            lambda module, args, output: norm_lengths.append(args[0].shape[1]))

        # The first step learns which linears read each block input; later steps run them in one autograd node.
        for step in range(3):
            input_ids = tp_shared_input_ids(config.vocab_size, 16, step)
            loss = engine(input_ids=input_ids, labels=input_ids).loss
            engine.backward(loss)
            engine.step()

            reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
            reference_loss.backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()

            torch.testing.assert_close(loss.detach(), reference_loss.detach(), rtol=1e-5, atol=1e-5)

        # Recomputation calls the norm again, so with checkpointing it runs more than once per step.
        assert len(norm_lengths) >= 3 and set(norm_lengths) == {16 // self.world_size}, norm_lengths
        if zero_stage < 3:
            # The decoder norms see half of the sequence on each rank; their summed gradient keeps them equal to
            # the reference after the updates.
            norm = engine.module.model.layers[0].input_layernorm.weight
            reference_norm = reference.model.layers[0].input_layernorm.weight
            torch.testing.assert_close(norm.detach(), reference_norm.detach(), rtol=1e-5, atol=1e-6)


class TestAutoTPSequenceShardingTwoForwards(DistributedTest):
    """Checkpointing recomputes each forward pass's blocks during backward, and a recomputation must take the path its
    forward took. With two forward passes before the first backward (a contrastive or DPO loss), the second pass must
    not start running the blocks' linears in one autograd node before the first pass has been recomputed."""
    world_size = 2

    def test_two_forwards_before_one_backward(self):
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        reference = copy.deepcopy(model).to(get_accelerator().current_device_name())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        model.gradient_checkpointing_enable()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))

        # The first step learns which linears read each block input; the second runs them in one autograd node.
        for step in range(2):
            first = tp_shared_input_ids(config.vocab_size, 16, 2 * step)
            second = tp_shared_input_ids(config.vocab_size, 16, 2 * step + 1)
            loss = engine(input_ids=first, labels=first).loss + engine(input_ids=second, labels=second).loss
            engine.backward(loss)
            engine.step()

            reference_loss = (reference(input_ids=first, labels=first).loss +
                              reference(input_ids=second, labels=second).loss)
            reference_loss.backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()

            torch.testing.assert_close(loss.detach(), reference_loss.detach(), rtol=1e-5, atol=1e-5)


ATTENTION_ONLY_PARTITION = {
    "use_default_specs":
    False,
    "layer_specs": [
        {
            "patterns": [r".*\.self_attn\.o_proj\.weight$"],
            "partition_type": "row"
        },
        {
            "patterns": [r".*\.self_attn\.[qkv]_proj\.weight$"],
            "partition_type": "column"
        },
    ],
}


@pytest.mark.parametrize("zero_stage", [1, 3])
class TestAutoTPSequenceShardingReplicatedMLP(DistributedTest):
    """With only attention sharded, the MLP linears stay replicated inside the sequence-sharded decoder layers. Each
    rank computes their gradients from its own tokens, so the gradients must be summed over TP, also when ZeRO-3
    replaces F.linear and the layers are recomputed in backward."""
    world_size = 2

    def test_matches_single_device_training(self, zero_stage):
        skip_on_device()
        pytest.importorskip("transformers")
        from deepspeed.module_inject.layers import TensorParallel_Layer

        config, model = build_qwen3_model()
        reference = copy.deepcopy(model).to(get_accelerator().current_device_name())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        model.gradient_checkpointing_enable()
        ds_config = build_config(zero_stage, "overlap")
        ds_config["tensor_parallel"]["partition_config"] = ATTENTION_ONLY_PARTITION
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
        assert not isinstance(engine.module.model.layers[0].mlp.down_proj, TensorParallel_Layer)

        for step in range(3):
            input_ids = tp_shared_input_ids(config.vocab_size, 16, step)
            loss = engine(input_ids=input_ids, labels=input_ids).loss
            engine.backward(loss)
            engine.step()

            reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
            reference_loss.backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()

            torch.testing.assert_close(loss.detach(), reference_loss.detach(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("zero_stage", [1, 3])
class TestAutoTPSequenceShardingUnreadColumn(DistributedTest):
    """A column-parallel linear inside a block that the block never calls with its input (MLA attention that reads
    kv_b_proj.weight directly, for example) must not keep the gathered block input alive after the step: one copy
    per decoder layer of the whole sequence's hidden states."""
    world_size = 2

    def test_gathered_input_is_freed(self, zero_stage):
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        for layer in model.model.layers:
            layer.self_attn.unused_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            layer.self_attn.unused_proj.weight.requires_grad_(False)
        ds_config = build_config(zero_stage, "overlap")
        partition = copy.deepcopy(ATTENTION_ONLY_PARTITION)
        partition["layer_specs"][1]["patterns"].append(r".*\.self_attn\.unused_proj\.weight$")
        ds_config["tensor_parallel"]["partition_config"] = partition
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

        gathered = []
        lengths = []

        def keep_gathered_input(module, args, kwargs):
            gathered.append(weakref.ref(kwargs["hidden_states"]))
            lengths.append(kwargs["hidden_states"].shape[1])

        for layer in engine.module.model.layers:
            # Registered after sequence sharding's own pre-hook, so it sees the gathered input.
            layer.self_attn.register_forward_pre_hook(keep_gathered_input, with_kwargs=True)

        input_ids = tp_shared_input_ids(config.vocab_size, 16, 0)
        loss = engine(input_ids=input_ids, labels=input_ids).loss
        engine.backward(loss)
        engine.step()
        del loss
        gc.collect()

        # The whole sequence: the hook watched the gathered input, not this rank's half.
        assert lengths == [16] * config.num_hidden_layers, lengths
        alive = [ref() is not None for ref in gathered]
        assert not any(alive), alive


class TestAutoTPSequenceShardingWeightHooks(DistributedTest):
    """A column-parallel weight may carry a tensor hook (AutoTP sums GPTBigCode's shared key/value rows in one).
    PyTorch stops calling such a hook once a custom autograd function saves the weight inside non-reentrant activation
    checkpointing, so in the overlap mode the hook must still run on every step."""
    world_size = 2

    def test_column_weight_hooks_run_every_step(self):
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        model.gradient_checkpointing_enable()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))
        calls = []
        for layer in engine.module.model.layers:
            layer.self_attn.q_proj.weight.register_hook(lambda grad: calls.append(grad.shape))

        steps = 3
        for step in range(steps):
            input_ids = tp_shared_input_ids(config.vocab_size, 16, step)
            loss = engine(input_ids=input_ids, labels=input_ids).loss
            engine.backward(loss)
            engine.step()

        # One call per decoder layer and step, also after the first step, from which the overlap mode would run the
        # block's linears in one autograd node.
        assert len(calls) == steps * config.num_hidden_layers, calls


class _ScaleSavingWeight(torch.autograd.Function):
    """``x * weight`` with the weight saved for backward, as fused norm kernels do."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return x * weight

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        return grad * weight, (grad * x).reshape(-1, x.shape[-1]).sum(0)


class _FusedStyleRMSNorm(torch.nn.Module):
    """RMSNorm that applies its weight through a custom autograd function, like a fused kernel."""

    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normed = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return _ScaleSavingWeight.apply(normed, self.weight)


def build_glm4_moe_model():
    from transformers import AutoModelForCausalLM, Glm4MoeConfig

    # One dense decoder layer, then one MoE layer for AutoEP to replace.
    config = Glm4MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        use_cache=False,
    )
    torch.manual_seed(0)
    return config, AutoModelForCausalLM.from_config(config)


def build_opt_model():
    from transformers import AutoModelForCausalLM, OPTConfig

    # OPT keeps fc1 and fc2 directly in the decoder layer, next to the self_attn module.
    config = OPTConfig(vocab_size=128,
                       hidden_size=64,
                       ffn_dim=128,
                       num_hidden_layers=2,
                       num_attention_heads=4,
                       max_position_embeddings=128,
                       word_embed_proj_dim=64)
    torch.manual_seed(0)
    return config, AutoModelForCausalLM.from_config(config)


OPT_PARTITION = {
    "use_default_specs":
    False,
    "layer_specs": [
        {
            "patterns": [r".*\.self_attn\.out_proj\.weight$", r".*\.fc2\.weight$"],
            "partition_type": "row"
        },
        {
            "patterns": [r".*\.self_attn\.[qkv]_proj\.weight$", r".*\.fc1\.weight$"],
            "partition_type": "column"
        },
    ],
}


class TestAutoTPSequenceShardingRejects(DistributedTest):
    world_size = 2

    def test_sequence_not_divisible_by_tp(self):
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))
        input_ids = tp_shared_input_ids(config.vocab_size, 15, 0)
        # pytest.raises inside a DistributedTest worker hangs the run when nothing is raised.
        try:
            engine(input_ids=input_ids, labels=input_ids)
            raised = False
        except ValueError as error:
            raised = "not divisible by the tensor-parallel size" in str(error)
        assert raised

    def test_autoep_moe_layers(self):
        """AutoEP + AutoTP folding expects every TP rank to hold the whole sequence. With sequence sharding each rank
        holds other tokens, so the MoE outputs would be wrong; initialization must refuse the combination."""
        skip_on_device()
        pytest.importorskip("transformers")
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoEP folding runs on an accelerator")

        config, model = build_glm4_moe_model()
        ds_config = build_config(zero_stage=1, mode="overlap")
        ds_config["tensor_parallel"]["partition_config"] = ATTENTION_ONLY_PARTITION
        ds_config["expert_parallel"] = {
            "enabled": True,
            "autoep_size": 2,
            "preset_model": "deepseek_v3",
            "use_grouped_mm": False,
        }
        try:
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
            raised = False
        except NotImplementedError as error:
            raised = "AutoEP" in str(error)
        assert raised

    def test_row_parallel_linear_in_decoder_layer(self):
        """OPT's fc2 is a row-parallel linear directly in the decoder layer. The whole layer would become a block: it
        would gather its input a second time around the attention block and reduce-scatter before the residual add,
        so initialization must refuse the layout."""
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_opt_model()
        ds_config = build_config(zero_stage=1, mode="overlap")
        ds_config["tensor_parallel"]["partition_config"] = OPT_PARTITION
        try:
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
            raised = False
        except ValueError as error:
            raised = "model.decoder.layers.0" in str(error)
        assert raised

    def test_routed_moe_layers(self):
        """A routed MoE layer calls only the experts its tokens chose. With sequence sharding each rank holds other
        tokens, so ranks could use different experts and a gradient sum on some ranks only would hang;
        initialization must refuse the layer."""
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_glm4_moe_model()
        ds_config = build_config(zero_stage=1, mode="overlap")
        ds_config["tensor_parallel"]["partition_config"] = ATTENTION_ONLY_PARTITION
        try:
            deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
            raised = False
        except NotImplementedError as error:
            raised = "model.layers.1.mlp" in str(error)
        assert raised

    def test_skipped_gradient_sum(self):
        """PyTorch skips the tensor hooks of a weight that a custom autograd function saves inside non-reentrant
        activation checkpointing. At ZeRO-1 the decoder norms' gradient sum runs in such a hook, so backward must
        fail instead of leaving each rank with the gradient of its own tokens."""
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        for layer in model.model.layers:
            layer.post_attention_layernorm = _FusedStyleRMSNorm(layer.post_attention_layernorm)
        model.gradient_checkpointing_enable()
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))
        input_ids = tp_shared_input_ids(config.vocab_size, 16, 0)
        loss = engine(input_ids=input_ids, labels=input_ids).loss
        try:
            engine.backward(loss)
            raised = False
        except RuntimeError as error:
            raised = "post_attention_layernorm" in str(error)
        assert raised

    def test_unfrozen_after_initialize(self):
        """A replicated decoder-layer parameter frozen at initialization gets no gradient sum. Unfrozen later, it
        would train on the gradient of this rank's tokens only, so backward must refuse."""
        skip_on_device()
        pytest.importorskip("transformers")

        config, model = build_qwen3_model()
        norm_weight = model.model.layers[0].input_layernorm.weight
        norm_weight.requires_grad_(False)
        engine, _, _, _ = deepspeed.initialize(model=model,
                                               model_parameters=model.parameters(),
                                               config=build_config(zero_stage=1, mode="overlap"))
        norm_weight.requires_grad_(True)
        input_ids = tp_shared_input_ids(config.vocab_size, 16, 0)
        loss = engine(input_ids=input_ids, labels=input_ids).loss
        try:
            engine.backward(loss)
            raised = False
        except RuntimeError as error:
            raised = "model.layers.0.input_layernorm.weight" in str(error)
        assert raised
