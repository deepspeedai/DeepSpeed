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
import os
from types import SimpleNamespace

import pytest
import torch

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.checkpoint import ds_to_universal
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.module_inject.auto_tp import AutoTP
from deepspeed.module_inject.layers import LinearAllreduce, LinearLayer, TensorParallel_Layer, VocabParallelLinear
from deepspeed.utils import groups, safe_get_full_fp32_param, safe_get_full_grad
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


def _folded_config(tp_size, ep_size, zero_stage, offload=False):
    config = _common_config()
    config["zero_optimization"] = {"stage": zero_stage, "overlap_comm": True, "reduce_scatter": True}
    if offload:
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}
        # SGD then steps the offloaded fp32 partitions on the CPU; DeepSpeedCPUAdam is not needed.
        config["zero_force_ds_cpu_optimizer"] = False
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


def _gather_expert_grads(layer, param, get=safe_get_full_grad):
    grad = get(param)
    assert grad is not None
    shards = [torch.empty_like(grad) for _ in range(dist.get_world_size(group=layer.ep_group))]
    dist.all_gather(shards, grad.contiguous(), group=layer.ep_group)
    return torch.cat(shards, dim=0)


def _folded_grads(model, get=safe_get_full_grad):
    """Every gradient of the folded model, rebuilt at the reference model's names and shapes.

    With ``get=safe_get_full_fp32_param`` it rebuilds the fp32 parameters the same way.
    """
    grads = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, AutoEPMoELayer):
            w1 = _gather_expert_grads(module, module.experts.w1, get)
            w2 = _gather_expert_grads(module, module.experts.w2, get)
            w3 = _gather_expert_grads(module, module.experts.w3, get)
            grads[f"{prefix}experts.gate_up_proj"] = torch.cat([w1, w3], dim=1)
            grads[f"{prefix}experts.down_proj"] = w2
            grads[f"{prefix}gate.weight"] = get(module.router.gate.weight)
            continue
        for name, param in module.named_parameters(recurse=False):
            if any(isinstance(owner, AutoEPMoELayer) for owner in _owners(model, module_name)):
                if ".shared_experts." not in f".{prefix}":
                    continue
            grad = get(param)
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


def _reference_params(model):
    return {name: param.detach().float().cpu() for name, param in model.named_parameters()}


def _assert_structure(model):
    """AutoTP shards the attention projections only; AutoEP owns every MoE block."""
    for name, module in model.named_modules():
        is_tp = isinstance(module, TensorParallel_Layer)
        is_attention_projection = ".self_attn." in f".{name}" and name.rsplit(
            ".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
        assert is_tp == is_attention_projection, f"{name}: tensor parallel {is_tp}"
    moe_layers = [module for module in model.modules() if isinstance(module, AutoEPMoELayer)]
    assert len(moe_layers) == model.config.num_hidden_layers - model.config.first_k_dense_replace


def _grad_mismatches(folded, reference, rounding=None):
    """Name, relative error and scale of every gradient that differs from the reference.

    ``rounding`` adds a per-element allowance for rounding that happened outside the computation
    being compared, such as recovering an update as ``after - before`` in fp32.
    """
    mismatches = []
    for name, want in reference.items():
        got = folded[name]
        bound = 1e-4 * want.abs() + 1e-5 * max(want.abs().max().item(), 1e-6)
        if rounding is not None:
            bound = bound + rounding[name]
        if ((got - want).abs() <= bound).all():
            continue
        want_norm_sq = want.square().sum().item()
        scale = got.mul(want).sum().item() / want_norm_sq if want_norm_sq else float("nan")
        relative = (got - want).norm().item() / max(want.norm().item(), 1e-12)
        mismatches.append(f"{name}: relative error {relative:.3e}, scale {scale:.4f}")
    return mismatches


def _check_folded_matches_reference(tp_size, ep_size, zero_stage, zero_init, offload=False, clip=None):
    """Compare a folded run with the unfolded reference: losses, gradients and the global norm.

    With ``clip`` both runs clip to that norm and take one optimizer step, and every parameter's
    update must match too; the clipped update depends on the global norm the optimizer computed.
    """
    torch.manual_seed(1234)
    config = _tiny_glm4_moe_config()
    reference_model = transformers.Glm4MoeForCausalLM(config).float()
    reference_state = copy.deepcopy(reference_model.state_dict())
    folded_config = _folded_config(tp_size, ep_size, zero_stage, offload=offload)
    reference_config = _reference_config()
    if clip is not None:
        # The clipped SGD step then moves the parameters by a vector of norm lr * clip = 1, so each
        # update stays far above the fp32 rounding of the parameter it is added to.
        for run_config in (folded_config, reference_config):
            run_config["gradient_clipping"] = clip
            run_config["optimizer"]["params"]["lr"] = 1.0 / clip

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

    reference_engine, _, _, _ = deepspeed.initialize(model=reference_model, config=reference_config)
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
    if clip is not None:
        reference_before = _reference_params(reference_engine.module)
        folded_before = _folded_grads(folded_engine.module, get=safe_get_full_fp32_param)
    folded_engine.step()
    folded_norm = float(folded_engine.get_global_grad_norm())
    if dist.get_rank() == 0:
        print(f"GRAD_NORM zero_stage={zero_stage} zero_init={zero_init} tp={tp_size} ep={ep_size} "
              f"offload={offload} folded={folded_norm:.6f} reference={reference_norm:.6f}")
    # ZeRO-2 takes a different, approximate norm for MoE parameter groups: it averages each
    # expert group's norm over the expert-data-parallel group (_average_expert_grad_norms)
    # instead of adding the squared norms across expert-parallel ranks, with or without folding.
    if zero_stage == 3:
        assert folded_norm == pytest.approx(reference_norm, rel=1e-4), (folded_norm, reference_norm)

    if clip is not None:
        assert reference_norm > clip, f"clip {clip} does not engage at norm {reference_norm}"
        reference_engine.step()
        reference_after = _reference_params(reference_engine.module)
        folded_after = _folded_grads(folded_engine.module, get=safe_get_full_fp32_param)
        reference_update = {name: reference_after[name] - reference_before[name] for name in reference_after}
        folded_update = {name: folded_after[name] - folded_before[name] for name in folded_after}
        # SGD stores fl(p - lr * g), so each recovered update carries up to half an ulp of the new
        # parameter; two exact steps can therefore differ by one ulp, i.e. eps * |p|.
        eps = torch.finfo(torch.float32).eps
        rounding = {name: 2 * eps * after.abs() for name, after in reference_after.items()}
        mismatches = _grad_mismatches(folded_update, reference_update, rounding)
        summary = f"{len(mismatches)} of {len(reference_update)} updates differ:\n"
        assert not mismatches, summary + "\n".join(mismatches)
        # The rounding allowance must not hide a wrong norm: a clip coefficient 0.1% off is still caught.
        scaled = {name: update * 1.001 for name, update in reference_update.items()}
        assert _grad_mismatches(scaled, reference_update, rounding), "update comparison cannot see a 0.1% error"


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


# Clipping norm below the gradient norm of the tiny model, so the step depends on the norm.
CLIP = 1e-2


@pytest.mark.parametrize("world_size, tp_size, ep_size", SHAPES[:4])
@pytest.mark.parametrize("zero_init", [False, True])
@pytest.mark.parametrize("offload", [False, True])
class TestFoldedZero3ClippedUpdate(DistributedTest):
    """ZeRO-3 folding applies the reference's clipped update, with the optimizer offloaded to the CPU or not.

    Offload keeps the fp32 partitions, their gradients and the norm on a different path, so the
    gradients, the norm and the clipped update are all compared with the unfolded reference.
    """

    def test_clipped_update_matches_reference(self, world_size, tp_size, ep_size, zero_init, offload):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoEP folding runs on an accelerator")
        assert dist.get_world_size() == world_size
        _check_folded_matches_reference(tp_size=tp_size,
                                        ep_size=ep_size,
                                        zero_stage=3,
                                        zero_init=zero_init,
                                        offload=offload,
                                        clip=CLIP)


class TestTiedOutputHead(DistributedTest):
    """Building the model under ``deepspeed.zero.Init`` must not change which layers AutoTP accepts.

    AutoTP cannot shard an output head whose weight is tied to the input embedding by rows, and says
    so. Under ``zero.Init`` AutoTP first gathers the partitioned head weight into a new parameter; if
    the embedding keeps the old one, the tie is gone, the head is sharded anyway, and the embedding and
    head train as two separate copies.
    """
    world_size = 4

    @pytest.mark.parametrize("zero_init", [False, True])
    def test_row_parallel_tied_head_is_refused(self, zero_init):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoEP folding runs on an accelerator")
        config = _tiny_glm4_moe_config()
        config.tie_word_embeddings = True
        folded_config = _folded_config(tp_size=2, ep_size=2, zero_stage=3)
        row_parallel_head = {"patterns": [r".*lm_head\.weight$"], "partition_type": "row"}
        folded_config["tensor_parallel"]["partition_config"]["layer_specs"] = ATTENTION_ONLY_TP_SPECS + [
            row_parallel_head
        ]
        if zero_init:
            with deepspeed.zero.Init(config_dict_or_path=folded_config):
                model = transformers.Glm4MoeForCausalLM(config)
        else:
            model = transformers.Glm4MoeForCausalLM(config).float()
        assert model.lm_head.weight is model.model.embed_tokens.weight

        # Not pytest.raises: when nothing is raised it raises pytest's Failed, a BaseException that the
        # DistributedTest pool worker does not report, so the test would hang instead of failing.
        try:
            deepspeed.initialize(model=model, config=folded_config)
        except NotImplementedError as error:
            assert "tied weight" in str(error), error
        else:
            raise AssertionError("AutoTP sharded an output head whose weight is tied to the input embedding")


class TestZero3FoldedCheckpoint(DistributedTest):
    """A ZeRO-3 checkpoint of a folded run restores the training state exactly.

    Train one step, save, load into an engine built the same way from a different random
    initialization, then train one more step on both: the losses and every parameter must agree.
    SGD momentum makes that step depend on the restored optimizer state. Universal Checkpoint
    conversion of such a checkpoint is not supported yet and must say so.
    """
    world_size = 4

    @pytest.mark.parametrize("zero_init", [False, True])
    def test_save_load_resumes_training(self, zero_init, tmpdir):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoEP folding runs on an accelerator")
        tp_size = 2
        config = _tiny_glm4_moe_config()
        folded_config = _folded_config(tp_size=tp_size, ep_size=2, zero_stage=3)
        folded_config["optimizer"]["params"]["momentum"] = 0.9

        def build_engine(seed):
            torch.manual_seed(seed)
            if zero_init:
                with deepspeed.zero.Init(config_dict_or_path=folded_config):
                    model = transformers.Glm4MoeForCausalLM(config)
            else:
                model = transformers.Glm4MoeForCausalLM(config).float()
            engine, _, _, _ = deepspeed.initialize(model=model, config=folded_config)
            return engine

        logical_world_size = dist.get_world_size() // tp_size
        logical_rank = dist.get_rank() // tp_size
        engine = build_engine(seed=1234)
        _run_to_boundary(engine, logical_rank, logical_world_size)
        engine.step()
        save_dir = str(tmpdir)
        engine.save_checkpoint(save_dir, tag="step1")

        resumed = build_engine(seed=999)
        resumed.load_checkpoint(save_dir, tag="step1")
        losses = _run_to_boundary(engine, logical_rank, logical_world_size)
        resumed_losses = _run_to_boundary(resumed, logical_rank, logical_world_size)
        engine.step()
        resumed.step()
        # Every rank gathers every parameter before any assertion: a rank that failed early would leave the
        # others waiting in the next gather.
        resumed_params = dict(resumed.module.named_parameters())
        differing = []
        for name, param in engine.module.named_parameters():
            want = safe_get_full_fp32_param(param)
            got = safe_get_full_fp32_param(resumed_params[name])
            if not torch.allclose(got, want, rtol=1e-6, atol=1e-7):
                differing.append(name)

        conversion_error = None
        args = SimpleNamespace(input_folder=os.path.join(save_dir, "step1"),
                               output_folder=os.path.join(save_dir, f"universal_rank{dist.get_rank()}"))
        try:
            ds_to_universal.main(args)
        except NotImplementedError as error:
            conversion_error = str(error)

        torch.testing.assert_close(resumed_losses, losses, rtol=1e-6, atol=1e-6)
        assert not differing, f"{len(differing)} parameters differ after resuming: {differing}"
        assert conversion_error is not None, "Universal conversion accepted a folded ZeRO-3 checkpoint"
        assert "folded ZeRO-3 checkpoint is not supported" in conversion_error, conversion_error


class _LlamaWithoutTPPlan(transformers.LlamaForCausalLM):
    # With no HuggingFace tensor-parallel plan, AutoTP falls back to its heuristic layer matching.
    _tp_plan = None


def _tiny_llama_config(tied):
    config = transformers.LlamaConfig(vocab_size=128,
                                      hidden_size=32,
                                      intermediate_size=64,
                                      num_hidden_layers=2,
                                      num_attention_heads=4,
                                      num_key_value_heads=4,
                                      head_dim=8,
                                      use_cache=False,
                                      tie_word_embeddings=tied)
    config.base_model_tp_plan = None
    config._attn_implementation = "eager"
    return config


def _zero_init_case(case):
    """Model class, model config, DeepSpeed config, TP size, and whether the caller builds the optimizer."""
    if case.startswith("folded"):
        config = _tiny_glm4_moe_config()
        ds_config = _folded_config(tp_size=2, ep_size=2, zero_stage=3)
        ds_config["optimizer"]["params"]["momentum"] = 0.9
        config.tie_word_embeddings = True
        ds_config["tensor_parallel"]["vocab_parallel_lm_head"] = True
        return transformers.Glm4MoeForCausalLM, config, ds_config, 2, False
    tied = case == "heuristic_vocab_parallel_head"
    ds_config = _common_config()
    ds_config.pop("optimizer")
    ds_config["zero_optimization"] = {"stage": 3}
    ds_config["tensor_parallel"] = {"autotp_size": 2}
    if tied:
        ds_config["tensor_parallel"]["vocab_parallel_lm_head"] = True
    return _LlamaWithoutTPPlan, _tiny_llama_config(tied), ds_config, 2, True


def _initialize(model, ds_config, client_optimizer):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) if client_optimizer else None
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=copy.deepcopy(ds_config))
    return engine


def _train_two_windows(engine, tp_size):
    logical_world_size = dist.get_world_size() // tp_size
    logical_rank = dist.get_rank() // tp_size
    losses = []
    for _ in range(2):
        losses += _run_to_boundary(engine, logical_rank, logical_world_size)
        engine.step()
    return losses


ZERO_INIT_CASES = [
    (4, "folded_vocab_parallel_head"),  # tied output head and embedding, vocabulary-parallel
    (2, "heuristic"),  # no partition plan: AutoTP's heuristic layer matching
    (2, "heuristic_vocab_parallel_head"),  # heuristic layers plus a tied vocabulary-parallel head
]


@pytest.mark.parametrize("world_size, case", ZERO_INIT_CASES)
class TestZeroInitBuildMatchesPlainBuild(DistributedTest):
    """Building the model under ``deepspeed.zero.Init`` must not change what AutoTP builds or how it trains.

    Under ``zero.Init`` a weight is an empty placeholder until it is gathered, so every AutoTP path that
    slices a weight must gather it first: a vocabulary-parallel output head and the embedding tied to
    it, and the heuristic layer matching used when there is no partition plan. The
    same weights are built plainly and under ``zero.Init``, trained for two accumulation windows, and
    must give the same losses and parameters. In the heuristic cases the caller builds the optimizer
    before ``deepspeed.initialize``, so it must also be re-pointed at the parameters AutoTP replaced.
    """

    def test_zero_init_matches_plain_build(self, world_size, case):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoTP under ZeRO-3 runs on an accelerator")
        model_class, config, ds_config, tp_size, client_optimizer = _zero_init_case(case)
        torch.manual_seed(1234)
        plain_model = model_class(config).float()
        state = copy.deepcopy(plain_model.state_dict())
        with deepspeed.zero.Init(config_dict_or_path=ds_config):
            zero_init_model = model_class(config)
        with deepspeed.zero.GatheredParameters(list(zero_init_model.parameters()), modifier_rank=0):
            if dist.get_rank() == 0:
                zero_init_model.load_state_dict(state)

        plain_engine = _initialize(plain_model, ds_config, client_optimizer)
        zero_init_engine = _initialize(zero_init_model, ds_config, client_optimizer)
        if case.endswith("vocab_parallel_head"):
            head = zero_init_engine.module.lm_head
            assert isinstance(head, VocabParallelLinear)
            assert head.weight is zero_init_engine.module.get_input_embeddings().weight
        plain_losses = _train_two_windows(plain_engine, tp_size)
        zero_init_losses = _train_two_windows(zero_init_engine, tp_size)

        # Every rank gathers every parameter before any assertion: a rank that failed early would leave the
        # others waiting in the next gather.
        plain_params = dict(plain_engine.module.named_parameters())
        zero_init_params = dict(zero_init_engine.module.named_parameters())
        assert zero_init_params.keys() == plain_params.keys(), sorted(set(zero_init_params) ^ set(plain_params))
        differing = []
        for name, param in zero_init_params.items():
            want = safe_get_full_fp32_param(plain_params[name])
            got = safe_get_full_fp32_param(param)
            if got.shape != want.shape or not torch.allclose(got, want, rtol=1e-5, atol=1e-6):
                differing.append(name)
        torch.testing.assert_close(zero_init_losses, plain_losses, rtol=1e-5, atol=1e-5)
        assert not differing, f"{len(differing)} parameters differ from the plain build: {differing}"


class TestZeroInitEmbeddingSlice(DistributedTest):
    """AutoTP must cut the same shard from an embedding table built under ``deepspeed.zero.Init`` as from a plain one.

    ``_slice_embedding`` splits a table along its output width (dim 1). That suits tables whose columns follow the
    attention heads, such as T5's relative-attention bias, which AutoTP's heuristic path slices inside a transformer
    layer. Under ``zero.Init`` the table is an empty placeholder until it is gathered. The private method is called
    directly to pin that fixed bug, because no model in these tests routes such a table through AutoTP.
    """
    world_size = 2

    def test_zero_init_embedding_slices_like_plain(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("ZeRO-3 partitioning runs on an accelerator")
        torch.manual_seed(1234)
        plain = torch.nn.Embedding(8, 4)  # [buckets, heads], like T5's relative_attention_bias
        zero_config = {"train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 3}}
        with deepspeed.zero.Init(config_dict_or_path=zero_config):
            partitioned = torch.nn.Embedding(8, 4)
        with deepspeed.zero.GatheredParameters([partitioned.weight], modifier_rank=0):
            if dist.get_rank() == 0:
                partitioned.weight.data.copy_(plain.weight.data)

        def slice_with_autotp(embedding):
            autotp = AutoTP(module=embedding,
                            all_reduce_linears=[],
                            prefix="",
                            state_dict=None,
                            linear_layer_setting=(torch.nn.Linear, torch.nn.Embedding),
                            orig_layer_impl=None)
            autotp.set_tensor_parallel_config(dist.get_world_size(), dist.get_world_group())
            return autotp._slice_embedding(embedding, "relative_attention_bias", False)

        want = slice_with_autotp(plain).weight.detach().cpu()
        got = slice_with_autotp(partitioned).weight.detach().cpu()
        assert want.shape == (8, 4 // dist.get_world_size())
        assert torch.equal(got, want), (got, want)
