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
from deepspeed.utils import groups, safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state
from unit.common import DistributedFixture, DistributedTest

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
    SGD momentum makes that step depend on the restored optimizer state.
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

        torch.testing.assert_close(resumed_losses, losses, rtol=1e-6, atol=1e-6)
        assert not differing, f"{len(differing)} parameters differ after resuming: {differing}"


UNIVERSAL_SOURCE = (2, 2)  # TP, EP of the run that saves, on 4 ranks
UNIVERSAL_STATES = ("fp32", "exp_avg", "exp_avg_sq")


def _universal_config(tp_size, ep_size, precision):
    config = _folded_config(tp_size, ep_size, zero_stage=3)
    # Universal Checkpointing converts Adam's states.
    config["optimizer"] = {"type": "AdamW", "params": {"lr": 1e-3, "weight_decay": 0.01}}
    if tp_size == 1:
        del config["tensor_parallel"]
    if precision == "bf16":
        config["bf16"] = {"enabled": True}
    return config


def _build_universal_engine(tp_size, ep_size, precision, seed, load_universal=False):
    config = _universal_config(tp_size, ep_size, precision)
    if load_universal:
        config["checkpoint"] = {"load_universal": True}
    torch.manual_seed(seed)
    model = transformers.Glm4MoeForCausalLM(_tiny_glm4_moe_config()).float()
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    return engine


def _logical_state(engine, key, tp_size):
    """Every parameter, or one of its Adam states, at full size under names every layout shares."""

    def full(param):
        if key == "fp32":
            return safe_get_full_fp32_param(param)
        return safe_get_full_optimizer_state(param, key)

    values = {}
    moe_layers = []
    for module_name, module in engine.module.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, AutoEPMoELayer):
            moe_layers.append((prefix, module))
        for name, param in module.named_parameters(recurse=False):
            value = full(param)
            assert value is not None, f"{prefix}{name} has no {key}"
            if tp_size > 1 and isinstance(module, LinearAllreduce) and name == "weight":
                value = _gather_tp_shards(value, dim=1)
            elif tp_size > 1 and isinstance(module, LinearLayer):
                value = _gather_tp_shards(value, dim=0)
            values[f"{prefix}{name}"] = value
    # Each EP rank holds a block of experts; the full tensor stacks them in EP-rank order.
    for prefix, module in moe_layers:
        for wname in ("w1", "w2", "w3"):
            local = values[f"{prefix}experts.{wname}"]
            shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=module.ep_group))]
            dist.all_gather(shards, local.contiguous(), group=module.ep_group)
            values[f"{prefix}experts.{wname}"] = torch.cat(shards, dim=0)
    return {name: value.detach().float().cpu() for name, value in values.items()}


def _universal_mismatches(got, want, rtol, atol):
    assert got.keys() == want.keys(), sorted(set(got) ^ set(want))
    mismatches = []
    for name, expected in want.items():
        actual = got[name]
        if actual.shape != expected.shape:
            mismatches.append(f"{name}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}")
        elif not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            relative = (actual - expected).abs().max().item() / max(expected.abs().max().item(), 1e-12)
            mismatches.append(f"{name}: largest relative difference {relative:.3e}")
    return mismatches


class _FoldedUniversalSource(DistributedFixture):
    """Train a folded TP 2 x EP 2 run one step, save it, convert it, and record what a load must restore.

    The record holds every parameter and Adam state at full size after the step, then the losses of
    the next accumulation window and the parameters after the next step.
    """
    world_size = 4

    def run(self, universal_precision, tmpdir):
        tp_size, ep_size = UNIVERSAL_SOURCE
        engine = _build_universal_engine(tp_size, ep_size, universal_precision, seed=1234)
        logical_rank, logical_world_size = dist.get_rank() // tp_size, dist.get_world_size() // tp_size
        _run_to_boundary(engine, logical_rank, logical_world_size)
        engine.step()
        save_dir = str(tmpdir)
        engine.save_checkpoint(save_dir, tag="step1")

        dist.barrier()
        error = None
        if dist.get_rank() == 0:
            try:
                ds_to_universal.main(
                    SimpleNamespace(input_folder=os.path.join(save_dir, "step1"),
                                    output_folder=os.path.join(save_dir, "step1_universal"),
                                    num_extract_workers=1,
                                    num_merge_workers=1,
                                    keep_temp_folder=False,
                                    strict=True,
                                    inject_missing_state=False))
            except Exception as exc:  # report on every rank instead of leaving them in the next collective
                error = f"{type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        assert errors[0] is None, f"universal conversion failed: {errors[0]}"

        states = {key: _logical_state(engine, key, tp_size) for key in UNIVERSAL_STATES}
        losses = _run_to_boundary(engine, logical_rank, logical_world_size)
        engine.step()
        params_after = _logical_state(engine, "fp32", tp_size)
        all_losses = [None] * dist.get_world_size()
        dist.all_gather_object(all_losses, losses)
        if dist.get_rank() == 0:
            torch.save({
                "states": states,
                "next_losses": all_losses,
                "next_params": params_after
            }, os.path.join(save_dir, "source_record.pt"))
        dist.barrier()


@pytest.fixture(params=["fp32", "bf16"])
def universal_precision(request):
    return request.param


@pytest.fixture
def folded_universal_source(request):
    _FoldedUniversalSource()(request)


UNIVERSAL_TARGETS = [
    (4, 2, 2),  # the same layout
    (4, 2, 4),  # EP 4: each EP group spans both TP groups
    (4, 4, 2),  # TP 4: one data-parallel group
    (8, 4, 2),  # more ranks with the same 2 data-parallel groups
    (2, 1, 2),  # AutoEP without AutoTP, 2 data-parallel groups
]


@pytest.mark.parametrize("world_size, tp_size, ep_size", UNIVERSAL_TARGETS)
class TestZero3FoldedUniversalCheckpoint(DistributedTest):
    """A folded ZeRO-3 checkpoint converted to the universal format loads into another layout.

    Every parameter and Adam state must come back exactly. Layouts with the source's 2 data-parallel
    groups read the same batches, so they must also repeat its next step.
    """

    def test_universal_load_restores_and_resumes(self, folded_universal_source, universal_precision, tmpdir,
                                                 world_size, tp_size, ep_size):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("AutoEP folding runs on an accelerator")
        save_dir = str(tmpdir)
        record = torch.load(os.path.join(save_dir, "source_record.pt"), weights_only=False)
        engine = _build_universal_engine(tp_size, ep_size, universal_precision, seed=999, load_universal=True)
        engine.load_checkpoint(save_dir, tag="step1_universal")

        # Every rank finishes its collectives before any assertion, so a failure cannot leave others waiting.
        problems = []
        for key in UNIVERSAL_STATES:
            loaded = _logical_state(engine, key, tp_size)
            problems += [f"{key}: {m}" for m in _universal_mismatches(loaded, record["states"][key], 1e-6, 1e-8)]

        source_tp, source_ep = UNIVERSAL_SOURCE
        logical_world_size = world_size // tp_size
        same_batches = logical_world_size == 4 // source_tp
        same_layout = (world_size, tp_size, ep_size) == (4, source_tp, source_ep)
        # In bf16 another TP split rounds the forward differently, so there only the same layout repeats the step.
        if same_batches and (universal_precision == "fp32" or same_layout):
            logical_rank = dist.get_rank() // tp_size
            losses = _run_to_boundary(engine, logical_rank, logical_world_size)
            engine.step()
            want_losses = record["next_losses"][logical_rank * source_tp]
            if max(abs(got - want) for got, want in zip(losses, want_losses)) > 1e-5:
                problems.append(f"next losses {losses}, expected {want_losses}")
            # The attention key bias has a zero true gradient, so its gradient is rounding noise that
            # another layout adds up in another order, and AdamW turns that noise into steps of about lr.
            after = {
                name: value
                for name, value in _logical_state(engine, "fp32", tp_size).items() if not name.endswith("k_proj.bias")
            }
            want = {name: record["next_params"][name] for name in after}
            problems += [f"after the next step: {m}" for m in _universal_mismatches(after, want, 1e-4, 1e-6)]

        assert not problems, f"{len(problems)} problems:\n" + "\n".join(problems[:40])


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
