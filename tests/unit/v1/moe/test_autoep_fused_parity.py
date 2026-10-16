# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""End-to-end parity for the eager and fused combine implementations."""

import functools

import deepspeed
import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
from deepspeed.ops.triton_ops import autoep_fused_token_ops as fused_ops
from deepspeed.utils import safe_get_full_grad
from unit.common import DistributedTest
from unit.v1.moe.autoep_test_utils import (
    MockMoEOnlyTransformer,
    MockMoETransformer,
    engine_input_dtype,
    mixed_precision_config,
    seed_everything,
)

HIDDEN_SIZE = 64
SEQ_LEN = 16
NUM_EXPERTS = 4

# The top-k accumulation order differs, so parity allows last-bit noise only.
PARITY_TOLERANCE = {"rtol": 1e-2, "atol": 1e-3}


def _fused_engine_available():
    accelerator = get_accelerator()
    return (accelerator.is_available() and accelerator.device_name().startswith("cuda") and fused_ops.is_available())


pytestmark = pytest.mark.skipif(not _fused_engine_available(),
                                reason="the fused weighted restore needs CUDA and Triton")


def _config(combine_impl, ep_size):
    return {
        **mixed_precision_config(),
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 0.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        "expert_parallel": {
            "enabled": True,
            "autoep_size": ep_size,
            "preset_model": "mixtral",
            "load_balance_coeff": None,
            "combine_impl": combine_impl,
        },
    }


def _build_engine(combine_impl, ep_size, reference_state, seed):
    seed_everything(seed)
    model = MockMoETransformer(num_layers=2,
                               num_experts=NUM_EXPERTS,
                               hidden_size=HIDDEN_SIZE,
                               intermediate_size=2 * HIDDEN_SIZE)
    model.load_state_dict(reference_state)
    engine, _, _, _ = deepspeed.initialize(model=model, config=_config(combine_impl, ep_size))
    return engine


def _build_folded_engine(combine_impl, zero_stage, reference_state, seed):
    # FP32: the default folded restore rounds each product and each top-k addition to the
    # activation dtype, so in fp16 it differs from the FP32 fused sum by about one unit in the
    # last place of the MoE output, more than PARITY_TOLERANCE allows after the LM head.
    config = {key: value for key, value in _config(combine_impl, 2).items() if key not in ("fp16", "bf16")}
    # The per-expert loop does not depend on which dtypes this torch's grouped GEMM accepts.
    config["expert_parallel"]["use_grouped_mm"] = False
    config["zero_optimization"] = {"stage": zero_stage}
    # Keep the dense weights whole so that only the MoE dispatch and restore are folded.
    config["tensor_parallel"] = {
        "autotp_size": 2,
        "partition_config": {
            "use_default_specs": False,
            "layer_specs": [{
                "patterns": [".*\\.weight$"],
                "partition_type": "skip",
            }],
        },
    }
    seed_everything(seed)
    model = _folded_model()
    model.load_state_dict(reference_state)
    engine, _, _, _ = deepspeed.initialize(model=model, config=config)
    return engine


def _folded_model():
    # No attention: the key bias has a zero true gradient, and Adam's first step turns its
    # rounding noise into +/- lr, which no parameter-update tolerance can compare.
    return MockMoEOnlyTransformer(num_layers=2,
                                  num_experts=NUM_EXPERTS,
                                  hidden_size=HIDDEN_SIZE,
                                  intermediate_size=2 * HIDDEN_SIZE)


def _checkpoint_moe_layers(engine):
    """Recompute each MoE block in backward, as the benchmarked runs do."""
    for module in engine.module.modules():
        if isinstance(module, AutoEPMoELayer):
            module.forward = functools.partial(torch.utils.checkpoint.checkpoint, module.forward, use_reentrant=False)


def _named_gradients(engine):
    gradients = {}
    for name, param in engine.module.named_parameters():
        if not param.requires_grad:
            continue
        grad = safe_get_full_grad(param)
        if grad is not None:
            gradients[name] = grad.detach().float().cpu().clone()
    return gradients


def _parameters(engine):
    return {
        name: param.detach().float().cpu().clone()
        for name, param in engine.module.named_parameters() if param.requires_grad
    }


def _take_one_step(engine, seed, *, checkpoint_activations):
    if checkpoint_activations:
        _checkpoint_moe_layers(engine)

    generator = torch.Generator().manual_seed(seed)
    batch = torch.randn((1, SEQ_LEN, HIDDEN_SIZE), generator=generator, dtype=torch.float32)
    batch = batch.to(engine.device, dtype=engine_input_dtype(engine)).requires_grad_(True)

    before = _parameters(engine)
    output = engine(batch)
    loss = output.float().pow(2).mean()
    engine.backward(loss)

    gradients = _named_gradients(engine)
    input_grad = batch.grad.detach().float().cpu().clone()
    engine.step()

    delta = {name: _parameters(engine)[name] - value for name, value in before.items()}
    return {
        "loss": loss.detach().float().cpu().clone(),
        "output": output.detach().float().cpu().clone(),
        "input_grad": input_grad,
        "gradients": gradients,
        "delta": delta,
    }


def _assert_step_matches(fused, eager):
    torch.testing.assert_close(fused["loss"], eager["loss"], **PARITY_TOLERANCE)
    torch.testing.assert_close(fused["output"], eager["output"], **PARITY_TOLERANCE)
    torch.testing.assert_close(fused["input_grad"], eager["input_grad"], **PARITY_TOLERANCE)

    assert fused["gradients"], "no gradients were captured, so the comparison would be vacuous"
    assert set(fused["gradients"]) == set(eager["gradients"])
    # Ensure both sides of the fused restore reached the comparison.
    assert any(".router." in name for name in fused["gradients"]), "no router gradient was captured"
    assert any(".experts.w" in name for name in fused["gradients"]), "no expert gradient was captured"

    for name in sorted(eager["gradients"]):
        torch.testing.assert_close(fused["gradients"][name],
                                   eager["gradients"][name],
                                   msg=lambda formatted, name=name: f"gradient mismatch for {name}\n{formatted}",
                                   **PARITY_TOLERANCE)

    for name in sorted(eager["delta"]):
        torch.testing.assert_close(fused["delta"][name],
                                   eager["delta"][name],
                                   msg=lambda formatted, name=name: f"parameter update mismatch for {name}\n"
                                   f"{formatted}",
                                   **PARITY_TOLERANCE)

    assert any(value.abs().sum() > 0 for value in eager["delta"].values()), "the optimizer step changed nothing"


class TestAutoEPFusedParityExpertParallel(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize("checkpoint_activations", [True, False])
    def test_fused_matches_eager_through_a_full_step(self, checkpoint_activations):
        seed = 4321
        seed_everything(seed)
        reference_state = MockMoETransformer(num_layers=2,
                                             num_experts=NUM_EXPERTS,
                                             hidden_size=HIDDEN_SIZE,
                                             intermediate_size=2 * HIDDEN_SIZE).state_dict()

        eager_engine = _build_engine("weighted_sum", 2, reference_state, seed)
        eager = _take_one_step(eager_engine, seed, checkpoint_activations=checkpoint_activations)

        fused_engine = _build_engine("fused_weighted_sum", 2, reference_state, seed)
        assert all(module.combine_impl == "fused_weighted_sum" for module in fused_engine.module.modules()
                   if isinstance(module, AutoEPMoELayer)), "the fused reduction was not actually selected"
        fused = _take_one_step(fused_engine, seed, checkpoint_activations=checkpoint_activations)

        _assert_step_matches(fused, eager)


class TestAutoEPFusedParityLocalExperts(DistributedTest):
    world_size = 1

    def test_fused_matches_eager_without_expert_parallelism(self):
        seed = 8765
        seed_everything(seed)
        reference_state = MockMoETransformer(num_layers=2,
                                             num_experts=NUM_EXPERTS,
                                             hidden_size=HIDDEN_SIZE,
                                             intermediate_size=2 * HIDDEN_SIZE).state_dict()

        eager = _take_one_step(_build_engine("weighted_sum", 1, reference_state, seed),
                               seed,
                               checkpoint_activations=False)
        fused = _take_one_step(_build_engine("fused_weighted_sum", 1, reference_state, seed),
                               seed,
                               checkpoint_activations=False)

        _assert_step_matches(fused, eager)


class TestAutoEPFusedParityFoldedTensorParallel(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("zero_stage", [0, 2])
    @pytest.mark.parametrize("checkpoint_activations", [True, False])
    def test_fused_matches_eager_with_autotp_folding(self, zero_stage, checkpoint_activations):
        seed = 2468
        seed_everything(seed)
        reference_state = _folded_model().state_dict()

        eager_engine = _build_folded_engine("weighted_sum", zero_stage, reference_state, seed)
        eager = _take_one_step(eager_engine, seed, checkpoint_activations=checkpoint_activations)

        fused_engine = _build_folded_engine("fused_weighted_sum", zero_stage, reference_state, seed)
        moe_layers = [module for module in fused_engine.module.modules() if isinstance(module, AutoEPMoELayer)]
        assert moe_layers
        assert all(layer.combine_impl == "fused_weighted_sum"
                   for layer in moe_layers), "the fused reduction was not actually selected"
        assert all(layer.folding_group_handles is not None and layer.folding_group_handles.spec.tp_size == 2
                   for layer in moe_layers), "the MoE layers were not folded with tensor parallelism"
        fused = _take_one_step(fused_engine, seed, checkpoint_activations=checkpoint_activations)

        _assert_step_matches(fused, eager)
