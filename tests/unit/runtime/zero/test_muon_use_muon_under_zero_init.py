# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""`use_muon` has to be decided on the layer's shape, not on the ZeRO-3 partition's.

`deepspeed.zero.Init` replaces a partitioned parameter's data with a flat placeholder and
records the shape it has as a layer in `ds_shape`. Muon is applied to matrices, so a check
against `param.shape` sees a 1-D tensor for every parameter in the model, tags none of them,
and ZeRO-3 then finds no sub-group using Muon. Training continues with every parameter on the
AdamW branch and nothing says so.
"""

from types import SimpleNamespace

import pytest
import torch

import deepspeed
from deepspeed.runtime.config import MUON_OPTIMIZER
from unit.common import DistributedTest


class _Model(torch.nn.Module):

    def __init__(self, hidden=16):
        super().__init__()
        self.attn = torch.nn.Linear(hidden, hidden, bias=False)
        self.mlp = torch.nn.Linear(hidden, hidden, bias=False)
        self.norm = torch.nn.Parameter(torch.ones(hidden))
        self.embed_tokens = torch.nn.Embedding(8, hidden)

    def forward(self, x):
        return (self.mlp(self.attn(x)) * self.norm).square().sum()


def _muon_flags(model):
    config = SimpleNamespace(optimizer_name=MUON_OPTIMIZER, optimizer_params={})
    deepspeed.set_optimizer_flags(config, model)
    return {name: p.use_muon for name, p in model.named_parameters()}


def _partition_like_zero3(model):
    """What `zero.Init` leaves behind: a flat placeholder plus the real shape on `ds_shape`."""
    for p in model.parameters():
        p.ds_shape = torch.Size(p.shape)
        p.data = torch.zeros(0, dtype=p.dtype)


def test_matrices_are_tagged_when_the_model_is_not_partitioned():
    flags = _muon_flags(_Model())

    assert flags["attn.weight"] is True
    assert flags["mlp.weight"] is True
    assert flags["norm"] is False
    assert flags["embed_tokens.weight"] is False


def test_matrices_are_still_tagged_once_zero3_has_partitioned_them():
    model = _Model()
    _partition_like_zero3(model)

    flags = _muon_flags(model)

    assert model.attn.weight.ndim == 1, "the partitioned parameter really is 1-D here"
    assert flags["attn.weight"] is True
    assert flags["mlp.weight"] is True


def test_the_partitioned_shape_does_not_promote_a_vector():
    """A 1-D parameter stays off Muon; `ds_shape` is read for its rank, not assumed to be >= 2."""
    model = _Model()
    _partition_like_zero3(model)

    flags = _muon_flags(model)

    assert flags["norm"] is False
    assert flags["embed_tokens.weight"] is False, "the name exclusions still apply"


class TestMuonRunsUnderZeroInit(DistributedTest):
    """The end-to-end consequence: `muon_update` is reached at all."""
    world_size = 1

    @pytest.mark.parametrize("zero_init", [False, True])
    def test_muon_update_is_called(self, zero_init):
        import deepspeed.runtime.zero.stage3 as stage3

        config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_scatter": False
            },
            "optimizer": {
                "type": "Muon",
                "params": {
                    "lr": 1e-3
                }
            },
        }

        if zero_init:
            with deepspeed.zero.Init(config_dict_or_path=config):
                model = _Model()
        else:
            model = _Model()

        calls = []
        original = stage3.muon_update

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        stage3.muon_update = counting
        try:
            engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            x = torch.ones(1, 16, dtype=torch.bfloat16, device=engine.device)
            engine.backward(engine(x))
            engine.step()
        finally:
            stage3.muon_update = original

        assert calls, "Muon never ran; every parameter was left on the AdamW branch"
