# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Muon has to build its own param groups, but the ones it was given must survive that.

Which half of Muon a parameter belongs to is a property of the parameter, so `_configure_basic_optimizer`
regroups them. Every other optimizer receives `model_parameters` unchanged, so a group's own
`lr` / `weight_decay` reaches it; Muon flattened the groups away, which silently dropped the
no-weight-decay-on-biases-and-norms grouping most recipes use.
"""

import pytest
import torch

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from unit.common import DistributedTest


def _param(shape, use_muon, requires_grad=True):
    p = torch.nn.Parameter(torch.zeros(*shape), requires_grad=requires_grad)
    p.use_muon = use_muon
    return p


def _groups(model_parameters, **optimizer_parameters):
    """`_muon_param_groups` is a staticmethod, so the grouping runs without an engine."""
    return DeepSpeedEngine._muon_param_groups(model_parameters, optimizer_parameters)


def _by_name(param_groups):
    return {g["name"]: g for g in param_groups}


def test_a_plain_parameter_list_keeps_the_historical_groups():
    groups = _by_name(_groups([_param((4, 4), True), _param((4, ), False)], lr=1e-3, weight_decay=0.01))

    assert set(groups) == {"muon-params", "adam-params"}
    assert groups["muon-params"]["lr"] == 1e-3
    assert groups["adam-params"]["weight_decay"] == 0.01


def test_muon_lr_and_adam_lr_still_override_the_shared_lr():
    groups = _by_name(_groups([_param((4, 4), True), _param((4, ), False)], lr=1e-4, muon_lr=2e-2, adam_lr=1e-5))

    assert groups["muon-params"]["lr"] == 2e-2
    assert groups["adam-params"]["lr"] == 1e-5


def test_a_group_keeps_its_own_weight_decay():
    """The pattern this is really about: no weight decay on biases and norms."""
    decayed, undecayed = _param((4, 4), True), _param((4, ), False)
    groups = _by_name(
        _groups([{
            "params": [decayed],
            "weight_decay": 0.1
        }, {
            "params": [undecayed],
            "weight_decay": 0.0
        }],
                lr=1e-3,
                weight_decay=0.01))

    assert [g["weight_decay"] for g in groups.values()] == [0.1, 0.0]


def test_a_group_keeps_its_own_lr_even_against_muon_lr():
    """Most specific wins: the config's shared lr, then muon_lr / adam_lr, then the group."""
    tagged = _param((4, 4), True)
    groups = _groups([{"params": [tagged], "lr": 7e-3}], lr=1e-4, muon_lr=2e-2)

    assert groups[0]["lr"] == 7e-3


def test_a_group_without_an_lr_still_takes_muon_lr():
    groups = _groups([{"params": [_param((4, 4), True)], "weight_decay": 0.0}], lr=1e-4, muon_lr=2e-2)

    assert groups[0]["lr"] == 2e-2
    assert groups[0]["weight_decay"] == 0.0


def test_a_group_holding_both_kinds_splits_in_two_and_both_keep_its_settings():
    groups = _by_name(
        _groups([{
            "params": [_param((4, 4), True), _param((4, ), False)],
            "weight_decay": 0.0,
            "name": "no-decay"
        }],
                lr=1e-3,
                weight_decay=0.1))

    assert set(groups) == {"no-decay-muon-params", "no-decay-adam-params"}
    assert all(g["weight_decay"] == 0.0 for g in groups.values())
    assert all(g["lr"] == 1e-3 for g in groups.values())


def test_group_names_stay_distinct_across_groups():
    """MoE regrouping keys its buckets by name, so two groups must not collide on one."""
    param_groups = _groups([{"params": [_param((4, 4), True)]}, {"params": [_param((4, 4), True)]}], lr=1e-3)

    names = [g["name"] for g in param_groups]
    assert len(set(names)) == len(names) == 2


def test_the_muon_half_gets_muon_keys_and_the_adam_half_gets_adam_keys():
    groups = _by_name(
        _groups([_param((4, 4), True), _param((4, ), False)],
                lr=1e-3,
                momentum=0.9,
                ns_method="standard",
                betas=[0.9, 0.95],
                eps=1e-8))

    assert groups["muon-params"]["momentum"] == 0.9
    assert groups["muon-params"]["ns_method"] == "standard"
    assert "betas" not in groups["muon-params"]
    assert groups["adam-params"]["betas"] == [0.9, 0.95]
    assert "momentum" not in groups["adam-params"]


def test_frozen_parameters_are_left_out():
    param_groups = _groups([_param((4, 4), True, requires_grad=False), _param((4, ), False)], lr=1e-3)

    assert [g["name"] for g in param_groups] == ["adam-params"]


def test_an_untagged_parameter_is_reported_rather_than_crashing_on_the_attribute():
    """Without use_muon the old path logged an error and then died on `p.use_muon` two lines later."""
    stray = torch.nn.Parameter(torch.zeros(4, 4))

    with pytest.raises(ValueError, match="use_muon"):
        _groups([stray], lr=1e-3)


class TestMuonParamGroupsEndToEnd(DistributedTest):
    world_size = 1

    def test_the_settings_reach_the_optimizer(self):
        model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=True), torch.nn.LayerNorm(8))
        decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]

        _, optimizer, _, _ = deepspeed.initialize(model=model,
                                                  model_parameters=[{
                                                      "params": decay,
                                                      "weight_decay": 0.1
                                                  }, {
                                                      "params": no_decay,
                                                      "weight_decay": 0.0
                                                  }],
                                                  config={
                                                      "train_micro_batch_size_per_gpu": 1,
                                                      "gradient_accumulation_steps": 1,
                                                      "bf16": {
                                                          "enabled": True
                                                      },
                                                      "zero_optimization": {
                                                          "stage": 1
                                                      },
                                                      "optimizer": {
                                                          "type": "Muon",
                                                          "params": {
                                                              "lr": 5e-4,
                                                              "weight_decay": 0.01
                                                          }
                                                      },
                                                  })

        decays = sorted(g["weight_decay"] for g in optimizer.param_groups)
        assert decays == [0.0, 0.1], f"the groups the user passed were not honoured: {decays}"
