# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from types import SimpleNamespace

import deepspeed.model_implementations.diffusers.unet as unet_module
from deepspeed.model_implementations.diffusers.unet import DSUNet
from deepspeed.model_implementations.features.cuda_graph import refresh_static_tensors


class RecordingUNet(torch.nn.Module):
    """Stands in for a diffusers UNet. DSUNet reads these four attributes in __init__."""

    def __init__(self):
        super().__init__()
        self.in_channels = 4
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = None

    def forward(self, *inputs, **kwargs):
        return "unet"


def test_refresh_static_tensors_follows_dicts_lists_and_tuples():
    static = {"a": torch.zeros(2), "b": [torch.zeros(2), (torch.zeros(2), )]}

    refresh_static_tensors(static, {"a": torch.ones(2), "b": [torch.full((2, ), 2.0), (torch.full((2, ), 3.0), )]})

    assert torch.equal(static["a"], torch.ones(2))
    assert torch.equal(static["b"][0], torch.full((2, ), 2.0))
    assert torch.equal(static["b"][1][0], torch.full((2, ), 3.0))


@pytest.mark.parametrize("b", ["missing", "extra", None, [torch.ones(2)]])
def test_refresh_static_tensors_rejects_what_capture_did_not_see(b):
    # A replay can only run the inputs it captured. A missing, extra or retyped entry would
    # otherwise replay stale values and return a plausible wrong result.
    static = {"a": torch.zeros(2), "b": torch.zeros(2)}
    new = {"a": torch.ones(2), "b": b}
    if b == "missing":
        del new["b"]
    elif b == "extra":
        new.update(b=torch.ones(2), c=torch.ones(2))
    with pytest.raises(ValueError, match="captured call"):
        refresh_static_tensors(static, new)


def test_ds_unet_graph_replay_refreshes_nested_conditioning(monkeypatch):
    # SDXL passes added_cond_kwargs as a dict of tensors. Copying only top-level tensors left
    # that dict holding what capture saw, so every later prompt replayed the first one's
    # conditioning and the pipeline returned a plausible image for the wrong text.
    monkeypatch.setattr(unet_module, "get_accelerator", lambda: SimpleNamespace(replay_graph=lambda graph: None))

    ds_unet = DSUNet(RecordingUNet(), enable_cuda_graph=True)
    added_cond_kwargs = {"text_embeds": torch.zeros(2), "time_ids": torch.zeros(2)}
    ds_unet.static_inputs = (torch.zeros(2), )
    ds_unet.static_kwargs = {"added_cond_kwargs": added_cond_kwargs}
    ds_unet.static_output = "replayed"
    ds_unet._cuda_graphs = None

    output = ds_unet._graph_replay(torch.ones(2),
                                   added_cond_kwargs={
                                       "text_embeds": torch.full((2, ), 3.0),
                                       "time_ids": torch.full((2, ), 4.0)
                                   })

    assert output == "replayed"
    assert torch.equal(ds_unet.static_inputs[0], torch.ones(2))
    assert torch.equal(added_cond_kwargs["text_embeds"], torch.full((2, ), 3.0))
    assert torch.equal(added_cond_kwargs["time_ids"], torch.full((2, ), 4.0))


def test_ds_unet_graph_replay_rejects_a_kwarg_added_after_capture(monkeypatch):
    # Capturing without an attention mask and passing one later used to raise KeyError from the
    # copy loop. Replaying the no-mask graph instead would ignore the mask without a word.
    monkeypatch.setattr(unet_module, "get_accelerator", lambda: SimpleNamespace(replay_graph=lambda graph: None))

    ds_unet = DSUNet(RecordingUNet(), enable_cuda_graph=True)
    ds_unet.static_inputs = (torch.zeros(2), )
    ds_unet.static_kwargs = {}
    ds_unet.static_output = "replayed"
    ds_unet._cuda_graphs = None

    with pytest.raises(ValueError, match="captured call"):
        ds_unet._graph_replay(torch.ones(2), attention_mask=torch.ones(2))


def test_refresh_static_tensors_ignores_plain_values():
    # Only tensors are captured, so a plain kwarg such as return_dict can come and go.
    static = {"a": torch.zeros(2), "return_dict": False}
    refresh_static_tensors(static, {"a": torch.ones(2), "scale": 2.0})
    assert torch.equal(static["a"], torch.ones(2))
