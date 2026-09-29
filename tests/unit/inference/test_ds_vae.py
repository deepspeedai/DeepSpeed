# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.model_implementations.diffusers.vae import DSVAE


class RecordingVAE(torch.nn.Module):
    """Records what DSVAE hands over. Signatures copied from diffusers `AutoencoderKL`."""

    def __init__(self):
        super().__init__()
        self.config = None
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.seen = {}

    def decode(self, z, return_dict=True, generator=None):
        self.seen["decode"] = {"return_dict": return_dict, "generator": generator}
        return "decoded"

    def encode(self, x, return_dict=True):
        self.seen["encode"] = {"return_dict": return_dict}
        return "encoded"

    def forward(self, sample, sample_posterior=False, return_dict=True, generator=None):
        self.seen["forward"] = {
            "sample_posterior": sample_posterior,
            "return_dict": return_dict,
            "generator": generator
        }
        return "forwarded"


def test_ds_vae_forward_reads_the_graph_flag_it_sets():
    # enable_cuda_graph defaults to True, and the flag DSVAE sets is `all_cuda_graph_created`
    vae = RecordingVAE()
    ds_vae = DSVAE(vae, enable_cuda_graph=True)
    ds_vae.all_cuda_graph_created = True
    ds_vae._graph_replay = lambda *inputs, **kwargs: "replayed"

    assert ds_vae(torch.zeros(1)) == "replayed"


def test_ds_vae_forwards_the_inputs_it_accepts():
    vae = RecordingVAE()
    ds_vae = DSVAE(vae, enable_cuda_graph=False)

    assert ds_vae(torch.zeros(1), sample_posterior=True, return_dict=False) == "forwarded"
    assert vae.seen["forward"] == {"sample_posterior": True, "return_dict": False, "generator": None}


def test_ds_vae_forward_runs_a_seeded_call_eagerly():
    # a captured graph cannot use the caller's generator, so sample_posterior would not follow the seed
    vae = RecordingVAE()
    ds_vae = DSVAE(vae, enable_cuda_graph=True)

    def no_capture(*inputs, **kwargs):
        raise AssertionError("a call with a generator must not be captured")

    ds_vae._create_cuda_graph = no_capture
    generator = torch.Generator()

    assert ds_vae(torch.zeros(1), sample_posterior=True, generator=generator) == "forwarded"
    assert vae.seen["forward"]["generator"] is generator
