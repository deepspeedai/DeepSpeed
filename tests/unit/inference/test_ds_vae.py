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


def test_ds_vae_decode_forwards_the_generator():
    # every diffusers pipeline calls vae.decode(latents, return_dict=False, generator=generator)
    vae = RecordingVAE()
    ds_vae = DSVAE(vae, enable_cuda_graph=False)
    generator = torch.Generator()

    ds_vae.decode(torch.zeros(1), return_dict=False, generator=generator)

    assert vae.seen["decode"] == {"return_dict": False, "generator": generator}
