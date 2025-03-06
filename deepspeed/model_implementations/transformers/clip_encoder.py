# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.accelerator import get_accelerator
from ..features.cuda_graph import CUDAGraph


class DSClipEncoder(CUDAGraph, torch.nn.Module):

    def __init__(self, enc, enable_cuda_graph=False):
        super().__init__(enable_cuda_graph=enable_cuda_graph)
        enc.text_model._build_causal_attention_mask = self._build_causal_attention_mask
        self.enc = enc
        self.device = self.enc.device
        self.dtype = self.enc.dtype
        self.cuda_graph_created = [False, False]
        self.static_inputs = [None, None]
        self.static_kwargs = [None, None]
        self.static_output = [None, None]
        self._cuda_graphs = [None, None]
        self.iter = 0
        self.config = self.enc.config

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=get_accelerator().current_device_name())
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[self.iter][i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[self.iter][k].copy_(kwargs[k])
        get_accelerator().replay_graph(self._cuda_graphs[self.iter])
        return self.static_output[self.iter]

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.cuda_graph_created[self.iter]:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
            self.iter = (self.iter + 1) % 2
            return outputs
        else:
            return self.enc(*inputs, **kwargs)

    def _create_cuda_graph(self, *inputs, **kwargs):
        # warmup to create the workspace and cublas handle
        cuda_stream = torch.cuda.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)

        # create cuda_graph and assign static_inputs and static_outputs
        self._cuda_graphs[self.iter] = get_accelerator().create_graph()
        self.static_inputs[self.iter] = inputs
        self.static_kwargs[self.iter] = kwargs

        with get_accelerator().capture_to_graph(self._cuda_graphs[self.iter]):
            self.static_output[self.iter] = self._forward(*self.static_inputs[self.iter],
                                                          **self.static_kwargs[self.iter])

        self.cuda_graph_created[self.iter] = True

    def _forward(self, *inputs, **kwargs):
        return self.enc(*inputs, **kwargs)
