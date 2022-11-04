'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch
import diffusers


class DSUNet(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        # SD pipeline accesses this attribute
        self.in_channels = unet.in_channels
        self._traced_unet = None
        self._trace_enabled = False
        self.device = self.unet.device
        self.dtype = self.unet.dtype
        self.fwd_count = 0
        self.unet.requires_grad_(requires_grad=False)
        self.unet.to(memory_format=torch.channels_last)
        self.cuda_graph_created = False

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._cuda_graphs.replay()
        return self.static_output

    def forward(self, *inputs, **kwargs):
        if self.cuda_graph_created:
            outputs = self._graph_replay(*inputs, **kwargs)
        else:
            self._create_cuda_graph(*inputs, **kwargs)
            outputs = self._graph_replay(*inputs, **kwargs)
        return outputs

    def _create_cuda_graph(self, *inputs, **kwargs):
        # warmup to create the workspace and cublas handle
        cuda_stream = torch.cuda.Stream()
        cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(cuda_stream):
            for i in range(3):
                ret = self._forward(*inputs, **kwargs)
        torch.cuda.current_stream().wait_stream(cuda_stream)

        # create cuda_graph and assign static_inputs and static_outputs
        self._cuda_graphs = torch.cuda.CUDAGraph()
        self.static_inputs = inputs
        self.static_kwargs = kwargs

        with torch.cuda.graph(self._cuda_graphs):
            self.static_output = self._forward(*self.static_inputs, **self.static_kwargs)

        self.cuda_graph_created = True

    def _forward(self, sample, timestamp, encoder_hidden_states, return_dict=True):
        if self._trace_enabled:
            if self._traced_unet is None:
                print("Unet: start tracing with Nvfuser")
                # force return tuple instead of dict
                self._traced_unet = torch.jit.trace(
                    lambda _sample,
                    _timestamp,
                    _encoder_hidden_states: self.unet(_sample,
                                                      _timestamp,
                                                      _encoder_hidden_states,
                                                      return_dict=False),
                    (sample,
                     timestamp,
                     encoder_hidden_states))
                return self.unet(sample, timestamp, encoder_hidden_states)
            else:
                # convert return type to UNet2DConditionOutput
                out_sample, *_ = self._traced_unet(sample, timestamp, encoder_hidden_states)
                return diffusers.models.unet_2d_condition.UNet2DConditionOutput(
                    out_sample)
        else:
            return self.unet(sample, timestamp, encoder_hidden_states, return_dict)
