'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import torch


class DSClipEncoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        enc.text_model._build_causal_attention_mask = self._build_causal_attention_mask
        self.enc = enc
        self.device = self.enc.device
        self.dtype = self.enc.dtype
        self.cuda_graph_created = False

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        mask = torch.empty(bsz,
                           seq_len,
                           seq_len,
                           dtype=dtype,
                           device=torch.cuda.current_device())
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)
        mask = mask.unsqueeze(1)
        return mask

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

    def _forward(self, *inputs, **kwargs):

        return self.enc(*inputs, **kwargs)
