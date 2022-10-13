'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import math
import torch
from torch.autograd import Function
from ... import op_builder
import torch.nn as nn
from packaging import version as pkg_version
from deepspeed.utils.logging import log_dist
# Cuda modules will be imported if needed
inference_cuda_module = None
minus_inf = -10000.0
triton_flash_attn = None


def load_triton_flash_attn():
    global triton_flash_attn
    try:
        import triton
    except ImportError:
        raise ImportError("Please install triton 2.0+ or `pip install deepspeed[sd]`")

    if pkg_version.parse(triton.__version__) < pkg_version.parse("2.0"):
        raise ImportError("Please install triton 2.0+ or `pip install deepspeed[sd]`")

    from .triton_ops import triton_flash_attn


class DeepSpeedAttentionFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                context,
                input_mask,
                config,
                attn_qkvw,
                attn_qw,
                attn_kw,
                attn_vw,
                attn_qkvb,
                num_attention_heads_per_partition,
                norm_factor,
                hidden_size_per_partition,
                attn_ow,
                attn_ob,
                score_context_func,
                linear_func,
                triton_flash_attn_kernel):
        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3)
            new_x_layer_shape = x.size()[:-2] + \
                                      (hidden_size_per_partition,)
            return x.reshape(*new_x_layer_shape)

        def _transpose_for_scores(x):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (num_attention_heads_per_partition,
                                           attention_head_size)
            x = x.reshape(*new_x_shape)
            x = x.permute(0, 2, 1, 3)
            return x.contiguous()

        def compute_attention(qkv_out, input_mask):
            no_masking = input_mask is None

            head_size = (qkv_out.shape[-1] // 3 // num_attention_heads_per_partition)
            if no_masking:
                input_mask = torch.empty(1)

            context_layer, _, _ = score_context_func(
                qkv_out,
                ((1 - input_mask).to(qkv_out.dype) *
                 minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                config.rotary_dim,
                config.rotate_half,
                config.rotate_every_two,
                num_attention_heads_per_partition,
                (1 / norm_factor if config.scale_attention else 1.0),
                config.triangular_masking,
                config.local_attention,
                config.window_size,
                no_masking,
                config.layer_id,
                DeepSpeedAttention.layer_id,
                torch.empty(1))
            return context_layer

        def selfAttention_fp(input, context, input_mask):
            if config.fp16 and input.dtype == torch.float32:
                input = input.half()
            head_size = input.shape[-1] // config.heads
            do_flash_attn = (head_size <= 128)
            scale = (1 / norm_factor) * (1 / norm_factor)
            if context == None:
                qkv_out = linear_func(input,
                                      attn_qkvw,
                                      attn_qkvb if attn_qkvb is not None else attn_qkvw,
                                      attn_qkvb is not None,
                                      True,
                                      do_flash_attn,
                                      config.heads,
                                      DeepSpeedAttention.layer_id)
                if do_flash_attn:
                    context_layer = triton_flash_attn_kernel(qkv_out[0],
                                                             qkv_out[1],
                                                             qkv_out[2],
                                                             scale,
                                                             input.shape[-2] % 128 == 0)
                    context_layer = _transpose_for_context(context_layer[:,:,:,:head_size])
                else:
                    context_layer = compute_attention(qkv_out, input_mask)
            else:
                query = torch.matmul(input, attn_qw)
                key = torch.matmul(context, attn_kw)
                value = torch.matmul(context, attn_vw)
                query, key, value = inference_cuda_module.pad_transform_fp16(query, key, value, config.heads, do_flash_attn)
                if do_flash_attn:
                    context_layer = triton_flash_attn_kernel(query,
                                                             key,
                                                             value,
                                                             scale,
                                                             input.shape[-2] % 128 == 0)
                    context_layer = _transpose_for_context(context_layer[:,:,:,:head_size])
                else:
                    attention_scores = (torch.matmul(query,
                                                     key.transpose(-1,
                                                                   -2)) *
                                        scale).softmax(dim=-1)
                    context_layer = _transpose_for_context(
                        torch.matmul(attention_scores,
                                     value))

            output = linear_func(context_layer,
                                 attn_ow,
                                 attn_ob,
                                 attn_ob is not None,
                                 True,
                                 False,
                                 config.heads,
                                 DeepSpeedAttention.layer_id)
            return output

        output = selfAttention_fp(input, context, input_mask)

        return output

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedAttention(nn.Module):
    """Initialize the DeepSpeed Transformer Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
    """
    layer_id = 0

    def __init__(
        self,
        config,
    ):
        super(DeepSpeedAttention, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedAttention.layer_id
        DeepSpeedAttention.layer_id += 1
        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3

        data_type = torch.int8 if config.q_int8 else torch.half if config.fp16 else torch.float
        data_type_fp = torch.half if config.fp16 else torch.float
        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        if DeepSpeedAttention.layer_id == 1:
            log_dist(f"DeepSpeed-Attention config: {self.config.__dict__}", [0])

        self.attn_qkvw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                  qkv_size_per_partition,
                                                  dtype=data_type,
                                                  device=device),
                                      requires_grad=False)
        self.attn_kw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_vw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_qw = nn.Parameter(torch.empty(self.config.hidden_size,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)
        self.attn_qkvb = nn.Parameter(torch.empty(qkv_size_per_partition,
                                                  dtype=data_type_fp,
                                                  device=device),
                                      requires_grad=False)
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(torch.empty(out_size_per_partition,
                                                self.config.hidden_size,
                                                dtype=data_type,
                                                device=device),
                                    requires_grad=False)

        self.attn_ob = nn.Parameter(torch.empty(self.config.hidden_size,
                                                dtype=data_type_fp,
                                                device=device),
                                    requires_grad=False)
        if triton_flash_attn is None:
            load_triton_flash_attn()
        self.triton_flash_attn_kernel = triton_flash_attn()
        self.num_attention_heads_per_partition = self.config.heads // self.config.mp_size
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = self.config.hidden_size // self.config.heads

        self.norm_factor = math.sqrt(
            math.sqrt(self.config.hidden_size // self.config.heads))

        self.score_context_func = inference_cuda_module.softmax_context_fp32 if (not config.fp16) else \
                                    inference_cuda_module.softmax_context_fp16
        self.linear_func = inference_cuda_module.linear_layer_fp16 if config.fp16 else \
                                    inference_cuda_module.linear_layer_fp32
        self.cuda_graph_created = False
        self.enable_cuda_graph = False

    def _graph_replay(self, *inputs, **kwargs):
        for i in range(len(inputs)):
            if torch.is_tensor(inputs[i]):
                self.static_inputs[i].copy_(inputs[i])
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                self.static_kwargs[k].copy_(kwargs[k])
        self._cuda_graphs.replay()
        return self.static_output

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

    def forward(self, *inputs, **kwargs):
        if self.enable_cuda_graph:
            if self.cuda_graph_created:
                outputs = self._graph_replay(*inputs, **kwargs)
            else:
                self._create_cuda_graph(*inputs, **kwargs)
                outputs = self._graph_replay(*inputs, **kwargs)
        else:
            outputs = self._forward(*inputs, **kwargs)
        return outputs

    def _forward(self, input, context=None, input_mask=None):
        output = DeepSpeedAttentionFunction.apply(input,
                                                  context,
                                                  input_mask,
                                                  self.config,
                                                  self.attn_qkvw,
                                                  self.attn_qw,
                                                  self.attn_kw,
                                                  self.attn_vw,
                                                  self.attn_qkvb,
                                                  self.num_attention_heads_per_partition,
                                                  self.norm_factor,
                                                  self.hidden_size_per_partition,
                                                  self.attn_ow,
                                                  self.attn_ob,
                                                  self.score_context_func,
                                                  self.linear_func,
                                                  self.triton_flash_attn_kernel)

        return output
