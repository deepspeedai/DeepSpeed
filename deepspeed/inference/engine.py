'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
import torch
import os
from datetime import timedelta
from torch.nn.modules import Module
import torch.distributed as dist
from ..runtime.state_dict_factory import SDLoaderFactory
from ..runtime.weight_quantizer import WeightQuantization
from ..module_inject.replace_module import replace_transformer_layer
from ..constants import INFERENCE_GENERIC_MODE
from ..module_inject import replace_policy
from ..utils import logger, init_distributed

from ..pipe import PipelineModule


class InferenceEngine(Module):
    def __init__(self,
                 model,
                 mp_size=1,
                 mpu=None,
                 checkpoint=None,
                 dtype=None,
                 injection_dict=None,
                 replace_method='auto',
                 quantization_setting=None):

        super(InferenceEngine, self).__init__()

        self.module = model

        self._get_model_config_generate()

        self.mp_world_size = mp_size
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.injection_dict = injection_dict
        self.mp_group = None
        self.mpu = mpu
        self._validate_args(mpu)
        self.replace_method = replace_method
        self.quantize_merge_count = 1
        self.quantization_scales = None

        if self.mpu:
            self.mp_world_size = dist.get_world_size(
                group=self.mpu.get_model_parallel_group())
            self.mp_group = self.mpu.get_model_parallel_group()
        elif self.mp_world_size > 1 and not dist.is_initialized():
            self._create_model_parallel_group()
        else:
            self.module.to(torch.cuda.current_device())

        self._check_quantize_setting(quantization_setting)

        if self.checkpoint:
            self._load_checkpoint(self.checkpoint)

        # convert model to intended dtype
        if self.dtype:
            self._convert_to_dtype()

        # apply injection policy
        if self.injection_dict:
            for client_module, injection_policy in self.injection_dict.items():
                self._apply_injection_policy(client_module, injection_policy)
        elif replace_method == 'auto':
            self._apply_injection_policy()

        if self.mp_world_size > 1:
            self.model_orig_fwd = self.module.forward
            self.module.forward = self.forward
        else:
            self.module.register_forward_pre_hook(self._pre_forward_hook)

    def _get_model_config_generate(self):
        if hasattr(self.module, 'config'):
            self.config = self.module.config
        else:
            self.config = None
        if hasattr(self.module, 'generate'):
            self.generate = self.module.generate
        else:
            self.generate = None

    def _create_model_parallel_group(self):
        # Call the init process

        init_distributed()

        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)

        ranks = [i for i in range(self.mp_world_size)]
        self.mp_group = dist.new_group(ranks)

        self.module.to(torch.cuda.current_device())
        for p in self.module.parameters():
            if torch.is_tensor(p):
                dist.broadcast(p, 0)

    def _check_quantize_setting(self, quantization_setting):
        self.quatize_bits = 8
        self.mlp_extra_grouping = False
        self.quantize_groups = 1
        if quantization_setting is None:
            return
        elif type(quantization_setting) is tuple:
            self.mlp_extra_grouping, \
            self.quantize_groups = quantization_setting
        else:
            self.quantize_groups = quantization_setting

    def _validate_args(self, mpu):
        assert isinstance(self.module, Module)
        assert isinstance(self.mp_world_size, int)
        assert self.mp_world_size >= 1

        if mpu:
            methods = [f"get_model_parallel_group"]
            methods.extend([f"get_data_parallel_group"])
            for method in methods:
                assert hasattr(mpu, method), f"mpu is missing {method}"

        assert self.checkpoint is None or isinstance(self.checkpoint, str)

        supported_dtypes = [torch.half, torch.int8, torch.float]
        assert self.dtype is None or self.dtype in supported_dtypes, f"dtype={self.dtype} is not in the \
            list of supported dtypes {supported_dtypes}"

        assert self.injection_dict is None or isinstance(self.injection_dict, dict)

    def _apply_injection_policy(self, client_module=None, injection_policy=None):
        replace_transformer_layer(client_module,
                                  self.module,
                                  policy=injection_policy,
                                  mp_size=self.mp_world_size,
                                  mp_group=self.mp_group,
                                  config=self.config,
                                  fp16=(self.dtype == torch.half),
                                  training=False,
                                  quantize=(self.dtype == torch.int8),
                                  quantize_settings=(self.quantization_scales,
                                                     self.quantize_merge_count,
                                                     self.mlp_extra_grouping,
                                                     self.quantize_groups))

    def _load_checkpoint(self, load_dir, load_module_strict=True):
        sd_loader = SDLoaderFactory.get_sd_loader_json(load_dir)
        is_pipe_parallel = isinstance(self.module, PipelineModule)

        assert (not is_pipe_parallel),\
        'pipeline parallelism is currently not supported in inference.'

        mp_rank = 0 if self.mp_group is None else dist.get_rank(group=self.mp_group)

        load_path, checkpoint, quantize_config = sd_loader.load(self.mp_world_size,
                                                  mp_rank,
                                                  is_pipe_parallel=is_pipe_parallel,
                                                  quantize=(self.dtype is torch.int8),
                                                  quantize_groups=self.quantize_groups,
                                                  mlp_extra_grouping=self.mlp_extra_grouping)

        self.quantization_scales, self.quantize_merge_count = quantize_config

        if is_pipe_parallel:
            # Pipeline parallelism uses this to load its own checkpoint files.
            self._curr_ckpt_path = load_dir

        self.module.load_state_dict(state_dict=checkpoint['model'],
                                    strict=load_module_strict)

    def _convert_to_dtype(self):
        if self.dtype is torch.int8 and self.quantization_scales is None:
            quantizer = WeightQuantization(mlp_extra_grouping=self.mlp_extra_grouping)
            model, self.quantization_scales = quantizer.model_quantize(self.module,
                                                                        self.injection_dict,
                                                                        self.quatize_bits,
                                                                        self.quantize_groups)
        elif self.dtype == torch.half:
            self.module.half()
        elif self.dtype == torch.float:
            self.module.float()

    def _pre_forward_hook(self, module, *inputs, **kwargs):
        for input in inputs:
            if torch.is_tensor(input):
                input = input.to(torch.cuda.current_device())
                if self.mp_world_size > 1:
                    if not input.is_contiguous():
                        input = input.contiguous()
                    dist.broadcast(input, 0)

        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                kwargs[k] = kwargs[k].to(torch.cuda.current_device())
                if self.mp_world_size > 1:
                    if not kwargs[k].is_contiguous():
                        kwargs[k] = kwargs[k].contiguous()
                    dist.broadcast(kwargs[k], 0)

    def forward(self, *inputs, **kwargs):
        """Execute forward propagation

        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        if self.mp_world_size > 1:
            if self.mpu is None:
                for input in inputs:
                    if torch.is_tensor(input):
                        input = input.to(torch.cuda.current_device())
                        if self.mp_world_size > 1:
                            if not input.is_contiguous():
                                input = input.contiguous()
                            dist.broadcast(input, 0)

                for k in kwargs:
                    if torch.is_tensor(kwargs[k]):
                        kwargs[k] = kwargs[k].to(torch.cuda.current_device())
                        if self.mp_world_size > 1:
                            if not kwargs[k].is_contiguous():
                                kwargs[k] = kwargs[k].contiguous()
                            dist.broadcast(kwargs[k], 0)

            return self.model_orig_fwd(*inputs, **kwargs)
        return self.module(*inputs, **kwargs)
