'''
Copyright 2021 The Microsoft DeepSpeed Team
'''
import torch
import os
from torch.nn.modules import Module
import torch.distributed as dist
from ..runtime.state_dict_factory import SDLoaderFactory
from ..runtime.weight_quantizer import WeightQuantization
from ..module_inject.replace_module import replace_transformer_layer
from ..utils import logger, init_distributed

from ..pipe import PipelineModule


class InferenceEngine(Module):
    inference_mp_group = None

    def __init__(self,
                 model,
                 mp_size=1,
                 mpu=None,
                 checkpoint=None,
                 dtype=None,
                 injection_dict=None,
                 return_tuple=True,
                 replace_method='auto',
                 quantization_setting=None,
                 replace_with_kernel_inject=False):
        """
        Args:
            model: torch.nn.Module
            mp_size: model-parallel size
            mpu: model-parallel unit (used for Megatron-type models)
            checkpoint: the json-path, showing the address of model-checkpoints
                Example: {type: 'Megatron', 'checkpoints': [ckpt_mp0.pt, ckpt_mp1.pt], 'version': 1.0}
            dtype: data-type by which inference is executed
            injection_dict: the dictionary that shows the injection policy:
                Example: {BertLayer: HFBertLayerPolicy}
            return_tuple: if true, inference-API returns a tuple, otherwise a tensor
            replace_method: the injection method, this can be passed as auto if no injection-policy is defined, in which case the injection is automatic based on the available policies
            quantization_setting:
                one of None, Tuple(mlp_extra_grouping, quantize_groups), quantize_groups
        """

        super().__init__()

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

        self._init_quantization_setting(quantization_setting)

        if self.checkpoint:
            self._load_checkpoint(self.checkpoint)

        # convert model to intended dtype
        if self.dtype:
            self._convert_to_dtype()

        if self.mpu:
            self.mp_world_size = dist.get_world_size(
                group=self.mpu.get_model_parallel_group())
            self.mp_group = self.mpu.get_model_parallel_group()
        elif self.mp_world_size > 1:
            self._create_model_parallel_group()
        # apply injection policy
        if self.injection_dict is not None:
            for client_module, injection_policy in self.injection_dict.items():
                self._apply_injection_policy(client_module,
                                             injection_policy,
                                             return_tuple,
                                             replace_with_kernel_inject)
        elif replace_method == 'auto':
            self._apply_injection_policy(
                return_tuple=return_tuple,
                replace_with_kernel_inject=replace_with_kernel_inject)

        device = torch.cuda.current_device()
        logger.info(f"Place model to device: {device}")
        self.module.to(device)

        if self.mp_world_size > 1:
            self.model_orig_fwd = self.module.forward
            self.module.forward = self.forward
        else:
            self.module.register_forward_pre_hook(self._pre_forward_hook)

    def _get_model_config_generate(self):
        self.config = getattr(self.module, 'config', None)
        self.generate = getattr(self.module, 'generate', None)

    def _create_model_parallel_group(self):
        # Call the init process
        if InferenceEngine.inference_mp_group is None:
            init_distributed()

            local_rank = int(os.getenv('LOCAL_RANK', '0'))
            torch.cuda.set_device(local_rank)

            ranks = [i for i in range(self.mp_world_size)]
            self.mp_group = dist.new_group(ranks)
            InferenceEngine.inference_mp_group = self.mp_group
        else:
            self.mp_group = InferenceEngine.inference_mp_group

    def _init_quantization_setting(self, quantization_setting):
        self.quantize_bits = 8
        self.mlp_extra_grouping = False
        self.quantize_groups = 1
        if type(quantization_setting) is tuple:
            self.mlp_extra_grouping, \
            self.quantize_groups = quantization_setting
        elif quantization_setting is not None:
            self.quantize_groups = quantization_setting
        logger.info(f"quantize_bits = {self.quantize_bits} "
                    f"mlp_extra_grouping = {self.mlp_extra_grouping}, "
                    f"quantize_groups = {self.quantize_groups}")

    def _validate_args(self, mpu):
        if not isinstance(self.module, Module):
            raise ValueError(f"model must be a torch.nn.Module, got {type(self.module)}")
        if not isinstance(self.mp_world_size, int) or self.mp_world_size < 1:
            raise ValueError(f"mp_size must be an int >= 1, got {self.mp_world_size}")

        if mpu:
            methods = ["get_model_parallel_group", "get_data_parallel_group"]
            for method in methods:
                if not hasattr(mpu, method):
                    raise ValueError(f"mpu is missing {method}")
        if self.checkpoint is not None and not isinstance(self.checkpoint, str):
            raise ValueError(
                f"checkpoint must be None or a str, got {type(self.checkpoint)}")

        supported_dtypes = [None, torch.half, torch.int8, torch.float]
        if self.dtype not in supported_dtypes:
            raise ValueError(
                f"{self.dtype} not supported, valid dtype: {supported_dtypes}")

        if self.injection_dict is not None and not isinstance(self.injection_dict, dict):
            raise ValueError(
                f"injection_dict must be None or a dict, got: {self.injection_dict}")

    def _apply_injection_policy(self,
                                client_module=None,
                                injection_policy=None,
                                return_tuple=True,
                                replace_with_kernel_inject=False):

        replace_transformer_layer(client_module,
                                  self.module,
                                  policy=injection_policy,
                                  mp_size=self.mp_world_size,
                                  mp_group=self.mp_group,
                                  config=self.config,
                                  fp16=(self.dtype == torch.half),
                                  training=False,
                                  return_tuple=return_tuple,
                                  quantize=(self.dtype == torch.int8),
                                  quantize_settings=(self.quantization_scales,
                                                     self.quantize_merge_count,
                                                     self.mlp_extra_grouping,
                                                     self.quantize_groups),
                                  replace_with_kernel_inject=replace_with_kernel_inject)

    def _load_checkpoint(self, load_dir, load_module_strict=True):
        sd_loader = SDLoaderFactory.get_sd_loader_json(load_dir)
        is_pipe_parallel = isinstance(self.module, PipelineModule)

        if is_pipe_parallel:
            raise RuntimeError(
                'pipeline parallelism is currently not supported in inference.')

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
                                                                        self.quantize_bits,
                                                                        self.quantize_groups)
        elif self.dtype == torch.half:
            self.module.half()
        elif self.dtype == torch.float:
            self.module.float()

    def _pre_forward_hook(self, module, *inputs, **kwargs):
        for input in inputs:
            if torch.is_tensor(input):
                input = input.to(torch.cuda.current_device())
        for k in kwargs:
            if torch.is_tensor(kwargs[k]):
                kwargs[k] = kwargs[k].to(torch.cuda.current_device())

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
                        if not input.is_contiguous():
                            input = input.contiguous()
                for k in kwargs:
                    if torch.is_tensor(kwargs[k]):
                        kwargs[k] = kwargs[k].to(torch.cuda.current_device())
                        if not kwargs[k].is_contiguous():
                            kwargs[k] = kwargs[k].contiguous()
                        dist.broadcast(kwargs[k], 0)

            outputs = self.model_orig_fwd(*inputs, **kwargs)
        else:
            outputs = self.module(*inputs, **kwargs)
        return outputs
