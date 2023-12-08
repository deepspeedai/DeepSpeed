# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.module_inject.replace_policy import replace_policies
from deepspeed.module_inject.utils import policy_to_ds_container
from .engine import DeepSpeedEngine
from .utils import TLinear, get_inactive_params
from deepspeed.runtime.zero import GatheredParameters
import time
import gc
import math
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from torch import nn
from deepspeed.utils import logger

from deepspeed.ops.op_builder import InferenceBuilder

from deepspeed.module_inject.layers import LinearLayer, Normalize, EmbeddingLayer, OPTEmbedding
try:
    import transformers
    OPTLearnedPositionalEmbedding = transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding
except:
    OPTLearnedPositionalEmbedding = None
inference_cuda_module = None


class DeepSpeedHybridEngine(DeepSpeedEngine):
    r"""DeepSpeed engine for training and inference."""
    inference_mp_group = None

    def __init__(self, args, model, **kwargs):

        super().__init__(args, model, **kwargs)

        # synch seed between all GPUs
        _rng_state = get_accelerator().get_rng_state().to(get_accelerator().current_device_name())
        dist.broadcast(_rng_state, 0)
        get_accelerator().set_rng_state(_rng_state.cpu())

        self.Z3_enabled = (self._config.zero_config.stage == 3)
        self.gather_all_layers = self._config.hybrid_engine.pin_parameters

        # inference containers / fwds
        self._inference_containers = []
        self._orig_modules = []
        self._orig_fwds = []
        self.create_inference_module()

        # Performance stats
        self._t_start = None
        self._total_latency = 0
        self._iters = 0
        self._training_start_time = None
        self._generate_latency = 0
        self._training_latency = 0
        self._total_batch_size = None
        self._gather_latency = 0

        global inference_cuda_module
        if inference_cuda_module is None:
            builder = InferenceBuilder()
            inference_cuda_module = builder.load()

        self.is_lora_fused = False

    def convert_to_linear_transposed(self, model):

        def _replace_linear_layer(r_module, parent_type=None, prev_type=None):
            for name, child in r_module.named_children():
                if child.__class__ in [torch.nn.Linear] and \
                    (parent_type is torch.nn.ModuleList or prev_type is torch.nn.ModuleList):
                    setattr(r_module, name, TLinear(child, name))
                else:
                    _replace_linear_layer(child, type(r_module), prev_type=parent_type)
            return r_module

        _replace_linear_layer(model)

    def new_inference_container(self, orig_layer, policy_cls, layer_id):
        policy = policy_cls(orig_layer, inference=True)

        if self._config.fp16_enabled:
            inference_dtype = torch.float16
        elif self._config.bfloat16_enabled:
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        _container = policy_to_ds_container(
            policy=policy,
            config=DeepSpeedInferenceConfig(
                set_empty_params=True,
                dtype=inference_dtype,
                max_out_tokens=self._config.hybrid_engine.max_out_tokens,
                min_out_tokens=self._config.hybrid_engine.max_out_tokens,
                transposed_mode=True,
            ),
            model_config=self.module.config if hasattr(self.module, 'config') else None,
            layer_id=layer_id,
            child=orig_layer)

        if self.mpu is not None:
            if hasattr(self.mpu, 'get_model_parallel_world_size'):
                _container.set_tensor_parallel_config(self.mpu.get_model_parallel_world_size(),
                                                      self.mpu.get_model_parallel_group())
            else:
                _container.set_tensor_parallel_config(self.mpu.get_tensor_model_parallel_world_size(),
                                                      self.mpu.get_tensor_model_parallel_group())
        else:
            _container.set_tensor_parallel_config(self._config.hybrid_engine.inference_tp_size, self.mp_group)
        _container.initialize_tensors(enable_training=True)
        _container.create_ds_model_config()
        _container.create_module()
        _container.set_params_wo_copy(Z3_enabled=self.Z3_enabled)
        return _container

    def populate_all_inference_policies(self):
        self.inference_policies = {}
        for plcy in replace_policies:
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    self.inference_policies.update({orig_layer_class: (self.new_inference_container, plcy)})
            elif plcy._orig_layer_class is not None:
                self.inference_policies.update({plcy._orig_layer_class: (self.new_inference_container, plcy)})
        self.inference_policies.update({
            nn.Linear: (LinearLayer, ),
            nn.Embedding: (EmbeddingLayer, ),
            nn.LayerNorm: (Normalize, ),
            OPTLearnedPositionalEmbedding: (OPTEmbedding, )
        })

    def _fuse_lora_layer(self, layer_id):
        self._inference_containers[layer_id].fuse_lora()

    def fuse_lora_weight(self):
        for layer_id in range(len(self.layer_params)):
            self._fuse_lora_layer(layer_id)

    def _unfuse_lora_layer(self, layer_id):
        self._inference_containers[layer_id].unfuse_lora()

    def unfuse_lora_weight(self):
        for layer_id in range(len(self.layer_params)):
            self._unfuse_lora_layer(layer_id)

    def unfuse_lora_weight_non_pinned(self):
        for layer_id in range(len(self.layer_params)):
            non_active_params = get_inactive_params(self.layer_params[layer_id])
            non_active_lora_params = get_inactive_params(self.layer_lora_params[layer_id])
            non_active_params.extend(non_active_lora_params)

            with GatheredParameters(non_active_params):
                self._unfuse_lora_layer(layer_id)

    def retake_inference_cache(self):
        if self._config.hybrid_engine.release_inference_cache:
            retake_success = inference_cuda_module.retake_workspace()

            if not retake_success:
                logger.warning("Unable to acquire workspace on first attempt, emptying cache and retrying.")
                gc.collect()
                get_accelerator().empty_cache()
                retake_success = inference_cuda_module.retake_workspace()

                if not retake_success:
                    raise RuntimeError("Unable to retake inference workspace.")

    def generate(self, *inputs, **kwargs):
        if self._total_batch_size is None:
            bsz = inputs[0].shape[0] if len(inputs) > 0 else \
                kwargs['input_ids'].shape[0]
            self._total_batch_size = bsz * dist.get_world_size()

        self._t0 = time.time()

        if self.Z3_enabled and self.gather_all_layers:
            if self._config.hybrid_engine.inference_tp_size > 1:
                non_tp_params = []
                for other_layer in self._other_layers:
                    non_tp_params.extend(list(other_layer.parameters()))

                partition_size = self._config.hybrid_engine.tp_gather_partition_size

                layer_groups = math.ceil(len(self.layer_params) / partition_size)
                for lg in range(layer_groups):
                    non_active_params = []
                    non_active_lora_params = []
                    for layer_id in range(lg * partition_size, min(len(self.layer_params), (lg + 1) * partition_size),
                                          1):
                        non_tp_params.extend(self.layer_params[layer_id][:4])
                        non_active_params.extend(get_inactive_params(self.layer_params[layer_id]))
                        non_active_params.extend(get_inactive_params(self.layer_lora_params[layer_id]))
                    with GatheredParameters(non_active_params):
                        for layer_id in range(lg * partition_size,
                                              min(len(self.layer_params), (lg + 1) * partition_size), 1):
                            if len(self.all_lora_params) > 0:
                                self._fuse_lora_layer(layer_id)

                            if self.mpu is not None:
                                self._inference_containers[layer_id].apply_tensor_parallelism(self.mp_replace,
                                                                                              reversed_dim=True)

                # TODO(cmikeh2) Evaluate if this can be deferred when release_inference_cache
                # is enabled.
                gc.collect()
                get_accelerator().empty_cache()

                self._gather_latency = time.time() - self._t0

                input_shape = inputs[0].shape if len(inputs) > 0 else \
                                kwargs['input_ids'].shape
                output = torch.zeros(
                    (input_shape[0] * self._config.hybrid_engine.inference_tp_size, ) + input_shape[1:],
                    dtype=inputs[0].dtype if len(inputs) > 0 else kwargs['input_ids'].dtype,
                    device=inputs[0].device if len(inputs) > 0 else kwargs['input_ids'].device)
                input_cont = inputs[0].contiguous() if len(inputs) > 0 else kwargs['input_ids'].contiguous()
                dist.all_gather_into_tensor(output, input_cont, group=self.mp_group)

                if len(inputs) > 0:
                    inputs = (output, *inputs[1:])
                else:
                    kwargs['input_ids'] = output

                self.retake_inference_cache()

                non_active_params = get_inactive_params(non_tp_params)
                with GatheredParameters(non_active_params):
                    generate_ret_vals = self._generate(*inputs, **kwargs)

                for layer_id in range(len(self.layer_params)):
                    self._inference_containers[layer_id].release_memory()

                rank = dist.get_rank(group=self.mp_group)
                generate_ret_vals = generate_ret_vals[input_shape[0] * rank:input_shape[0] * (rank + 1)]

            else:
                non_active_layers = get_inactive_params(self.all_layers_params)
                non_active_lora_params = get_inactive_params(self.all_lora_params)
                non_active_layers.extend(non_active_lora_params)
                with GatheredParameters(non_active_layers):
                    self._gather_latency = time.time() - self._t0

                    if len(self.all_lora_params) > 0:
                        self.fuse_lora_weight()

                    self.retake_inference_cache()
                    generate_ret_vals = self._generate(*inputs, **kwargs)

                    if len(self.all_lora_params) > 0:
                        self.unfuse_lora_weight()
        else:
            if len(self.all_lora_params) > 0 and (not self.Z3_enabled):
                self.fuse_lora_weight()

            self.retake_inference_cache()
            generate_ret_vals = self._generate(*inputs, **kwargs)

            if len(self.all_lora_params) > 0:
                if (not self.Z3_enabled):
                    self.unfuse_lora_weight()
                else:
                    self.unfuse_lora_weight_non_pinned()
                self.is_lora_fused = False

        if self._config.hybrid_engine.release_inference_cache:
            inference_cuda_module.release_workspace()
            gc.collect()
            get_accelerator().empty_cache()

        self._generate_latency = time.time() - self._t0 - self._gather_latency

        return generate_ret_vals

    def create_inference_containers(self, module, layer_id=0):
        for name, child in module.named_children():
            if child.__class__ in self.inference_policies:
                if self.inference_policies[child.__class__][0] == self.new_inference_container:
                    self._inference_containers.append(self.inference_policies[child.__class__][0](
                        child, self.inference_policies[child.__class__][-1], layer_id))
                    self._orig_modules.append(child)
                    self._orig_fwds.append(child.forward)

                    self.layer_params.append(self._inference_containers[layer_id].get_all_params())

                    self.lora_params.append(self._inference_containers[layer_id].get_lora_params())
                    self.layer_lora_params.append([])
                    for lora_param in self.lora_params[layer_id]:
                        self.layer_lora_params[layer_id].extend(lora_param[:-1])
                        self.all_lora_params.extend(lora_param[:-1])

                    layer_id += 1
                else:
                    self._other_layers.append(self.inference_policies[child.__class__][0](
                        weight=child.weight, bias=child.bias if hasattr(child, 'bias') else None))
                    self._orig_modules_others.append(child)
                    self._orig_fwds_others.append(child.forward)
            else:
                self.create_inference_containers(child, layer_id=layer_id)

    def create_inference_module(self):
        self.layer_params = []
        self.layer_lora_params = []
        self.lora_params = []
        self.all_lora_params = []

        self._other_layers = []
        self._orig_modules_others = []
        self._orig_fwds_others = []

        if self._config.hybrid_engine.inference_tp_size > 1:
            if self.mpu is None:
                global_rank = dist.get_rank()
                world_size = dist.get_world_size()
                mp_group_id = global_rank // self._config.hybrid_engine.inference_tp_size
                num_mp_groups = world_size // self._config.hybrid_engine.inference_tp_size
                for mp_group_id in range(num_mp_groups):
                    ranks = list(
                        range(mp_group_id * self._config.hybrid_engine.inference_tp_size, \
                            (mp_group_id + 1) * self._config.hybrid_engine.inference_tp_size, \
                            1)
                    )
                    mp_group = dist.new_group(ranks)
                    if global_rank in ranks:
                        # mp_group is used for broader collective
                        self.mp_group = mp_group

                        # mp_replace is used for container tensor slicing
                        from deepspeed.module_inject import ReplaceWithTensorSlicing
                        self.mp_replace = ReplaceWithTensorSlicing(
                            mp_group=self.mp_group,
                            mp_size=self._config.hybrid_engine.inference_tp_size,
                            out_dim=0,
                            in_dim=1)

            else:
                self.mp_group = self.mpu.get_model_parallel_group() if hasattr(self.mpu, 'get_model_parallel_group') else \
                    self.mpu.get_tensor_model_parallel_group()

                from deepspeed.module_inject import ReplaceWithTensorSlicing
                self.mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group,
                                                           mp_size=self._config.hybrid_engine.inference_tp_size,
                                                           out_dim=0,
                                                           in_dim=1)
        else:
            self.mp_group = None
            self.mp_replace = None
        self.populate_all_inference_policies()
        self.all_layers_params = list(self.module.parameters())
        self.create_inference_containers(self.module)

        if len(self._inference_containers) > 0:
            self._generate = self.module.generate
            self.module.generate = self.generate

        self._t0 = time.time()

    def _zero3_forward(self, layer_id):

        def run_forward(*inputs, **kwargs):
            non_active_params = get_inactive_params(self.layer_params[layer_id])
            non_active_lora_params = get_inactive_params(self.layer_lora_params[layer_id])
            non_active_params.extend(non_active_lora_params)

            with GatheredParameters(non_active_params):
                if len(self.all_lora_params) > 0:
                    # Use the is_lora_fused flag to prevent multiple fusion in Z3 with non-pinned memory
                    if not self.is_lora_fused:
                        self._fuse_lora_layer(layer_id)
                    # Set the is_lora_fused to true when reaching the last layer
                    if layer_id == len(self.layer_params) - 1:
                        self.is_lora_fused = True
                return self._inference_containers[layer_id].module.forward(*inputs, **kwargs)

        return run_forward

    def eval(self):
        if self._t_start is not None:
            latency = time.time() - self._t_start
            self._total_latency = self._total_latency + latency
            self._iters = self._iters + 1
            if not dist.is_initialized() or dist.get_rank() == 0:
                if self._total_batch_size is not None:
                    cur_samples_p_sec = f'|CurSamplesPerSec={(1 / latency * self._total_batch_size):.2f} '
                    avg_samples_p_sec = f'|AvgSamplesPerSec={(1 / (self._total_latency / self._iters) * self._total_batch_size):.2f}'
                else:
                    cur_samples_p_sec = ''
                    avg_samples_p_sec = ''
                others = latency - (self._generate_latency + self._training_latency)
                print(f'|E2E latency={(latency):.2f}s ' + \
                      f'|Gather latency={self._gather_latency:.2f}s ({(self._gather_latency / latency * 100):.2f}%) '
                      f'|Generate time={(self._generate_latency):.2f}s ({(self._generate_latency / latency * 100):.2f}%) ' + \
                      f'|Training time={(self._training_latency):.2f}s ({(self._training_latency / latency * 100):.2f}%) ' + \
                      f'|Others={others:.2f} ({(others / latency * 100):.2f}%)' + \
                      cur_samples_p_sec + \
                      avg_samples_p_sec)
            self._t_start = time.time()
        self._training_latency = 0
        super().eval()
        if len(self._inference_containers) > 0:
            for i, (orig_module, inference_container) in enumerate(zip(self._orig_modules,
                                                                       self._inference_containers)):
                if self.Z3_enabled and not self.gather_all_layers:
                    orig_module.forward = self._zero3_forward(i)
                else:
                    orig_module.forward = inference_container.module.forward

                inference_container.transform_for_inference()

            if not self.Z3_enabled or self.gather_all_layers:
                for orig_module, inference_layer in zip(self._orig_modules_others, self._other_layers):
                    orig_module.forward = inference_layer.forward
        if self.Z3_enabled:
            gc.collect()
            get_accelerator().empty_cache()
        if self._t_start is None:
            self._t_start = time.time()

    def train(self, mode=True):
        if mode and len(self._orig_modules) > 0:
            for inference_container, orig_module, orig_fwd in zip(self._inference_containers, self._orig_modules,
                                                                  self._orig_fwds):
                inference_container.transform_for_training()
                orig_module.forward = orig_fwd
            for orig_module, orig_fwd in zip(self._orig_modules_others, self._orig_fwds_others):
                orig_module.forward = orig_fwd
        super().train(mode)
        if mode:
            self._training_start_time = time.time()

    def step(self, lr_kwargs=None):
        super().step(lr_kwargs=lr_kwargs)

        if len(self._inference_containers) > 0:
            if not self.Z3_enabled:
                for inference_container in self._inference_containers:
                    inference_container.reset_params()

        if self._training_start_time is not None:
            self._training_latency += (time.time() - self._training_start_time)
            self._training_start_time = time.time()
