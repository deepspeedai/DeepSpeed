# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from ..config import DeepSpeedInferenceConfig
from .base import BaseOp

minus_inf = -10000.0
key_idx = 0
value_idx = 1


class InferenceContext:

    __instance = None

    def __init__(self):
        self.kv_cache = None
        self.kv_cache_elem_dtype = None
        self.num_tokens = 1
        self.kv_cache_num_layers = None
        self.kv_cache_size = None
        self.max_out_tokens = None
        self.rotary = None
        self.allocate_called = False
        self.static_shapes = True

    @classmethod
    def Instance(cls):
        if InferenceContext.__instance is None:
            InferenceContext.__instance = InferenceContext()
        return InferenceContext.__instance

    def gen_workspace(self, num_layers, num_heads, batch_size, prompt_len, hidden_dim, mp_size, external_cache,
                      elem_dtype, rank, max_out_tokens, min_out_tokens):
        self.allocate_called = True
        self.kv_cache = None
        if not external_cache:
            self.kv_cache_num_layers = num_layers
            self.max_out_tokens = max_out_tokens
            head_size = hidden_dim // num_heads
            self.kv_cache_size = torch.Size([batch_size, (num_heads // mp_size), max_out_tokens, head_size])
            self.kv_cache_elem_dtype = elem_dtype
        self.num_tokens = 0
        self.static_shapes = True
        return True

    def retake_workspace(self):
        return True

    def _retake_workspace(self):
        assert self.allocate_called, "retake workspace called before allocate workspace"

        import deepspeed.accelerator as accelerator
        if self.kv_cache is None:
            self.kv_cache = []
            for layer in range(self.kv_cache_num_layers):
                self.kv_cache.append((torch.zeros(self.kv_cache_size,
                                                  dtype=self.kv_cache_elem_dtype,
                                                  device=accelerator.get_accelerator().device_name()),
                                      torch.zeros(self.kv_cache_size,
                                                  dtype=self.kv_cache_elem_dtype,
                                                  device=accelerator.get_accelerator().device_name())))

        return True

    def update_cache(self, layer_id, token_idx, is_prompt, bat_0213_key, bat_0213_value):
        has_workspace = self._retake_workspace()
        assert has_workspace, "Could not allocate workspace"

        # Update current token
        if is_prompt:
            self.static_shapes = True
            if token_idx is None:
                self.static_shapes = False
                InferenceContext.Instance().reset_tokens(bat_0213_key.shape[2])
            else:
                InferenceContext.Instance().reset_tokens(token_idx)

        if token_idx is None:
            token_idx = InferenceContext.Instance().current_tokens()

        bsz = bat_0213_key.shape[0]

        # Update cache content
        if is_prompt:
            cache_max_seq = self.kv_cache_size[2]
            cache_max_head_dim = self.kv_cache_size[3]
            seq = bat_0213_key.shape[2]

            mask = torch.arange(cache_max_seq, device=bat_0213_key.device)
            mask = mask.ge(token_idx)
            mask = mask.unsqueeze(-1)
            mask = mask.expand([cache_max_seq, cache_max_head_dim])

            self.kv_cache[layer_id][key_idx][:bsz, :, :seq, :].copy_(bat_0213_key)
            self.kv_cache[layer_id][key_idx][:bsz, :].masked_fill_(mask, 0)
            self.kv_cache[layer_id][value_idx][:bsz, :, :seq, :].copy_(bat_0213_value)
            self.kv_cache[layer_id][value_idx][:bsz, :].masked_fill_(mask, 0)
        else:
            if self.static_shapes:
                assert type(token_idx) == torch.Tensor, "token_idx is expected to be torch.Tensor"
                self.kv_cache[layer_id][key_idx][:bsz].index_copy_(2, token_idx - 1, bat_0213_key)
                self.kv_cache[layer_id][value_idx][:bsz].index_copy_(2, token_idx - 1, bat_0213_value)
            else:
                assert type(token_idx) == int, "token_idx is expected to be int"
                self.kv_cache[layer_id][key_idx][:bsz, :, token_idx - 1:token_idx, :] = bat_0213_key
                self.kv_cache[layer_id][value_idx][:bsz, :, token_idx - 1:token_idx, :] = bat_0213_value

        bat_0213_key = self.kv_cache[layer_id][key_idx][:bsz]
        bat_0213_value = self.kv_cache[layer_id][value_idx][:bsz]

        if not self.static_shapes:
            bat_0213_key = bat_0213_key[:, :, :token_idx, :]
            bat_0213_value = bat_0213_value[:, :, :token_idx, :]

        return bat_0213_key, bat_0213_value

    def release_workspace(self):
        self.kv_cache = None
        self.rotary = None

    def reset_tokens(self, initial_tokens=1):
        self.num_tokens = initial_tokens

    def current_tokens(self):
        return self.num_tokens

    def advance_tokens(self):
        self.num_tokens = self.num_tokens + 1

    def get_kv_cache(self):
        return self.kv_cache

    def get_rotary(self, rotary_dim, rope_theta, device=None):
        if self.rotary is None:
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

            self.rotary = LlamaRotaryEmbedding(rotary_dim, base=rope_theta, device=device)

        return self.rotary

    def get_max_tokens_num(self):
        return self.max_out_tokens


class WorkspaceOp(BaseOp):

    def __init__(self, config: DeepSpeedInferenceConfig = None):
        if config is None:
            config = DeepSpeedInferenceConfig()
        self.inference_context = InferenceContext.Instance()
        self._is_allocated = False
        try:
            super(WorkspaceOp, self).__init__(config)
            if config.dtype == torch.float32:
                self.allocate_workspace_func = self.inference_module.allocate_workspace_fp32
            elif config.dtype == torch.bfloat16:
                self.allocate_workspace_func = self.inference_module.allocate_workspace_bf16
            else:
                self.allocate_workspace_func = self.inference_module.allocate_workspace_fp16
            self.release_workspace_func = self.inference_module.release_workspace
            self.retake_workspace_func = self.inference_module.retake_workspace
            self.reset_cache_func = self.inference_module.reset_cache
        except (ValueError, AttributeError) as e:
            print(f"Using fallback functions in workspace because of {e}")
            if config.dtype == torch.float32:
                self.allocate_workspace_func = self.allocate_workspace_fp32_fallback
            elif config.dtype == torch.bfloat16:
                self.allocate_workspace_func = self.allocate_workspace_bf16_fallback
            else:
                self.allocate_workspace_func = self.allocate_workspace_fp16_fallback
            self.release_workspace_func = self.release_workspace_fallback
            self.retake_workspace_func = self.retake_workspace_fallback
            self.reset_cache_func = self.reset_cache_fallback

    def allocate_workspace(self, *args, **kwargs):
        self._is_allocated = True
        return self.allocate_workspace_func(*args, **kwargs)

    def release_workspace(self):
        self._is_allocated = False
        return self.release_workspace_func()

    def reset_cache(self):
        return self.reset_cache_func() if self.reset_cache_func else None

    def retake_workspace(self):
        return self.retake_workspace_func() if self.retake_workspace_func else None

    def allocate_workspace_fp32_fallback(self, hidden_dim, num_heads, prompt_length, batch_size, num_layers, mp_size,
                                         external_cache, rank, max_out_tokens, min_out_tokens):
        return self.inference_context.gen_workspace(num_layers, num_heads, batch_size, prompt_length, hidden_dim,
                                                    mp_size, external_cache, torch.float, rank, max_out_tokens,
                                                    min_out_tokens)

    def allocate_workspace_bf16_fallback(self, hidden_dim, num_heads, prompt_length, batch_size, num_layers, mp_size,
                                         external_cache, rank, max_out_tokens, min_out_tokens):
        return self.inference_context.gen_workspace(num_layers, num_heads, batch_size, prompt_length, hidden_dim,
                                                    mp_size, external_cache, torch.bfloat16, rank, max_out_tokens,
                                                    min_out_tokens)

    def allocate_workspace_fp16_fallback(self, hidden_dim, num_heads, prompt_length, batch_size, num_layers, mp_size,
                                         external_cache, rank, max_out_tokens, min_out_tokens):
        return self.inference_context.gen_workspace(num_layers, num_heads, batch_size, prompt_length, hidden_dim,
                                                    mp_size, external_cache, torch.half, rank, max_out_tokens,
                                                    min_out_tokens)

    def reset_cache_fallback(self):
        return self.inference_context.reset_tokens()

    def release_workspace_fallback(self):
        return self.inference_context.release_workspace()

    def retake_workspace_fallback(self):
        return self.inference_context.retake_workspace()

    def is_allocated(self):
        return self._is_allocated
