'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch
import os
import copy
import collections
import json
from abc import ABC, abstractmethod
from deepspeed.utils import logger
from .weight_quantizer import WeightQuantization

AUTO_MODULE_KEY = 'auto'


class SDLoaderFactory:
    @staticmethod
    def get_sd_loader_json(json_file):
        with open(json_file) as f:
            data = json.load(f)
            sd_type = data['type']
            ckpt_list = data['checkpoints']
            version = data['version']
            return SDLoaderFactory.get_sd_loader(ckpt_list, sd_type, version)

    @staticmethod
    def get_sd_loader(ckpt_list, sd_type='Megatron', version=None):
        if sd_type == 'Megatron':
            return MegatronSDLoader(ckpt_list, version)
        else:
            assert False, '{} checkpoint type is not supported'.format(sd_type)


class SDLoaderBase(ABC):
    def __init__(self, ckpt_list, version):
        self.module_key = None
        self.ckpt_list = ckpt_list
        self.check_ckpt_list()
        self.version = version

    def load(self,
             mp_world_size,
             mp_rank,
             module_key=AUTO_MODULE_KEY,
             is_pipe_parallel=False,
             quantize=False,
             quantize_bits=8,
             quantize_groups=64,
             mlp_extra_grouping=True):
        self.module_key = module_key
        num_ckpt = len(self.ckpt_list)
        idx = mp_rank * num_ckpt // mp_world_size

        logger.info(
            f'mp_world_size: {mp_world_size}, mp_rank: {mp_rank}, module_key: {module_key}'
        )
        """ We have multiple cases to handle here for both training and inference:
            1. PipeModule loading mp_rank_*.pt files, is_pipe_parallel=True, module_key is not None
                a. if no mp_size/pp_size resizing occurs, for both training & inference, loading
                   the mp_rank related checkpoint directly.
                b. if has mp_size/pp_size resizing, only Megatron model inference is supported,
                   in this case each mp_rank_*.pt have same content, we will load the first checkpoint
                   file (idx=0), to avoid idx exceeding file list boundary.

            2. PipeModule loading layer_*.pt files, is_pipe_parallel=True, module_key is None
                a. if no mp_size resizing occurs, for both training & inference, loading
                   the mp_rank related checkpoint directly.
                b. if has mp_size resizing, only Megatron model inference is supported,
                   checkpoint file(s) will be merged/splitted according to mp_rank, mp_world_size and
                   checkpoint file list.

            3. Non-PipeModule loading mp_rank_*.pt files, is_pipe_parallel=False
                Same with case (2).
        """
        if is_pipe_parallel and module_key is not None and mp_world_size != num_ckpt:
            mp_world_size = num_ckpt
            idx = 0

        load_path = self.ckpt_list[idx]

        merge_count = 1
        if num_ckpt == mp_world_size:
            assert os.path.exists(load_path)
            logger.info(f'rank: {mp_rank} loading checkpoint: {load_path}')
            sd = torch.load(load_path, map_location=lambda storage, loc: storage)

            if quantize:
                quantizer = WeightQuantization(mlp_extra_grouping=mlp_extra_grouping,
                                               mp_size=mp_world_size)
                sd_module, all_scales = quantizer.sd_quantize_megatron(self.get_module(sd), quantize_bits, quantize_groups)
                self.set_module(sd, sd_module)
            else:
                all_scales = None
        elif num_ckpt > mp_world_size:
            sd, all_scales, merge_count = self.merge_state_dict(mp_world_size, mp_rank, quantize, \
                quantize_bits, quantize_groups, mlp_extra_grouping)
        else:
            sd, all_scales = self.split_state_dict(mp_world_size, mp_rank, quantize, quantize_bits, \
                quantize_groups, mlp_extra_grouping)
        return load_path, sd, (all_scales, merge_count)

    def get_merge_state_dicts(self, mp_world_size, mp_rank):
        num_ckpt = len(self.ckpt_list)
        assert num_ckpt % mp_world_size == 0, 'Invalid checkpoints and world size for sd merge'

        num_to_merge = num_ckpt // mp_world_size
        ckpt_list = [
            self.ckpt_list[i] for i in range(num_to_merge * mp_rank,
                                             num_to_merge * (mp_rank + 1))
        ]

        logger.info(f"mp_rank: {mp_rank}, ckpt_list: {ckpt_list}")
        sd_list = [
            torch.load(ckpt,
                       map_location=lambda storage,
                       loc: storage) for ckpt in ckpt_list
        ]
        return sd_list

    def get_split_state_dict(self, mp_world_size, mp_rank):
        num_ckpt = len(self.ckpt_list)
        assert mp_world_size % num_ckpt == 0, 'Invalid checkpoints and world size for sd split'

        num_to_split = mp_world_size // num_ckpt
        ckpt_index = mp_rank // num_to_split
        ckpt_offset = mp_rank % num_to_split

        logger.info(
            f"mp_rank: {mp_rank}, ckpt_list: {self.ckpt_list[ckpt_index]}, offset: {ckpt_offset}"
        )

        sd = torch.load(self.ckpt_list[ckpt_index],
                        map_location=lambda storage,
                        loc: storage)

        return sd, num_to_split, ckpt_offset

    def _choose_module_key(self, sd):
        assert not ('module' in sd and 'model' in sd), "checkpoint has both 'model' and 'module' keys, not sure how to proceed"
        assert 'module' in sd or 'model' in sd, "checkpoint contains neither 'model' or 'module' keys, not sure how to proceed"
        if 'module' in sd:
            return 'module'
        elif 'model' in sd:
            return 'model'

    def get_module(self, sd):
        if sd is None:
            return sd
        elif self.module_key == AUTO_MODULE_KEY:
            return sd[self._choose_module_key(sd)]
        else:
            return sd[self.module_key]

    def set_module(self, sd, module):
        if self.module_key is None:
            sd = module
        elif self.module_key == AUTO_MODULE_KEY:
            sd[self._choose_module_key(sd)] = module
        else:
            sd[self.module_key] = module
        return sd

    def check_ckpt_list(self):
        logger.info(f'checkpoint file list: {self.ckpt_list}')
        assert len(self.ckpt_list) > 0

        sd = torch.load(self.ckpt_list[0], map_location=lambda storage, loc: storage)

        # check checkpoint count is same with saved mp_world_size
        if 'mp_world_size' in sd.keys():
            assert len(self.ckpt_list) == sd['mp_world_size'], f"checkpoint count {len(self.ckpt_list)} is different from saved mp_world_size {sd['mp_world_size']}"

    @abstractmethod
    def merge_state_dict(self,
                         mp_world_size,
                         mp_rank,
                         quantize,
                         quantize_bits,
                         groups,
                         mlp_extra_grouping):
        pass

    @abstractmethod
    def split_state_dict(self,
                         mp_world_size,
                         mp_rank,
                         quantize,
                         quantize_bits,
                         groups,
                         mlp_extra_grouping):
        pass

    @abstractmethod
    def sanity_check(self, ckpt_file_name):
        pass


class MegatronSDLoader(SDLoaderBase):
    def __init__(self, ckpt_list, version):
        super().__init__(ckpt_list, version)
        """
        ## Q/K/V data need special processing
        key: transformer.layers.0.attention.query_key_value.weight, shape: torch.Size([3192, 4256])
        key: transformer.layers.0.attention.query_key_value.bias, shape: torch.Size([3192])

        ## merge or split on axis=0
        key: word_embeddings.weight, shape: torch.Size([12672, 4256])
        key: transformer.layers.0.mlp.dense_h_to_4h.bias, shape: torch.Size([4256])
        key: transformer.layers.0.mlp.dense_h_to_4h.weight, shape: torch.Size([4256, 4256])

        ## merge or split on axis=1
        key: transformer.layers.0.attention.dense.weight, shape: torch.Size([4256, 1064])
        key: transformer.layers.0.mlp.dense_4h_to_h.weight, shape: torch.Size([4256, 4256])

        ## no change required
        key: transformer.layers.0.mlp.dense_4h_to_h.bias, shape: torch.Size([4256])
        key: transformer.final_layernorm.weight, shape: torch.Size([4256])
        key: transformer.final_layernorm.bias, shape: torch.Size([4256])
        key: transformer.layers.0.attention.dense.bias, shape: torch.Size([4256])
        key: transformer.layers.0.post_attention_layernorm.weight, shape: torch.Size([4256])
        key: transformer.layers.0.post_attention_layernorm.bias, shape: torch.Size([4256])
        key: transformer.layers.0.input_layernorm.weight, shape: torch.Size([4256])
        key: transformer.layers.0.input_layernorm.bias, shape: torch.Size([4256])
        key: position_embeddings.weight, shape: torch.Size([1024, 4256])
        """

    def merge_query_key_value(self, param_list, ckpt_ver):
        """
        Up to now we found 3 Q/K/V parameter formats in different Megatron checkpoint versions:

        1. version 0, there is no version information saved in checkpoint.
            format: [(3 * np * hn), h]
        2. version 1.0
            format: [(np * hn * 3), h]
        3. version 2.0
            format: [(np * 3 * hn), h]

        h: hidden size
        n: number of attention heads
        p: number of model parallel partitions
        np: n/p
        hn: h/n
        """

        new_qkv = None
        if ckpt_ver == 0:
            # [(3 * np * hn), h]
            assert param_list[0].shape[0] % 3 == 0
            size_qkv = param_list[0].shape[0] // 3
            split_tensors = [torch.split(param, size_qkv, dim=0) for param in param_list]

            tensors = []
            for i in range(3):
                tensor_tuple = [t[i] for t in split_tensors]
                tensors.append(torch.cat(tensor_tuple, axis=0))
            new_qkv = torch.cat(tensors, axis=0)
        elif ckpt_ver == 1.0 or ckpt_ver == 2.0:
            # [(np * hn * 3), h] or [(np * 3 * hn), h]
            new_qkv = torch.cat(param_list, axis=0)
        else:
            assert False, f'checkpoint version: {ckpt_ver} is not supported'

        return new_qkv

    def split_query_key_value(self, param, num_to_split, offset, ckpt_ver):
        """
        Up to now we found 3 Q/K/V parameter formats in different Megatron checkpoint versions:

        1. version 0, there is no version information saved in checkpoint.
            format: [(3 * np * hn), h]
        2. version 1.0
            format: [(np * hn * 3), h]
        3. version 2.0
            format: [(np * 3 * hn), h]

        h: hidden size
        n: number of attention heads
        p: number of model parallel partitions
        np: n/p
        hn: h/n
        """

        new_qkv = None
        if ckpt_ver == 0:
            # [(3 * np * hn), h]
            assert param.shape[0] % 3 == 0
            size_qkv = param.shape[0] // 3
            split_tensors = torch.split(param, size_qkv, dim=0)

            assert split_tensors[0].shape[0] % num_to_split == 0
            split_size = split_tensors[0].shape[0] // num_to_split

            tensors = []
            for i in range(3):
                tensors.append(torch.split(split_tensors[i], split_size, dim=0)[offset])
            new_qkv = torch.cat(tensors, axis=0)
        elif ckpt_ver == 1.0 or ckpt_ver == 2.0:
            # [(np * hn * 3), h] or [(np * 3 * hn), h]
            assert param.shape[0] % num_to_split == 0
            size_qkv = param.shape[0] // num_to_split
            split_tensors = torch.split(param, size_qkv, dim=0)
            new_qkv = split_tensors[offset]
        else:
            assert False, f'checkpoint version: {ckpt_ver} is not supported'

        return new_qkv

    def merge_state_dict(self,
                         mp_world_size,
                         mp_rank,
                         quantize=False,
                         quantize_bits=8,
                         groups=64,
                         mlp_extra_grouping=True):
        self.sanity_check(self.ckpt_list[0])

        sd_list = self.get_merge_state_dicts(mp_world_size, mp_rank)
        ds_sd = copy.deepcopy(sd_list[0])
        new_client_sd = collections.OrderedDict()

        client_sd_list = [self.get_module(sd) for sd in sd_list]
        keys = client_sd_list[0].keys()

        ckpt_ver = self.get_checkpoint_version(ds_sd)
        logger.info(f"checkpoint version: {ckpt_ver}")
        if quantize:
            quantizer = WeightQuantization(mlp_extra_grouping=mlp_extra_grouping,
                                           mp_size=mp_world_size)

        for key in keys:
            value_list = [sd[key] for sd in client_sd_list]

            if "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
                if quantize:
                    value_list = quantizer.Quantize(value_list,
                                                    quantize_bits,
                                                    groups,
                                                    key=key,
                                                    merge_dim=1)
                new_client_sd[key] = torch.cat(value_list, axis=1)
            elif "attention.query_key_value" in key:
                if quantize and "attention.query_key_value.weight" in key:
                    value_list = quantizer.Quantize(value_list,
                                                    quantize_bits,
                                                    groups,
                                                    key=key)
                    new_client_sd[key] = torch.cat(value_list, axis=0)
                else:
                    if quantize:
                        new_client_sd[key] = torch.cat(value_list, axis=0)
                    else:
                        new_client_sd[key] = self.merge_query_key_value(
                            value_list,
                            ckpt_ver)
            elif "mlp.dense_h_to_4h.weight" in key or "word_embeddings.weight" in key or "mlp.dense_h_to_4h.bias" in key:
                if quantize and "mlp.dense_h_to_4h.weight" in key:
                    value_list = quantizer.Quantize(value_list,
                                                    quantize_bits,
                                                    groups,
                                                    key=key)
                new_client_sd[key] = torch.cat(value_list, axis=0)
            else:
                new_client_sd[key] = value_list[0]
        if quantize:
            all_scales = quantizer.merge_scales()
        ds_sd = self.set_module(ds_sd, new_client_sd)

        return ds_sd, (all_scales if quantize else None), len(client_sd_list)

    def split_state_dict(self,
                         mp_world_size,
                         mp_rank,
                         quantize=False,
                         quantize_bits=8,
                         groups=64,
                         mlp_extra_grouping=True):
        self.sanity_check(self.ckpt_list[0])

        sd, num_to_split, ckpt_offset = self.get_split_state_dict(mp_world_size, mp_rank)
        ds_sd = copy.deepcopy(sd)
        new_client_sd = collections.OrderedDict()

        client_sd = self.get_module(sd)

        ckpt_ver = self.get_checkpoint_version(ds_sd)
        logger.info(f"checkpoint version: {ckpt_ver}")

        if quantize:
            quantizer = WeightQuantization(mlp_extra_grouping=mlp_extra_grouping,
                                           mp_size=mp_world_size)

        for key in client_sd.keys():
            value = client_sd[key]

            if "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
                assert value.shape[1] % num_to_split == 0
                split_size = value.shape[1] // num_to_split
                if quantize:
                    q_vals = quantizer.Quantize([value], quantize_bits, groups, key)
                    value = q_vals[0]
                new_client_sd[key] = torch.split(value, split_size, dim=1)[ckpt_offset]
            elif "attention.query_key_value" in key:
                if quantize and "attention.query_key_value.weight" in key:
                    q_vals = quantizer.Quantize([value], quantize_bits, groups, key)
                    value = q_vals[0]
                new_client_sd[key] = self.split_query_key_value(
                    value,
                    num_to_split,
                    ckpt_offset,
                    ckpt_ver)
            elif "mlp.dense_h_to_4h.weight" in key or "word_embeddings.weight" in key or "mlp.dense_h_to_4h.bias" in key:
                assert value.shape[0] % num_to_split == 0
                split_size = value.shape[0] // num_to_split
                if quantize and "mlp.dense_h_to_4h.weight" in key:
                    q_vals = quantizer.Quantize([value], quantize_bits, groups, key)
                    value = q_vals[0]
                new_client_sd[key] = torch.split(value, split_size, dim=0)[ckpt_offset]
            else:
                new_client_sd[key] = value

        if quantize:
            all_scales = quantizer.merge_scales_split(num_to_split)

        ds_sd = self.set_module(ds_sd, new_client_sd)

        return ds_sd, (all_scales if quantize else None)

    def sanity_check(self, ckpt_file_name):
        keys_to_check = [
            "attention.dense.weight",
            "mlp.dense_4h_to_h.weight",
            "attention.query_key_value",
            "mlp.dense_h_to_4h.weight",
            "mlp.dense_h_to_4h.bias"
        ]

        sd = torch.load(ckpt_file_name, map_location=lambda storage, loc: storage)

        # partail_key is a sub-string of one key in the sd
        def check_key_exist(partial_key, sd):
            keys = sd.keys()
            found = False
            for k in keys:
                if partial_key in k:
                    found = True
                    break
            return found

        for key in keys_to_check:
            assert check_key_exist(key, self.get_module(sd)), f'key: {key} is not found in the checkpoint {ckpt_file_name}'

    def get_checkpoint_version(self, state_dict):
        # Use 0 if version info doesn't exist
        return self.version if self.version is not None else state_dict.get(
            'checkpoint_version',
            0)
