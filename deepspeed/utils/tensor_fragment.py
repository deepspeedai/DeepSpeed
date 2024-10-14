# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from dataclasses import dataclass
from deepspeed import comm as dist
from typing import Dict


@dataclass
class fragment_address:
    numel: int
    start: int


@dataclass
class tensor_fragment:
    lp_fragment: torch.Tensor
    lp_fragment_address: fragment_address
    hp_fragment: torch.Tensor
    hp_fragment_address: fragment_address
    gradient_dict: Dict
    offload_gradient_dict: Dict
    use_offload: bool
    param_group_index: int
    optim_fragment: Dict = None

    def update_hp(self):
        self.hp_fragment.data.copy_(self.lp_fragment.data)

    def update_lp(self):
        self.lp_fragment.data.copy_(self.hp_fragment.data)

    def get_optim_state_fragment(self, key):
        if key in self.optim_fragment:
            return self.optim_fragment[key]
        else:
            raise ValueError(f'{key} not found in optimizer state fragment')

    def set_optim_state_fragment(self, flat_hp_partition, optim_fragment):
        self.optim_fragment = {
            key: value.narrow(0, self.hp_fragment_address.start, self.hp_fragment_address.numel)
            for key, value in optim_fragment.items()
            if torch.is_tensor(value) and value.shape == flat_hp_partition.shape
        }

    def get_hp_fragment_address(self):
        return self.hp_fragment_address

    def get_optim_state_keys(self):
        return list(self.optim_fragment.keys())

    def get_hp_fragment(self, optim_state_key=None):
        if optim_state_key is None:
            return self.hp_fragment
        return self.get_optim_state_fragment(optim_state_key)

    def get_lp_grad_fragment(self, index_in_param_group):
        if self.use_offload:
            gradient_dict = self.offload_gradient_dict
        else:
            gradient_dict = self.gradient_dict

        if self.param_group_index not in gradient_dict or gradient_dict[self.param_group_index] is None:
            raise ValueError("Gradients are only available immediately after backward and before engine step")

        return gradient_dict[self.param_group_index][index_in_param_group]


def map_to_flat_opt_states(flat_hp_tensor, lp_tensors, optim_state, opt_keys):
    for key in opt_keys:
        hp_param = flat_hp_tensor
        buffer = torch.zeros_like(hp_param)

        for lp in lp_tensors:
            if lp._hp_mapping is not None:
                hp_fragment_address = lp._hp_mapping.get_hp_fragment_address()
                hp_fragment = buffer.narrow(0, hp_fragment_address.start, hp_fragment_address.numel)
                hp_fragment.data.copy_(lp._hp_mapping.get_hp_fragment(optim_state_key=key).data)
                lp._hp_mapping.hp_fragment = hp_fragment

        optim_state[hp_param][key] = buffer


def get_full_hp_param(self, optim_state_key=None):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)
        hp_fragment = self._hp_mapping.get_hp_fragment(optim_state_key)
        reduce_fragment.data.copy_(hp_fragment.data)
    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def set_full_hp_param(self, value, optim_state_key=None):
    if self._hp_mapping is not None:
        lp_frag_address = self._hp_mapping.lp_fragment_address
        value_fragment = torch.narrow(value.flatten(), 0, lp_frag_address.start, lp_frag_address.numel)
        hp_fragment = self._hp_mapping.get_hp_fragment(optim_state_key)
        hp_fragment.data.copy_(value_fragment.data)


def get_full_hp_grad(self):
    reduce_buffer = torch.zeros_like(self, dtype=torch.float32).flatten()
    if self._hp_mapping is not None:
        lp_grad_fragment = self._hp_mapping.get_lp_grad_fragment(self._index_in_param_group)
        hp_grad_fragment = lp_grad_fragment.to(torch.float32).flatten()

        lp_frag_address = self._hp_mapping.lp_fragment_address
        reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)

        if self.view(-1).shape == hp_grad_fragment.shape:
            reduce_buffer.data.copy_(hp_grad_fragment.data)
        else:
            reduce_fragment.data.copy_(hp_grad_fragment.data)

    dist.all_reduce(reduce_buffer, group=self._dp_group)
    return reduce_buffer.reshape_as(self)


def set_full_hp_grad(self, value):
    if self._hp_mapping is not None:
        lp_grad_fragment = self._hp_mapping.get_lp_grad_fragment(self._index_in_param_group)
        lp_frag_address = self._hp_mapping.lp_fragment_address
        value_fragment = torch.narrow(value.flatten(), 0, lp_frag_address.start, lp_frag_address.numel)
        lp_grad_fragment.data.copy_(value_fragment.data.reshape_as(lp_grad_fragment.data))


def safe_get_full_fp32_param(param):
    """Assemble and return the fp32 parameter of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
    """
    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        return param._z3_optimizer.get_full_hp_param(param)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        return param.get_full_hp_param()
    return None


def safe_set_full_fp32_param(param, value):
    """Update the partitioned fp32 parameter of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
            value (``torch.Tensor``): New value
    """
    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        param._z3_optimizer.set_full_hp_param(value, param)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        param.set_full_hp_param(value)


def safe_get_full_optimizer_state(param, optim_state_key):
    """Assemble and return the fp32 optimizer state of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
            optim_state_key (``string``): Key value of optimizer state (e.g., `exp_avg` in Adam optimizer)
    """
    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        return param._z3_optimizer.get_full_hp_param(param, optim_state_key)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        return param.get_full_hp_param(optim_state_key)
    return None


def safe_set_full_optimizer_state(param, value, optim_state_key):
    """Update the partitioned fp32 optimizer state of a low-precision (e.g., fp16) parameter.

        Args:
            param (``torch.nn.Parameter``): A model parameter
            value (``torch.Tensor``): New value
            optim_state_key (``string``): Key value of optimizer state (e.g., `exp_avg` in Adam optimizer)
    """
    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        param._z3_optimizer.set_full_hp_param(value, param, optim_state_key)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        param.set_full_hp_param(value, optim_state_key)


# TODO: Figure out the correct return dtype
def safe_get_full_grad(param):
    """
        Assemble and return the fp32 gradient of a low-precision (e.g., fp16) parameter.
        The return data type is that used for gradient accumulation. This is usually the param data type,
        but could also be different (e.g., bf16 param training with fp32 gradient accumulation).

        Args:
            param (``torch.nn.Parameter``): A model parameter
    """
    if param.grad is not None:
        return param.grad

    # ZeRO stage 3 param
    if hasattr(param, 'ds_id'):
        return param._z3_optimizer.get_fp32_grad_for_param(param)

    # ZeRO stage 1, 2, and bf16_optimizer params
    if hasattr(param, '_hp_mapping'):
        return param.get_full_hp_grad()

    return None


def safe_set_full_grad(param, value):
    """
        Update the partitioned gradient of a low-precision (e.g., fp16) parameter.
        To avoid precision issues, the update value should have the data type of
        gradient accumulation.

        Args:
            param (``torch.nn.Parameter``): A model parameter
            value (``torch.Tensor``): The un-partitioned new gradient value.
    """
    if param.grad is not None:
        param.grad.copy_(value)
    elif hasattr(param, 'ds_id'):
        # ZeRO stage 3 param
        param._z3_optimizer.set_fp32_grad_for_param(value, param)
    elif hasattr(param, '_hp_mapping'):
        # ZeRO stage 1, 2, and bf16_optimizer params
        param.set_full_hp_grad(value)


### Local API  START ###
def safe_get_local_grad(param):
    """
        Get the local gradient partition of a ZeRO-3 partitioned parameter.
        The return data type is that used for gradient accumulation. This is usually the param data type,
        but could also be different (e.g., bf16 param training with fp32 gradient accumulation).
        Args:
            param (``torch.nn.Parameter``): A model parameter
    """
    assert hasattr(param, 'ds_id'), f'This API is only defined for ZeRO-3 partitioned parameters'
    return param._z3_optimizer.get_local_fp32_grad_for_param(param)


def safe_set_local_grad(param, value):
    """
        Update the local gradient partition of a ZeRO-3 partitioned parameter.
        To avoid precision issues, the update value should have the data type of
        gradient accumulation.

        Args:
            param (``torch.nn.Parameter``): A model parameter.
            value (``torch.Tensor``): New value of local gradient partition.
    """
    assert hasattr(param, 'ds_id'), f'This API is only defined for ZeRO-3 partitioned parameters'
    param._z3_optimizer.set_local_grad_for_param(value, param)


def safe_get_local_fp32_param(param):
    """Get the local partition of a ZeRO-3 partitioned parameter in fp32 precision.
        Args:
            param (``torch.nn.Parameter``): A model parameter.
    """
    assert hasattr(param, 'ds_id'), f'This API is only defined for ZeRO-3 partitioned parameters'
    return param._z3_optimizer.get_local_fp32_param(param)


def safe_get_local_optimizer_state(param, optim_state_key):
    """Get the local optimizer state partition of ZeRO-3 partitioned parameter in fp32 precision.
        Args:
            param (``torch.nn.Parameter``): A model parameter
            optim_state_key (``string``): Key value of optimizer state (e.g., `exp_avg` in Adam optimizer)
    """
    assert hasattr(param, 'ds_id'), f'This API is only defined for ZeRO-3 partitioned parameters'
    return param._z3_optimizer.get_local_fp32_param(param, optim_state_key)


def safe_set_local_optimizer_state(param, value, optim_state_key):
    """Update the local optimizer state partition of a ZeRO-3 partitioned parameter.
        Args:
            param (``torch.nn.Parameter``): A model parameter.
            value (``torch.Tensor``): New value of local optimizer state partition.
            optim_state_key (``string``): Key value of optimizer state (e.g., `exp_avg` in Adam optimizer).
    """
    assert hasattr(param, 'ds_id'), f'This API is only defined for ZeRO-3 partitioned parameters'
    param._z3_optimizer.set_local_hp_param(value, param, optim_state_key)


def safe_set_local_fp32_param(param, value):
    """Update the local partition of ZeRO-3 partitioned parameter.
        Args:
            param (``torch.nn.Parameter``): A model parameter.
            value (``torch.Tensor``): New value of local parameter partition.
    """
    assert hasattr(param, 'ds_id'), f'This API is only defined for ZeRO-3 partitioned parameters'
    param._z3_optimizer.set_local_hp_param(value, param)


### Local API  END ###


def get_hp_fragment_mapping(lp_param, lp_start, flat_hp_partition, gradient_dict, offload_gradient_dict, use_offload,
                            param_group_index, partition_start, partition_size):
    lp_end = lp_param.numel() + lp_start
    hp_start = partition_start
    hp_end = partition_start + partition_size

    fragment_start = max(lp_start, hp_start)
    fragment_end = min(lp_end, hp_end)
    assert fragment_start < fragment_end, \
        f'fragment start {fragment_start} should be < fragment_end {fragment_end}'

    fragment_numel = fragment_end - fragment_start
    hp_frag_address = fragment_address(start=fragment_start - hp_start, numel=fragment_numel)
    hp_fragment_tensor = flat_hp_partition.narrow(0, hp_frag_address.start, hp_frag_address.numel)

    lp_frag_address = fragment_address(start=fragment_start - lp_start, numel=fragment_numel)
    lp_fragment_tensor = lp_param.flatten().narrow(0, lp_frag_address.start, lp_frag_address.numel)

    return tensor_fragment(lp_fragment=lp_fragment_tensor,
                           lp_fragment_address=lp_frag_address,
                           hp_fragment=hp_fragment_tensor,
                           hp_fragment_address=hp_frag_address,
                           gradient_dict=gradient_dict,
                           offload_gradient_dict=offload_gradient_dict,
                           use_offload=use_offload,
                           param_group_index=param_group_index)


'''
Logic for lp_param to hp_param mapping

lp      lp0 lp1 lp2         lp3  lp4            <-------  indices/names
lp      [  ][  ][          ][   ][         ]    <-------- tensors
flat_lp [                                  ]     <-------- flat lp params
flat_hp            [                 ]   <------------------ flat hp partition on current rank
full_hp [                                        ] <------- full flat hp params


lp2
 full numel = 16
 lp_frag
   numel = 12
   frag_start = 3
   frag_end  = 15
 hp_frag
    numel = 12
    frag_start = 0
    frag_end = 11

 hp_frag.copy_(lp_frag)


lp3:
  full numel = 4
  lp_frag
     numel = 4
     start = 0
     end = 3
  hp_frag
     numel = 4
     start = 12
     end = 15


lp4:
   full numel = 12
   lp_frag
     numel = 4
     start = 0
     end = 3
  hp_frag
     numel = 4
     start = 16
     end = 19



Visual depiction of above
lp              {         }
flat_lp [                                ]
flat_hp            (                 )


flat_lp [       {  (      }          )   ]
                lx  hx   ly          hy
                    ly-hx


lp                             {       }
flat_lp [                                ]
flat_hp            (                 )


flat_lp [          (            {     ) }  ]
                   hx           lx   hy ly
                                   hy-lx

lp                        {   }
flat_lp [                                ]
flat_hp            (                 )


flat_lp [          (       {   }      )   ]
                   hx      lx  ly    hy
                             ly-lx

lp -> (lx, hy)
flat_hp -> (hx, hy)
'''
