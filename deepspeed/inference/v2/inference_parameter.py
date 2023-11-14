# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Dict

import torch

CORE_PARAM = "_ds_core_param_key"

STR_TO_DTYPE = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.int16": torch.int16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


class InferenceParameter(torch.Tensor):
    """
    An extension of the torch.Tensor class to support our inference focused features. One important
    thing to note here is that an InferenceParam can be used a torch.Tensor, but outputs of
    torch.Tensor operations will not be InferenceParams.
    """

    @staticmethod
    def __new__(cls, tensor, *args, **kwargs):
        new_tensor = super().__new__(cls, tensor, *args, **kwargs)
        if hasattr(tensor, "_aux_attrs"):
            setattr(new_tensor, "_aux_attrs", tensor.aux_attrs)
        return new_tensor

    def to(self, *args, **kwargs):
        new_tensor = super().to(*args, **kwargs)
        if hasattr(self, "_aux_attrs"):
            setattr(new_tensor, "_aux_attrs", self.aux_attrs)
        try:
            _ = torch.device(args[0])
            for name, attr in new_tensor.aux_attrs.items():
                new_attr = attr.to(*args, **kwargs)
                setattr(new_tensor, name, new_attr)
                new_tensor.aux_attrs[name] = new_attr
        except:
            pass

        return new_tensor

    @classmethod
    def initialize(cls, core_param: torch.Tensor, **kwargs) -> 'InferenceParameter':
        """
        Create the inference parameter.
        """
        param = InferenceParameter(core_param)
        setattr(param, "_aux_attrs", kwargs)

        for attr_name, attr in kwargs.items():
            if hasattr(param, attr_name):
                raise ValueError(f"Attribute {attr_name} already exists on param.")

            if not isinstance(attr, torch.Tensor):
                raise ValueError(f"Attribute {attr_name} must be a tensor.")

            setattr(param, attr_name, attr)

        return param

    @classmethod
    def initialize_raw(self, **kwargs) -> 'InferenceParameter':
        """
        All kwargs must be torch.Tensors and must include the core parameter.
        """
        if CORE_PARAM not in kwargs:
            raise ValueError(f"Must provide core parameter, with key {CORE_PARAM}.")

        return InferenceParameter.initialize(kwargs[CORE_PARAM], **kwargs)

    @property
    def aux_attrs(self) -> Dict[str, torch.Tensor]:
        """
        Dictionary of auxiliary attributes.
        """
        return self._aux_attrs
