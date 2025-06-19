# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Tuple, Optional
from dataclasses import dataclass

import torch


@dataclass
class TensorMetadata:
    """Metadata for a tensor to be stored in CPU memory"""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    stride: Tuple[int, ...]
    storage_offset: int
    requires_grad: bool
    layout: torch.layout
    memory_format: torch.memory_format = torch.contiguous_format
    real_data: Optional[torch.Tensor] = None  # Store actual tensor data when configured


class InputStorage:
    """Storage class to keep real inputs in CPU memory with tensor metadata"""

    def __init__(self, keep_int_input_tensors: bool = False, keep_all_input_tensors: bool = False):
        self._stored_inputs: Any = None
        self._has_data: bool = False
        self._keep_int_input_tensors: bool = keep_int_input_tensors
        self._keep_all_input_tensors: bool = keep_all_input_tensors

    def _is_int_tensor(self, tensor: torch.Tensor) -> bool:
        """Check if tensor has integer dtype"""
        return tensor.dtype in [
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64,
            torch.bool
        ]

    def _extract_tensor_metadata(self, tensor: torch.Tensor) -> TensorMetadata:
        """Extract metadata from a tensor"""
        # Get memory format safely
        try:
            memory_format = tensor.memory_format() if hasattr(tensor, 'memory_format') else torch.contiguous_format
        except:
            memory_format = torch.contiguous_format

        # Store real data for tensors if configured to do so
        real_data = None
        if self._keep_all_input_tensors or (self._keep_int_input_tensors and self._is_int_tensor(tensor)):
            # Move to CPU to save GPU memory
            real_data = tensor.detach().cpu()

        return TensorMetadata(shape=tuple(tensor.shape),
                              dtype=tensor.dtype,
                              device=tensor.device,
                              stride=tuple(tensor.stride()),
                              storage_offset=tensor.storage_offset(),
                              requires_grad=tensor.requires_grad,
                              layout=tensor.layout,
                              memory_format=memory_format,
                              real_data=real_data)

    def _store_value(self, value: Any) -> Any:
        """
        Recursively store a value, converting tensors to metadata and keeping non-tensors as-is
        """
        if isinstance(value, torch.Tensor):
            return self._extract_tensor_metadata(value)
        elif isinstance(value, (list, tuple)):
            stored_items = [self._store_value(item) for item in value]
            return type(value)(stored_items) if isinstance(value, tuple) else stored_items
        elif isinstance(value, dict):
            return {k: self._store_value(v) for k, v in value.items()}
        else:
            # For non-tensor values (int, float, str, bool, etc.), store as-is
            return value

    def _materialize_value(self, stored_value: Any) -> Any:
        """
        Recursively materialize a stored value, creating tensors from metadata and keeping non-tensors as-is
        """
        if isinstance(stored_value, TensorMetadata):
            # If we have real data stored, use it
            if stored_value.real_data is not None:
                try:
                    # Use the stored real data
                    tensor = stored_value.real_data.clone()

                    # Set stride if different from default and tensor is contiguous
                    if tensor.stride() != stored_value.stride and len(stored_value.shape) > 0:
                        try:
                            # Create tensor with specific stride
                            tensor = torch.as_strided(tensor, stored_value.shape, stored_value.stride,
                                                      stored_value.storage_offset)
                        except RuntimeError:
                            # If stride setting fails, use default stride
                            pass

                    # Move to target device and set requires_grad
                    tensor = tensor.to(device=stored_value.device)
                    tensor.requires_grad_(stored_value.requires_grad)

                    return tensor

                except Exception as e:
                    # Fallback to dummy data if real data fails
                    pass

            # Create a tensor with the stored metadata (original behavior for non-int tensors)
            # Use CPU first to avoid GPU memory issues, then move to target device
            try:
                tensor = torch.empty(stored_value.shape,
                                     dtype=stored_value.dtype,
                                     layout=stored_value.layout,
                                     device='cpu')

                # Fill with dummy data (ones) for profiling purposes
                tensor.fill_(1.0)

                # Set stride if different from default and tensor is contiguous
                if tensor.stride() != stored_value.stride and len(stored_value.shape) > 0:
                    try:
                        # Create tensor with specific stride
                        tensor = torch.as_strided(tensor, stored_value.shape, stored_value.stride,
                                                  stored_value.storage_offset)
                    except RuntimeError:
                        # If stride setting fails, use default stride
                        pass

                # Move to target device and set requires_grad
                tensor = tensor.to(device=stored_value.device)
                tensor.requires_grad_(stored_value.requires_grad)

                return tensor

            except Exception as e:
                # Fallback: create a simple tensor if anything fails
                tensor = torch.ones(stored_value.shape, dtype=stored_value.dtype, device=stored_value.device)
                tensor.requires_grad_(stored_value.requires_grad)
                return tensor

        elif isinstance(stored_value, (list, tuple)):
            materialized_items = [self._materialize_value(item) for item in stored_value]
            return type(stored_value)(materialized_items) if isinstance(stored_value, tuple) else materialized_items
        elif isinstance(stored_value, dict):
            return {k: self._materialize_value(v) for k, v in stored_value.items()}
        else:
            # Non-tensor values are returned as-is
            return stored_value

    def put(self, real_inputs: Any) -> None:
        """
        Store real inputs

        Args:
            real_inputs: The real inputs to store (can be tensors, lists, tuples, etc.)
        """
        stored_inputs = self._store_value(real_inputs)
        self._stored_inputs = stored_inputs
        self._has_data = True

    def get(self) -> Any:
        """
        Retrieve and materialize stored real inputs

        Returns:
            Materialized real inputs with actual tensors

        Raises:
            RuntimeError: If no inputs are stored
        """
        if not self._has_data:
            raise RuntimeError("No inputs stored in InputStorage")

        return self._materialize_value(self._stored_inputs)

    def has_data(self) -> bool:
        """
        Check if storage contains inputs

        Returns:
            True if inputs are stored, False otherwise
        """
        return self._has_data

    def clear(self) -> None:
        """Clear stored inputs"""
        self._stored_inputs = None
        self._has_data = False
