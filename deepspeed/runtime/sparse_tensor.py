# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Implementation of a compressed sparse tensor. Similar in
functionality to TensorFlow's IndexedSlices implementation.
"""

import torch


class SparseTensor(object):
    """ Compressed Sparse Tensor """

    def __init__(self, dense_tensor=None):
        self.orig_dense_tensor = dense_tensor
        self.dtype = self.orig_dense_tensor.dtype
        self.is_sparse = dense_tensor.is_sparse
        if dense_tensor is not None:
            if dense_tensor.is_sparse:
                dense_tensor = dense_tensor.coalesce()
                self.indices = dense_tensor.indices().flatten()
                self.values = dense_tensor.values()
            else:
                result = torch.sum(dense_tensor, dim=1)
                self.indices = result.nonzero().flatten()
                self.values = dense_tensor[self.indices]
            self.dense_size = list(dense_tensor.size())
        else:
            self.indices = None
            self.values = None
            self.dense_size = None

    def to_coo_tensor(self):
        return torch.sparse_coo_tensor(self.indices.unsqueeze(0), self.values, self.dense_size)

    @staticmethod
    def type():
        return "deepspeed.SparseTensor"

    def to_dense(self):
        it = self.indices.unsqueeze(1)
        full_indices = torch.cat([it for _ in range(self.dense_size[1])], dim=1)
        return self.values.new_zeros(self.dense_size).scatter_add_(0, full_indices, self.values)

    def sparse_size(self):
        index_size = list(self.indices.size())
        index_size = index_size[0]
        value_size = list(self.values.size())
        value_size = value_size[0] * value_size[1]
        dense_size = self.dense_size[0] * self.dense_size[1]
        return index_size + value_size, dense_size

    def add(self, b):
        assert self.dense_size == b.dense_size
        self.indices = torch.cat([self.indices, b.indices])
        self.values = torch.cat([self.values, b.values])

    def __str__(self):
        sparse_size, dense_size = self.sparse_size()
        return "DeepSpeed.SparseTensor(indices_size={}, values_size={}, " \
               "dense_size={}, device={}, reduction_factor={})".format(
            self.indices.size(), self.values.size(), self.dense_size,
            self.indices.get_device(), dense_size / sparse_size
        )

    def __repr__(self):
        return self.__str__()
