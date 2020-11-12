'''
Copyright 2020 The Microsoft DeepSpeed Team

Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex, commit a109f85
'''
import torch


class MultiTensorApply(object):
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        return op(self.chunk_size, noop_flag_buffer, tensor_lists, *args)
