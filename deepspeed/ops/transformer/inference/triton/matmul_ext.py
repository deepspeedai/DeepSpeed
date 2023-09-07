# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import triton
import os
from filelock import FileLock
import deepspeed.ops.transformer.inference.triton.triton_matmul_kernel as triton_matmul_kernel
import pickle
from io import open
import deepspeed
from pathlib import Path
import atexit


# -----------------------------------------------------------------------------
# util class/functions for triton
def _default_cache_dir():
    return os.path.join(Path.home(), ".triton", "autotune")


def bias_add_activation(C, bias=None, activation=""):
    if bias is not None:
        C += bias
    # activation
    if activation == "relu":
        relu = torch.nn.Relu()
        C = relu(C)
    elif activation == "leaky_relu":
        leaky_relu = torch.nn.LeakyReLU(0.01)
        C = leaky_relu(C)
    elif activation == "gelu":
        sigmoid = torch.nn.Sigmoid()
        C = sigmoid(1.702 * C) * C
    elif activation == "sigmoid":
        sigmoid = torch.nn.Sigmoid()
        C = sigmoid(C)
    return C


class AutotuneCacheManager:
    """
        Cache manager for autotune
    """

    def __init__(self, key):
        self.key = key
        self.file_path = None
        self.lock_path = None
        # if caching is enabled, get the lock and bin path
        self.cache_dir = os.environ.get('TRITON_CACHE_DIR', _default_cache_dir())
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.cache_dir:
            self.file_path = os.path.join(self.cache_dir, self.key + ".pickle")
            self.lock_path = self.file_path + ".lock"

    def has_file(self):
        return self.file_path and os.path.exists(self.file_path)

    def put(self, table):
        if self.file_path:
            assert self.lock_path is not None
            with FileLock(self.lock_path):
                with open(self.file_path + ".tmp", 'wb') as handle:
                    pickle.dump(table, handle)
                os.rename(self.file_path + ".tmp", self.file_path)

    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as handle:
                loaded_dict = pickle.load(handle)
            return loaded_dict
        else:
            return None


# -----------------------------------------------------------------------------
# triton matmul class


class MatmulExt(torch.autograd.Function):
    """
        a wrapper class that can call different triton matmul kernels depending on the input parameters
    """

    @staticmethod
    def forward(A, B, bias=None, activation="", use_triton=True, update_autotune_table=False):
        """
            A: input, activation matrix A
            B: input, weight matrix B
        """
        matmul = None
        quantize_activation = False
        Batch = 0

        if len(A.shape) == 3:  # if A is 3d-tensor where batch index is given as 0-axis
            assert A.is_contiguous(), "matrix A must be contiguous"
            Batch, M, K = A.shape
            A = A.view(-1, K)

        # fp16 activation and fp16 weight matmul into fp16 output
        matmul = fp16_matmul
        C = matmul.forward(A, B, use_triton=use_triton, bias=bias, activation=activation)

        if matmul and update_autotune_table:
            matmul._update_autotune_table()

        if Batch > 0:
            C = C.view(Batch, M, -1)

        return C


class TritonMatmul(torch.autograd.Function):
    """
        triton matmul kernel superclass
    """

    def __init__(self):
        pass

    @staticmethod
    def _ref_forward(A, B, ref_dtype=torch.float32):
        C = torch.matmul(A.type(ref_dtype), B.type(ref_dtype))
        return C

    @staticmethod
    def _read_autotune_table(cache_key, triton_kernel):
        cache_manager = AutotuneCacheManager(cache_key)
        table = cache_manager.load()
        if table:
            triton_kernel.cache = table

    @staticmethod
    def _write_autotune_table(cache_key, triton_kernel):
        cache_manager = AutotuneCacheManager(cache_key)
        cache_manager.put(triton_kernel.cache)

    @staticmethod
    def _update_autotune_table(cache_key, triton_kernel):
        cache_manager = AutotuneCacheManager(cache_key)
        autotune_table = cache_manager.load()
        if autotune_table is None:
            autotune_table = dict()
        autotune_table.update(triton_kernel.cache)  # always overwrite with the new autotune results
        cache_manager = AutotuneCacheManager(cache_key)
        cache_manager.put(autotune_table)

    @staticmethod
    def forward(
            A,
            B,
            ref_dtype=torch.float32,  # fp32 only
            bias=None,
            activation=""):
        C = torch.matmul(A.type(ref_dtype), B.type(ref_dtype))
        C = bias_add_activation(C, bias, activation)
        return C


class Fp16Matmul(TritonMatmul):
    """
        fp16 matrix multiplication kernel
        dtypes: fp16 x fp16 = fp16
    """

    _2d_kernel = triton_matmul_kernel._fp_matmul
    _4d_kernel = triton_matmul_kernel.matmul_4d_kernel
    _cache_stride = 32

    def __init__(self, read_cache=True):
        super().__init__()
        if read_cache:
            __class__._read_autotune_table()

    def skip_autotune(self):
        __class__._2d_kernel.configs = [__class__._2d_kernel.configs[0]]
        __class__._4d_kernel.configs = [__class__._4d_kernel.configs[0]]

    @staticmethod
    def forward(A, B, use_triton=True, bias=None, activation=""):
        if use_triton:
            device = A.device
            # handle non-contiguous inputs if necessary
            if A.stride(0) > 1 and A.stride(1) > 1:
                A = A.contiguous()
            if B.stride(0) > 1 and B.stride(1) > 1:
                B = B.contiguous()
            # checks constraints
            assert A.shape[1] == B.shape[0], "incompatible dimensions"
            M, K = A.shape
            _, N = B.shape
            # allocates output
            C = torch.empty((M, N), device=device, dtype=A.dtype)
            # accumulator types
            ACC_TYPE = triton.language.float32 if A.dtype in [torch.float16, torch.bfloat16, torch.float32
                                                              ] else triton.language.int32
            # launch kernel
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
            __class__._2d_kernel[grid](A,
                                       B,
                                       C,
                                       M,
                                       N,
                                       K,
                                       bias,
                                       A.stride(0),
                                       A.stride(1),
                                       B.stride(0),
                                       B.stride(1),
                                       C.stride(0),
                                       C.stride(1),
                                       M // __class__._cache_stride,
                                       N // __class__._cache_stride,
                                       K // __class__._cache_stride,
                                       GROUP_M=8,
                                       ACC_TYPE=ACC_TYPE,
                                       BIAS_ADD=(0 if bias is None else 1),
                                       ACTIVATION=activation)
        else:
            C = torch.matmul(A, B)
        return C

    @staticmethod
    def _matmul_4d(a, b):
        assert a.shape[-1] == b.shape[-2], "incompatible dimensions"
        assert a.is_contiguous(), "matrix A must be contiguous"
        assert b.is_contiguous(), "matrix B must be contiguous"

        B, H, M, K = a.shape
        B, H, K, N = b.shape

        assert K > 1, "inner-product dimension K should be larger than 1"

        c = torch.empty((B, H, M, N), device=a.device, dtype=a.dtype)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            H,
            B,
        )

        __class__._4d_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            M // __class__._cache_stride,
            N // __class__._cache_stride,
            K // __class__._cache_stride,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            a.stride(3),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            b.stride(3),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            c.stride(3),
            scale=-1.0,
            MASK=False,
        )
        return c

    @staticmethod
    def _score_4d_matmul(input, head_size, input_mask, scale=-1.0):
        assert input.is_contiguous(), "matrix input must be contiguous"

        batches = input.shape[0]
        d_model = input.shape[-1] // 3
        num_of_heads = d_model // head_size

        q = input[:, :, :d_model]
        k = input[:, :, d_model:d_model * 2]

        q = q.view(batches, -1, num_of_heads, head_size)
        k = k.view(batches, -1, num_of_heads, head_size)

        # checks constraints
        assert q.shape == k.shape, "incompatible dimensions"
        B, M, H, K = q.shape
        B, N, H, K = k.shape

        assert K > 1, "inner-product dimension K should be larger than 1"

        # allocates output
        output = torch.empty((B, H, M, N), device=q.device, dtype=q.dtype)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            H,
            B,
        )
        __class__._4d_kernel[grid](
            q,
            k,
            output,
            M,
            N,
            K,
            M // __class__._cache_stride,
            N // __class__._cache_stride,
            K // __class__._cache_stride,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(3),
            k.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            scale=scale,
            MASK=False,
        )
        return output

    @staticmethod
    def _context_4d_matmul(prob, input, head_size):
        assert prob.is_contiguous(), "matrix prob must be contiguous"
        assert input.is_contiguous(), "matrix input must be contiguous"

        batches = input.shape[0]
        d_model = input.shape[-1] // 3
        num_of_heads = d_model // head_size

        v = input[:, :, d_model * 2:]

        v = v.view(batches, -1, num_of_heads, head_size)

        # checks constraints
        assert (prob.shape[0] == v.shape[0] and prob.shape[1] == v.shape[2] and prob.shape[2] == v.shape[1]
                and prob.shape[3] == v.shape[1]), "incompatible dimensions"
        B, H, M, K = prob.shape
        B, K, H, N = v.shape

        assert K > 1, "inner-product dimension K should be larger than 1"

        # allocates output
        output = torch.empty((B, M, H, N), device=v.device, dtype=v.dtype)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            H,
            B,
        )

        __class__._4d_kernel[grid](
            prob,
            v,
            output,
            M,
            N,
            K,
            M // __class__._cache_stride,
            N // __class__._cache_stride,
            K // __class__._cache_stride,
            prob.stride(0),
            prob.stride(1),
            prob.stride(2),
            prob.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            # Here we also transpose the output when writing to memory.
            output.stride(0),
            output.stride(2),
            output.stride(1),
            output.stride(3),
            scale=-1,
            MASK=False,
        )
        return output.view(batches, -1, d_model)

    @staticmethod
    def _ref_forward(A, B, ref_dtype=torch.float32, bias=None, activation=""):
        C = torch.matmul(A.type(ref_dtype), B.type(ref_dtype))
        C = bias_add_activation(C, bias, activation)
        return C

    @staticmethod
    def _check_parity(A,
                      B,
                      output_dtype,
                      SA=None,
                      SB=None,
                      qblock_size=None,
                      ref_dtype=torch.float32,
                      tol=0.01,
                      use_triton=True,
                      bias=None,
                      activation=""):
        torch_output = __class__._ref_forward(A, B, ref_dtype=ref_dtype, bias=bias, activation=activation)
        triton_output = __class__.forward(A, B, use_triton=use_triton, bias=bias, activation=activation)
        assert torch.allclose(triton_output.cpu().type(torch_output.dtype), torch_output.cpu(), rtol=tol)
        print(f"{__class__.__name__}: PASSed the parity check")
        return triton_output, torch_output

    @staticmethod
    def _read_autotune_table():
        TritonMatmul._read_autotune_table(__class__.__name__ + "_2d_kernel", __class__._2d_kernel)
        TritonMatmul._read_autotune_table(__class__.__name__ + "_4d_kernel", __class__._4d_kernel)

    @staticmethod
    def _write_autotune_table():
        TritonMatmul._write_autotune_table(__class__.__name__ + "_2d_kernel", __class__._2d_kernel)
        TritonMatmul._write_autotune_table(__class__.__name__ + "_4d_kernel", __class__._4d_kernel)

    @staticmethod
    def _update_autotune_table():
        TritonMatmul._update_autotune_table(__class__.__name__ + "_2d_kernel", __class__._2d_kernel)
        TritonMatmul._update_autotune_table(__class__.__name__ + "_4d_kernel", __class__._4d_kernel)


# -----------------------------------------------------------------------------
# mapping
if deepspeed.HAS_TRITON:
    fp16_matmul = Fp16Matmul()
    matmul = MatmulExt.forward
    matmul_4d = fp16_matmul._matmul_4d
    score_4d_matmul = fp16_matmul._score_4d_matmul
    context_4d_matmul = fp16_matmul._context_4d_matmul
else:
    fp16_matmul = None
    matmul = None
    matmul_4d = None
    score_4d_matmul = None
    context_4d_matmul = None


@atexit.register
def matmul_ext_update_autotune_table():
    if deepspeed.HAS_TRITON:
        fp16_matmul._update_autotune_table()
