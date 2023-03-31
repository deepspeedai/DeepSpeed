# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# DeepSpeed note, code taken & adapted from commit 9aa94789f13ada713af36cfd8cca2fc9a7f6b79a
# https://github.com/ptillet/torch-blocksparse/blob/master/torch_blocksparse/matmul.py

import torch

import triton
import triton.language as tl


def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def num_warps(n):
    if n < 512:
        return 4
    if n < 2048:
        return 8
    return 16


@triton.heuristics({'num_warps': lambda *args, **meta: num_warps(args[6] * meta['BLOCK'])})
@triton.heuristics({'TN': lambda *args, **meta: next_power_of_2(args[6] * meta['BLOCK'])})
@triton.jit
def _forward(X, scale, LUT, RPE, KP_M, ATTN_M, sizemax, stride_zx, stride_zrpe, stride_hrpe, stride_srpe, stride_zkpm,
             stride_zattnm, **meta):
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from LUT
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # block id and column id
    blockid = tl.load(LUT + offset + rbmn * 4 + 0)
    columnid = tl.load(LUT + offset + rbmn * 4 + 1)
    rowid = tl.load(LUT + offset + rbmn * 4 + 2)
    headid = tl.load(LUT + offset + rbmn * 4 + 3)
    # pointers to X
    px = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    x = tl.load(px, mask=check, other=-float('inf'))
    x = x.to(tl.float32)
    # apply scale
    if meta['APPLY_SCALE']:
        x = x * scale
    # apply RPE
    if meta['APPLY_RPE']:
        prpe = RPE + pidz * stride_zrpe + headid * stride_hrpe + columnid * BLOCK + rowid * BLOCK * stride_srpe + rxm * stride_srpe + rxn
        rpe = tl.load(prpe, mask=check, other=0)
        x = x + rpe
    # apply key-padding mask
    if meta['APPLY_KP_MASK']:
        pkp_m = KP_M + pidz * stride_zkpm + columnid * BLOCK + rxn
        kp_m = tl.load(pkp_m, mask=check, other=-float('inf'))
        if meta['KP_MASK_MUL']:
            kp_m = tl.where(kp_m == 0, -float('inf'), 0.)
        x = x + kp_m
    # apply attention mask
    if meta['APPLY_ATTN_MASK']:
        pattn_m = ATTN_M + columnid * BLOCK + rowid * BLOCK * stride_zattnm + rxm * stride_zattnm + rxn
        attn_m = tl.load(pattn_m, mask=check, other=-float('inf'))
        if meta['ATTN_MASK_MUL']:
            attn_m = tl.where(attn_m == 0, -float('inf'), 0.)
        x = x + attn_m
    # computation
    x = tl.softmax(x)
    tl.store(px, x, mask=check)


@triton.heuristics({'num_warps': lambda *args, **meta: num_warps(args[4] * meta['BLOCK'])})
@triton.heuristics({'TN': lambda *args, **meta: next_power_of_2(args[4]) * meta['BLOCK']})
@triton.jit
def _backward(X, scale, DX, LUT, sizemax, stride_zx, stride_zdx, **meta):
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    # create index ranges
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    # extract information from look-up table
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    # bounds checking on lut
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    # initialize pointers to block-sparse input
    blockid = tl.load(LUT + offset + rbmn * 4)
    X = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    DX = DX + pidz * stride_zdx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    # compute fused softmax backward
    x = tl.load(X, mask=check, other=0)
    dx = tl.load(DX, mask=check, other=0)
    x = x.to(tl.float32)
    dx = dx.to(tl.float32)
    y = x * (dx - tl.sum(x * dx, 0)) * scale
    tl.store(DX, y, mask=check)


class _sparse_softmax(torch.autograd.Function):

    bwd_kernels = dict()

    @staticmethod
    def make_lut(layout, block, device):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        # sizes along rows
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
        # offsets in block format
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        # block indices
        idx = torch.arange(layout.sum())
        head = layout.nonzero()[:, 0]
        rows = layout.nonzero()[:, 1]
        columns = layout.nonzero()[:, 2]
        core = torch.stack((idx, columns, rows, head), dim=1).view(-1)
        # construct look-up table
        offsets = offsets * 4 + 2 * sizes.numel()
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, core)).type(torch.int32).to(device)
        return lut, int(sizes.max())

    @staticmethod
    def forward(ctx, x, scale, rpe, key_padding_mask, attn_mask, kp_mask_mode, attn_mask_mode, spdims, block, lut,
                num_blocks, maxlut, bench, time):

        apply_scale = False if scale == 1.0 else True

        # handle None rpe
        if rpe is None:
            apply_rpe = False
            stride_zrpe, stride_hrpe, stride_srpe = 0, 0, 0
            rpe = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_rpe = True
            stride_zrpe, stride_hrpe, stride_srpe = rpe.stride(0), rpe.stride(1), rpe.stride(2)

        # handle None key_padding_mask
        if key_padding_mask is None:
            apply_kp_mask = False
            stride_zkpm = 0
            key_padding_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_kp_mask = True
            stride_zkpm = key_padding_mask.stride(0)

        # handle None attention_mask
        if attn_mask is None:
            apply_attn_mask = False
            stride_zattnm = 0
            attn_mask = torch.empty(0, dtype=x.dtype, device=x.device)
        else:
            apply_attn_mask = True
            stride_zattnm = attn_mask.stride(0)

        # run kernel
        M = x.shape[0]
        meta = {
            'BLOCK': block,
            'APPLY_SCALE': apply_scale,
            'APPLY_RPE': apply_rpe,
            'APPLY_KP_MASK': apply_kp_mask,
            'APPLY_ATTN_MASK': apply_attn_mask,
            'KP_MASK_MUL': kp_mask_mode == 'mul',
            'ATTN_MASK_MUL': attn_mask_mode == 'mul',
        }
        grid = lambda opt: [spdims[0] * spdims[1] * block, M]
        _forward[grid](x, scale, lut, rpe, key_padding_mask, attn_mask, maxlut, x.stride(0),\
                       stride_zrpe, stride_hrpe, stride_srpe, stride_zkpm, stride_zattnm, **meta)

        # save to context
        ctx.mark_dirty(x)
        ctx.save_for_backward(x, lut)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        ctx.scale = scale
        ctx.apply_scale = apply_scale
        ctx.apply_rpe = apply_rpe
        ctx.apply_kp_mask = apply_kp_mask
        ctx.apply_attn_mask = apply_attn_mask
        ctx.kp_mask_mode = kp_mask_mode
        ctx.attn_mask_mode = attn_mask_mode
        return x

    @staticmethod
    def backward(ctx, dx):

        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        M = x.shape[0]
        grid = lambda opt: [ctx.spdims[0] * ctx.spdims[1] * ctx.block, M]
        _backward[grid](x, ctx.scale, dx, lut, ctx.maxlut, x.stride(0), dx.stride(0), BLOCK=ctx.block)
        return dx, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class Softmax:
    """Block-Sparse Softmax class; this class computes softmax on a block sparse matrix. It is also able to apply either/all of the following masks:
       - relative position embedding
       - key padding mask
       - attention mask

    For more details about sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509
    """

    def sparse_softmax(*args, **kwargs):
        return _sparse_softmax.apply(*args, **kwargs)

    def make_lut(self, device):
        """Generates the sparsity layout used in block-sparse softmax
        """
        key = (device, )
        if key not in self.lut_cache:
            self.lut_cache[key] = _sparse_softmax.make_lut(self.layout, self.block, device)
        return self.lut_cache[key]

    def __init__(self, layout, block, bench=False):
        """Initialize the Block-Sparse Softmax class.

        Arguments:
             layout: required: sparsity layout tensor
             block: required: an integer determining the block size.
             bench: optional: set if you want to do benchmarking
        """

        self.num_blocks = layout.sum().item()
        self.spdims = layout.shape
        self.layout = layout
        self.block = block
        self.bench = bench
        self.lut_cache = dict()

    def __call__(self,
                 x,
                 scale=1.,
                 rpe=None,
                 key_padding_mask=None,
                 attn_mask=None,
                 key_padding_mask_mode='add',
                 attn_mask_mode='add'):
        """Applies softmax on a Block-Sparse input tensor.

        For more details about sparsity config, please see `Generative Modeling with Sparse Transformers`: https://arxiv.org/abs/1904.10509

        Arguments:
             x: required: a block-sparse tensor that softmax is applied on it; computation will be in place and result will be returned in the same tensor
             scale: optional: a float value; x values will be multiplied by this value before normalization. Default value is 1.0.
             rpe: optional: a tensor same dimension as x that is used as relative position embedding
             key_padding_mask: optional: a mask tensor of size (BatchSize X SequenceLength)
             attn_mask: optional: a mask tensor of size (SequenceLength X SequenceLength); currently only 2D is supported
             key_padding_mask_mode: optional: a boolean determining if key_padding_mask needs to be added or multiplied
             attn_mask_mode: optional: a boolean determining if attn_mask needs to be added or multiplied

        Return:
             x: a block-sparse tensor contains normalized input x using softmax; and masks applied if given
        """

        time_y = [None]
        if rpe is not None and rpe.dtype != x.dtype:
            raise ValueError('relative position embedding must be %s' % x.dtype)
        if attn_mask is not None and attn_mask.dtype != x.dtype:
            raise ValueError('Attention mask must be %s' % x.dtype)
        if key_padding_mask is not None and key_padding_mask.dtype != x.dtype:
            raise ValueError('Key padding mask must be %s' % x.dtype)
        lut, maxlut = self.make_lut(x.device)
        x = Softmax.sparse_softmax(x, scale, rpe, key_padding_mask, attn_mask, key_padding_mask_mode, attn_mask_mode,
                                   self.spdims, self.block, lut, self.num_blocks, maxlut, self.bench, time_y)
        self.time_y = time_y[0]
        return x
