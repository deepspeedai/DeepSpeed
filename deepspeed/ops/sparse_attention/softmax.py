# DeepSpeed note, code taken & adapted from commit 9aa94789f13ada713af36cfd8cca2fc9a7f6b79a
# https://github.com/ptillet/torch-blocksparse/blob/master/torch_blocksparse/matmul.py

import warnings
import importlib
import torch
import math
from .trsrc import softmax_fwd, softmax_bwd

fwd_kernels = dict()
bwd_kernels = dict()

# Delay importing triton unless we need it
triton = None


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
    def make_kernel(cache,
                    src,
                    max_k,
                    dtype,
                    block,
                    apply_scale,
                    apply_rpe,
                    apply_kp_mask,
                    apply_attn_mask,
                    kp_mask_mode,
                    attn_mask_mode):
        global triton
        if triton is None:
            triton = importlib.import_module('triton')

        if max_k >= 32768:
            raise NotImplementedError('Reductions larger than 32768 elements '\
                                      'are not yet implemented')
        num_warps = 4 if max_k < 512 else (8 if max_k < 2048 else 16)
        pad = num_warps * 32 * 2
        TN = (int(max_k) + pad - 1) // pad * pad
        # just-in-time compile kernel
        key = (block,
               dtype,
               num_warps,
               TN,
               apply_scale,
               apply_rpe,
               apply_kp_mask,
               apply_attn_mask,
               kp_mask_mode,
               attn_mask_mode)
        if key not in cache:
            defines = {
                'TM': 1,
                'TN': TN,
                'TYPE': dtype,
                'BLOCK': block,
                'INFINITY': {
                    torch.float32: 'F32_INFINITY',
                    torch.float16: 'F16_INFINITY'
                }[dtype]
            }
            if apply_scale:
                defines['APPLY_SCALE'] = True
            if apply_rpe:
                defines['APPLY_RPE'] = True
            if apply_kp_mask:
                defines['APPLY_KP_MASK'] = True
                if kp_mask_mode == 'mul':
                    defines['KP_MASK_MUL'] = True
            if apply_attn_mask:
                defines['APPLY_ATTN_MASK'] = True
                if attn_mask_mode == 'mul':
                    defines['ATTN_MASK_MUL'] = True
            kernel = triton.kernel(src,
                                   defines=defines,
                                   device=torch.device('cuda'),
                                   num_warps=num_warps)
            cache[key] = kernel
        return cache[key]

    @staticmethod
    def forward(ctx,
                x,
                scale,
                rpe,
                key_padding_mask,
                attn_mask,
                kp_mask_mode,
                attn_mask_mode,
                spdims,
                block,
                lut,
                num_blocks,
                maxlut,
                bench,
                time):
        global triton
        if triton is None:
            triton = importlib.import_module('triton')

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
        kernel = _sparse_softmax.make_kernel(fwd_kernels,
                                             softmax_fwd,
                                             maxlut * block,
                                             x.dtype,
                                             block,
                                             apply_scale,
                                             apply_rpe,
                                             apply_kp_mask,
                                             apply_attn_mask,
                                             kp_mask_mode,
                                             attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [triton.cdiv(spdims[0] * spdims[1] * block, opt.TM), M]

        # run kernel
        time[0] = kernel(x.data_ptr(), scale, lut.data_ptr(), rpe.data_ptr(), key_padding_mask.data_ptr(), attn_mask.data_ptr(),\
                         num_blocks, maxlut,\
                         x.stride(0),\
                         stride_zrpe, stride_hrpe,\
                         stride_srpe,\
                         stride_zkpm, stride_zattnm,\
                         grid=grid)
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
        global triton
        if triton is None:
            triton = importlib.import_module('triton')

        # retrieve from context
        x, lut = ctx.saved_tensors
        # run kernel
        kernel = _sparse_softmax.make_kernel(bwd_kernels,
                                             softmax_bwd,
                                             ctx.maxlut * ctx.block,
                                             x.dtype,
                                             ctx.block,
                                             ctx.apply_scale,
                                             ctx.apply_rpe,
                                             ctx.apply_kp_mask,
                                             ctx.apply_attn_mask,
                                             ctx.kp_mask_mode,
                                             ctx.attn_mask_mode)
        M = x.shape[0]
        grid = lambda opt: [
            triton.cdiv(ctx.spdims[0] * ctx.spdims[1] * ctx.block,
                        opt.TM),
            M
        ]
        kernel(x.data_ptr(),
               ctx.scale,
               dx.data_ptr(),
               lut.data_ptr(),
               ctx.maxlut,
               x.stride(0),
               dx.stride(0),
               grid=grid)
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
            self.lut_cache[key] = _sparse_softmax.make_lut(self.layout,
                                                           self.block,
                                                           device)
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
        x = Softmax.sparse_softmax(x,
                                   scale,
                                   rpe,
                                   key_padding_mask,
                                   attn_mask,
                                   key_padding_mask_mode,
                                   attn_mask_mode,
                                   self.spdims,
                                   self.block,
                                   lut,
                                   self.num_blocks,
                                   maxlut,
                                   self.bench,
                                   time_y)
        self.time_y = time_y[0]
        return x
