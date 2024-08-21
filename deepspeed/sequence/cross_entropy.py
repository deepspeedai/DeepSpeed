# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

import deepspeed.comm as dist


class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, sp_group):
        # vocab_seq_parallel_logits: [S/P, B, V]
        # target: [S/P, B]
        # return: [S, B]

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(softmax.log().view(-1, ctx.vocab_size), target.view(-1), reduction='none')

        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        ctx.sp_world_size = sp_world_size
        ctx.sp_rank = sp_rank
        ctx.seqlen = vocab_seq_parallel_logits.size(0) * sp_world_size
        batch_size = vocab_seq_parallel_logits.size(1)

        loss_all = torch.empty(ctx.seqlen,
                               batch_size,
                               dtype=vocab_seq_parallel_logits.dtype,
                               device=vocab_seq_parallel_logits.device)
        dist.all_gather_into_tensor(loss_all, loss, group=sp_group)

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        step_seqlen = ctx.seqlen // ctx.sp_world_size
        sp_rank = ctx.sp_rank
        grad_output_part = grad_output[step_seqlen * sp_rank:step_seqlen * (sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        return grad_input, None, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, sp_group):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, sp_group)
