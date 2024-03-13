# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import deepspeed.comm as dist
from deepspeed.pipe import PipelineModule

from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import get_dataloader_and_lr_scheduler_for_variable_batch_size_deepspeed

if __name__ == "__main__":

    class TestData(torch.utils.data.Dataset):
        """ A test dataset with sequences of random length, and the sequence length as the label"""

        def __init__(self, seq_count, min_seqlen=1, max_seqlen=20, embed_dim=5, seed=0):
            data_random = random.Random(seed)
            self.padding_value = 0
            self.embed_dim = embed_dim
            self.seqs = [
                torch.ones(data_random.randrange(min_seqlen, max_seqlen), embed_dim) for _ in range(seq_count)
            ]

        __len__ = lambda self: len(self.seqs)
        __getitem__ = lambda self, idx: (self.seqs[idx], len(self.seqs[idx]))

        def collate_fn(self, batch):
            """ collate sequences of different lengths into batch of size BxTxE, where T is max seqlen """
            seqs, labels = zip(*batch)
            seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=self.padding_value)
            labels = torch.tensor(labels, dtype=float)
            return seqs, labels

        def padding_fn(self, sample, size):
            """ pad sequence `seq` of shape TxE to size T'xE where T' is given by `size` """
            seq, label = sample
            seq = F.pad(seq, pad=(0, 0, 0, size - len(seq)), value=self.padding_value)
            return seq, label

    class AttentionHeadAndFeedForward(nn.Module):
        """
        A single attention head of batch of shape BxTxE (with variable T) and attention matrix
        BxTxT, followed by a feed-forward network of input size BxMxE, where T<<M. No embeddings.
        """

        def __init__(self, max_seqlen, embed_dim):
            super(AttentionHeadAndFeedForward, self).__init__()
            self.padding_value = 0
            self.max_seqlen = max_seqlen  # M: max possible seqlen, and input size to feedforward
            self.qe = nn.Linear(embed_dim, embed_dim)
            self.ke = nn.Linear(embed_dim, embed_dim)
            self.ve = nn.Linear(embed_dim, embed_dim)
            self.attn_head = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
            self.fc1 = nn.Linear(embed_dim, 128)
            self.fc2 = nn.Linear(128, embed_dim)
            self.nansum_of_last_two_dims = lambda x: x.nansum(-1).nansum(-1)

        def forward(self, x):

            # compute length of each sequence as first index of padding value, or max length if no padding
            B, T, E = x.shape
            seqlens = torch.full(size=(B, ), fill_value=T, dtype=int, device=x.device)
            seq_ids, seq_padding_ids = torch.where(x[:, :, 0] == self.padding_value)
            seqlens[seq_ids] = seq_padding_ids

            # optional: 3D masks for attention, shaped BxTxT, padded to individual input sequence lengths
            masks = torch.tril(torch.ones((B, T, T), dtype=bool)).to(x.device)
            for i, seqlen in enumerate(seqlens):
                masks[i, seqlen:, :] = masks[i, :, seqlen:] = False

            # linear projections and attention head. Attention size BxTxT
            q, k, v = self.qe(x), self.ke(x), self.ve(x)
            out, _ = self.attn_head(q, k, v, need_weights=False, attn_mask=masks)

            # feedforward: needs to convert BxTxE to BxMxE by padding extra tokens
            out = F.pad(out, pad=(0, 0, 0, self.max_seqlen - T), value=self.padding_value)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            return self.nansum_of_last_two_dims(out)

        def to_layers(self):
            return [self.fc1, self.fc2, self.nansum_of_last_two_dims]

    deepspeed.init_distributed()
    device = f"cuda:{dist.get_local_rank()}"
    pipeline_num_stages = 2

    max_seqlen = 15
    dataset = TestData(seq_count=300, min_seqlen=5, max_seqlen=max_seqlen)
    dataset_seqlens = [len(s[0]) for s in dataset]
    model = AttentionHeadAndFeedForward(max_seqlen, dataset.embed_dim).to(device)
    loss_fn = lambda x, y: F.mse_loss(x.float(), y.float())

    if pipeline_num_stages > 0:
        model = PipelineModule(layers=model.to_layers(), num_stages=pipeline_num_stages, loss_fn=loss_fn)

    # DeepSpeed config includes the dynamic batching
    config = {
        "train_batch_size": 16,
        "train_micro_batch_size_per_gpu": 2,  # Note: each microbatch per GPU will fill up to N tokens
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.001,
                "warmup_max_lr": 0.005,
                "warmup_num_steps": 1000
            }
        },
        "data_efficiency": {
            "enabled": True,
            "dynamic_batching": {
                "enabled": True,
                "dataloader_num_workers": 0,
                "dataloader_pin_memory": 0,
                "lr_scaling_method": "linear",
                "min_batch_size": 1,
                "max_batch_size": 10,
                "samples_order": "dataloader",  # "random" / "seqlen" / "default"
                "max_tokens_per_batch": 40,
                "verbose": False,
            }
        },
    }

    # initialize deepspeed engine without dataset/dataloader
    engine, _, _, _ = deepspeed.initialize(config=config, model=model)

    # We will simulate a curriculum step, by filtering only a subset of sequences with a given seqlen
    dataset_filter_ids = [i for i, seqlen in enumerate(dataset_seqlens) if seqlen > 7 and seqlen < 14]
    dataloader, lr_scheduler, _ = \
        get_dataloader_and_lr_scheduler_for_variable_batch_size_deepspeed(
            dataset=dataset,
            dataset_seqlens=dataset_seqlens,
            dataset_filter_ids=dataset_filter_ids, #remove or None to include the whole dataset
            engine=engine,
            dataloader_collate_fn=dataset.collate_fn,
            sample_padding_fn=dataset.padding_fn)

    gradient_acc_steps = engine.gradient_accumulation_steps()
    n_batches_per_rank = len(dataloader) // (gradient_acc_steps * engine.train_micro_batch_size_per_gpu())

    for epoch in range(10):
        data_iter = iter(dataloader)  # point data iterator to first batch
        lr_scheduler.step(0)  # point LR scheduler to first batch
        for batch_id in range(n_batches_per_rank):
            if pipeline_num_stages > 0:
                engine.reset_activation_shape()  # reset, as each batch has a diff BxT dimension
                loss = engine.train_batch(data_iter=data_iter)  # lr_kwargs={"epoch": batch_id}
            else:
                for i in range(gradient_acc_steps):
                    seqs, labels = next(data_iter)
                    seqs, labels = seqs.to(device), labels.to(device)
                    outputs = engine(seqs)
                    loss = loss_fn(outputs, labels)
                    engine.backward(loss)
                    engine.step()  # lr_kwargs={"epoch": batch_id})

            if engine.data_parallel_group.rank():
                print(f"epoch {epoch}, batch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}")
