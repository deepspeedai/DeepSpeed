# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as torch_dist
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule

from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import get_dataloader_and_lr_scheduler_for_variable_batch_size


if __name__ == "__main__":

    class TestData(torch.utils.data.Dataset):
        """ A test dataset with sequences of random length, and the sequence length as the label"""

        def __init__(self, seq_count, min_seqlen=1, max_seqlen=21, embed_dim=5, seed=0):
            data_random = random.Random(seed)
            self.mask_size = max_seqlen # M: size of mask
            self.padding_value = 0
            self.embed_dim = embed_dim
            self.seqs = [torch.ones(data_random.randrange(min_seqlen, max_seqlen), embed_dim) for _ in range(seq_count)]

        __len__ = lambda self: len(self.seqs)
        __getitem__ = lambda self, idx: ( self.seqs[idx], len(self.seqs[idx]) )

        def collate_fn(self, batch):
            """ pad sequences of different lenghts into batch of size BxTxE """
            seqs, labels = zip(*batch)
            seqlens = torch.tensor([ len(s) for s in seqs ]) 
            seqs = nn.utils.rnn.pad_sequence([s[0] for s in batch], batch_first=True, padding_value=self.padding_value)
            labels = torch.tensor([s[1] for s in batch], dtype=float)
            return seqs, seqlens, labels


    class SingleHeadAttentionAndFeedForward(nn.Module):
        """ a test feedforward model """

        def __init__(self, max_seqlen, embed_dim):
            super(SingleHeadAttentionAndFeedForward, self).__init__()

            self.padding_value = 0
            self.max_seqlen = max_seqlen # M: size of mask
            self.attn_head = nn.MultiheadAttention(embed_dim, num_heads=1)
            self.fc1 = nn.Linear(embed_dim, 128)
            self.fc2 = nn.Linear(128, embed_dim)

        def forward(self, x, attn_mask_seqlens=None):

            # optional: 3D masks for attention, padded to individual input sequence lengths
            B, T = len(x), max(attn_mask_seqlens)
            if attn_mask_seqlens is not None:
                masks = torch.tril(torch.ones((B,T,T), dtype=torch.float32)).to(x[0].device)
                for i, seqlen in enumerate(attn_mask_seqlens):
                    masks[i, seqlen:, :] = masks[i, :, seqlen:] = 0

            # collates sequences of different lengths into a batch of size BxTxE
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_value)

            # compute q@k / sqrt(d_k) on input of shape BxTxE (where B and T can change)
            k, q, v = x, x, x
            out = q@k.transpose(-2,-1) # BxTxE @ BxExT --> BxTxT
            out = out / (x.shape[-1]**0.5) # √d_k

            if attn_mask_seqlens is not None: # mask, if needed
                out = out.masked_fill(masks==0, value=float('-inf'))

            # softmax and multiply by values vector
            out = F.softmax(out, dim=-1) # softmax --> BxTxT 
            out = out@v # BxTxT @ BxTxE --> BxTxE

            # feedforward: needs to converts BxTxE to BxMxE by padding extra tokens
            out = F.pad(out, pad=(0, 0, 0, self.max_seqlen-T), value=self.padding_value)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            return torch.tensor(out.sum(-1).sum(-1), requires_grad=True, dtype=float)


        def to_layers(self):
            return [self.fc1, self.fc2, lambda x: x.sum(-1).sum(-1)]

    dataloader_rank = int(os.environ.get('RANK', 0))
    dataloader_num_replicas = int(os.environ.get('WORLD_SIZE', 1))
    device_id = int(os.environ.get('LOCAL_RANK', 0))
    device = f"cuda:{device_id}"
    max_seqlen_per_batch = 40
    base_batch_size = 8
    base_lr = 1e-3
    gradient_accumulation_steps = base_batch_size // dataloader_num_replicas
    pipeline_parallelism = True
    order_by_seqlen = True  #enable for curriculum

    max_seqlen = 15
    torch_dist.init_process_group(backend='nccl')
    dataset = TestData(seq_count=300, min_seqlen=5, max_seqlen=max_seqlen)
    model = SingleHeadAttentionAndFeedForward(max_seqlen, dataset.embed_dim).to(device)
    model_ddp = DDP(model, device_ids=[device])
    optimizer = torch.optim.Adam(model_ddp.parameters(), lr=1e-3)
    loss_fn = lambda x, y: F.mse_loss(x, y)

    seqlens = [len(s[0]) for s in dataset]
    dataloader, lr_scheduler, deepspeed_io_kwargs = \
        get_dataloader_and_lr_scheduler_for_variable_batch_size(
            dataset=dataset,
            dataset_seqlens=seqlens,
            base_batch_size=base_batch_size,
            max_seqlen_per_batch=max_seqlen_per_batch,
            dataloader_rank=dataloader_rank,
            dataloader_num_replicas=dataloader_num_replicas,
            lr_scaling_method="linear",
            order_by_seqlen=order_by_seqlen,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=0,
            dataloader_collate_fn=dataset.collate_fn,
            optimizer=optimizer,
            # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs=dict(optimizer=optimizer, step_size=1, gamma=0.1),
            required_microbatches_of_same_size=pipeline_parallelism,
        )

    # PyTorch example iterating whole dataset in one epoch
    for epoch in range(2):
        for sample_idx, (seqs, seqlens, labels) in enumerate(dataloader):
            batch_id = sample_idx // gradient_accumulation_steps
            microbatch_id = sample_idx % gradient_accumulation_steps
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model_ddp(seqs, seqlens)
            loss = loss_fn(outputs, labels)
            loss.backward()
            if (microbatch_id + 1) % gradient_accumulation_steps == 0:
                if dataloader_rank == 0:
                    print(f"batch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}")
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

    torch_dist.destroy_process_group()

    # DeepSpeed example
    config = {
        "train_batch_size": base_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": base_lr
            }
        },
    }

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        config=config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    # engine.training_dataloader = dataloader #use this or the deepspeed_io()
    engine.training_dataloader = engine.deepspeed_io(**deepspeed_io_kwargs)

    lr_scheduler.step(0)  # reset LR scheduler
    for epoch in range(2):
        for sample_idx, (seqs, seqlens, labels) in enumerate(dataloader):
            batch_id = sample_idx // gradient_accumulation_steps
            microbatch_id = sample_idx % gradient_accumulation_steps
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = engine(seqs, seqlens)
            loss = loss_fn(outputs, labels)
            engine.backward(loss)
            if dataloader_rank == 0:
                print(
                    f"batch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}"
                )
            engine.step()

    # Deepspeed example for pipeline parallelism
    if pipeline_parallelism:
        model = PipelineModule(layers=model.to_layers(), num_stages=2, loss_fn=loss_fn)
        engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            config=config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        engine.training_dataloader = engine.deepspeed_io(**deepspeed_io_kwargs)

        dataloader_it = iter(dataloader)  # reset dataloader
        lr_scheduler.step(0)  # reset LR scheduler
        for epoch in range(2):
            for batch_id in range(len(dataloader) // gradient_accumulation_steps):
                engine.reset_activation_shape() # each batch has a diff length
                loss = engine.train_batch(data_iter=dataloader_it)
                if dataloader_rank == 0:
                    print(f"batch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}")
