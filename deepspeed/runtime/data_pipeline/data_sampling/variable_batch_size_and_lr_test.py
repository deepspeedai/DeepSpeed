# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as torch_dist
import torch.nn.functional as F
import deepspeed
from deepspeed.pipe import PipelineModule

from deepspeed.runtime.data_pipeline.data_sampling.variable_batch_size_and_lr import get_dataloader_and_lr_scheduler_for_variable_batch_size


if __name__ == "__main__":

    class TestData(torch.utils.data.Dataset):
        """ A test dataset with sequences of random length, and their sum as the target"""

        def __init__(self, seq_count, min_seq_len=1, max_seq_len=21, seed=0):
            data_random = random.Random(seed)
            self.seqs = [torch.ones(data_random.randrange(min_seq_len, max_seq_len)) for _ in range(seq_count)]

        __len__ = lambda self: len(self.seqs)
        __getitem__ = lambda self, idx: [self.seqs[idx], self.seqs[idx].sum()]

        # collate_fn merges sequences and trims/pads them to the max_len specified
        @staticmethod
        def collate_fn(batch, max_seqlen=None, padding_value=0):
            if max_seqlen is not None:
                for i, (seq, _) in enumerate(batch):
                    batch[i][0] = torch.nn.ConstantPad1d((0, max_seqlen - seq.shape[0]), padding_value)(seq)
            seqs, labels = zip(*batch)
            padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)
            labels = torch.tensor(labels)
            return padded, labels


    class TestFeedForward(torch.nn.Module):
        """ a test feedforward model """

        def __init__(self):
            super(TestFeedForward, self).__init__()
            self.fc1 = torch.nn.Linear(max_seq_len, 128)
            self.fc2 = torch.nn.Linear(128, 128)
            self.fc3 = torch.nn.Linear(128, 128)
            self.fc4 = torch.nn.Linear(128, 128)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return x.sum()

        def to_layers(self):
            return [self.fc1, self.fc2, self.fc3, self.fc4, lambda x: x.sum()]

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

    torch_dist.init_process_group(backend='nccl')
    model = TestFeedForward().to(device)
    dataset = TestData(seq_count=300, min_seq_len=5, max_seq_len=15)
    model_ddp = DDP(model, device_ids=[device])
    optimizer = torch.optim.Adam(model_ddp.parameters(), lr=1e-3)

    seqlens = [len(s[0]) for s in dataset]  # difficulty = input sequence length

    if pipeline_parallelism:
        collate_fn = lambda b, m: TestData.collate_fn(b, m, padding_value=0)
    else:
        collate_fn = lambda b: TestData.collate_fn(b, padding_value=0)

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
            dataloader_collate_fn=collate_fn,
            optimizer=optimizer,
            # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            # lr_scheduler_kwargs=dict(optimizer=optimizer, step_size=1, gamma=0.1),
            required_microbatches_of_same_size=pipeline_parallelism,
            required_return_of_batch_max_seqlen=pipeline_parallelism,
        )

    # PyTorch example iterating whole dataset in one epoch
    for epoch in range(2):
        for sample_idx, (inputs, labels) in enumerate(dataloader):
            batch_id = sample_idx // gradient_accumulation_steps
            microbatch_id = sample_idx % gradient_accumulation_steps
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ddp(inputs)
            loss = F.mse_loss(outputs, labels)
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
        for sample_idx, (inputs, labels) in enumerate(dataloader):
            batch_id = sample_idx // gradient_accumulation_steps
            microbatch_id = sample_idx % gradient_accumulation_steps
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = engine(inputs)
            loss = F.mse_loss(outputs, labels)
            engine.backward(loss)
            if dataloader_rank == 0:
                print(
                    f"batch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}"
                )
            engine.step()

    # Deepspeed example for pipeline parallelism
    if pipeline_parallelism:
        model = PipelineModule(layers=model.to_layers(), num_stages=2)
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
