import random
import os
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
import deepspeed
from deepspeed.utils import logger
from deepspeed.pipe import PipelineModule


def batch_by_size(
    metric_values,
    max_metric_value_per_batch,
    sample_ids=None,
    min_batch_size=1,
    max_batch_size=None,
    shuffle_metric_values=False,
    order_by_metric_value=False,
    dataloader_num_replicas=1,
    gradient_accumulation_steps=1,
    required_microbatches_of_same_size=False,
    verbose=False,
    seed=0,
    ):

    """
    Yield mini-batches of indices bucketed by size. Batches may contain sequences of different lengths.
    Similar to "Attention is all you need", Section 5.1: 
    "Sentence pairs were batched together by approximate sequence length. Each training batch
    contained a set of sentence pairs containing approximately X source tokens and X target tokens"

    Arguments:
    - `metric_values`: a list of difficulties (metric values) for every sample in the dataset;
    - `max_metric_value_per_batch`: upper cap in total difficulty in a batch;
    - `sample_ids`: user-defined ids of the samples in metric_values. If not provided,
      automatically assigns a sequential order;
    - `min_batch_size`: smallest allowed size of a batch;
    - `min_batch_size`: largest allowed size of a batch;
    - `shuffle_metric_values`: shuffle metric values before packing samples into batches;
    - `order_by_metric_value`: order samples by ascending metric values before packing into batches;
    - `dataloader_num_replicas`: number of dataloaders
    - `gradient_accumulation_steps`: number of gradient accumulation steps;
    - `required_microbatches_of_same_size`: enable if each mini-batch (in a total of `batch_size_multiple`
       micro-batches per batch), should have all micro-batches with the same batch size.
       Required for pipeline parallelism (as activation shapes is uniform across mini-batches), or
       in regular data parallelism if we want the same number of samples per accumulation step.

    Returns a list of the ids of each micro-batch and a list of effective batch sizes.
    """

    assert not shuffle_metric_values or not order_by_metric_value, \
        "either sort_metric_values or shuffle_metric_values can be True, not both."

    sample_ids = sample_ids or list(range(len(metric_values))) 
    metrics = list(zip(metric_values, sample_ids))

    if shuffle_metric_values:
        metric_random = random.Random(seed)
        metric_random.shuffle(metrics)
    if order_by_metric_value:
        metrics = sorted(metrics)

    # go through metrics and warn user and filter samples that alone exceed the max batch threshold
    long_ids = [ idx for val, idx in metrics if val>max_metric_value_per_batch ]
    if len(long_ids)>0:
        logger.warning(f"Data indices {long_ids} ignored as metrics exceed {max_metric_value_per_batch}.")
        logger.info(f"Original dataset length: {len(metrics)}. New dataset length: {len(long_ids)}")
        metrics = [ m for m in metrics if m[1] not in long_ids ]

    def is_microbatch_valid(metrics):
        if min_batch_size and len(metrics)<min_batch_size: return False # insufficient sample count
        if max_batch_size and len(metrics)>max_batch_size: return False # too many samples
        if sum([m[0] for m in metrics]) > max_metric_value_per_batch: return False # exceeds max
        return True
    
    # go through all samples and pack then in microbatches of metric sums below the threshold
    # `required_microbatches_of_same_size` means all minibatches in a batch must be of equal size
    num_microbatches_per_batch = dataloader_num_replicas * gradient_accumulation_steps
    equal_size_multiple = num_microbatches_per_batch if required_microbatches_of_same_size else 1
    microbatches = []
    batch_init = 0
    while batch_init < len(metrics):

        # we iterate over possible effective batch sizes (groups of microbatches of same size)
        for batch_size in range(equal_size_multiple, len(metrics), equal_size_multiple):

            # attempt effective batch
            batch = metrics[batch_init:batch_init+batch_size]

            # pick interleaved samples for each microbatch to help with load balancing
            # (in the ordered use case), and to replicate what the distributed sampler does.
            microbatch = [ batch[b::equal_size_multiple] for b in range(equal_size_multiple) ]

            # if they are all valid micro-batches, keep them until you find longer mbatches, if any
            is_batch_valid = all([is_microbatch_valid(mb) for mb in microbatch] )
            if not is_batch_valid:
                break

        if not is_batch_valid: batch_size -= equal_size_multiple #ignore last iteration (not valid)
        batch = metrics[batch_init:batch_init+batch_size]
        microbatch = [ batch[b::equal_size_multiple] for b in range(equal_size_multiple) ]
        batch_init += sum( [ len(l) for l in microbatch ] )
        microbatches += microbatch

    # make sure we give the same number of (micro-)batches to each dataloader by trimming dataset
    microbatches = microbatches[:len(microbatches) - len(microbatches) % num_microbatches_per_batch]

    #compute the effective batch size for each microbatch.
    batch_sizes, batch_metrics, microbatch_sample_ids = [], [], []
    for rank in range(0, len(microbatches), num_microbatches_per_batch):
        microbatch = microbatches[rank: rank+num_microbatches_per_batch]
        batch_size = sum([len(mb) for mb in microbatch])
        batch_metric = sum([m[0] for m in microbatch[0]])
        batch_sample_ids = [ [m[1] for m in metrics] for metrics in microbatch]
        batch_sizes.append(batch_size)
        batch_metrics.append(batch_metric)
        microbatch_sample_ids += batch_sample_ids
        if verbose:
            print(f"Batch size {batch_size} samples, metric value {batch_metric}, samples: {batch_sample_ids}")

    # return the sample ids of each microbatch, and the batch sizes 
    assert len(batch_sizes) == len(microbatch_sample_ids)//num_microbatches_per_batch
    return microbatch_sample_ids, batch_sizes, batch_metrics
        

def scale_lr(base_batch_size, batch_size, base_lr=1, method="linear"):
    """ given a reference lr and batch_size, compute the new LR for a given batch size """
    if method == "linear":
        # Linear Scaling Rule: "When the minibatch size is multiplied by k, multiply the learning
        # rate by k" (Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al)
        return base_lr * batch_size / base_batch_size
    if method == "sqrt":
        # Square Root scaling: "when multiplying the batch size by k, multiply the learning rate
        # by √k, to keep the variance in the gradient expectation constant" 
        # (A. Krizhevsky. One weird trick for parallelizing convolutional neural networks)
        return base_lr * torch.sqrt(batch_size / base_batch_size)
    raise ValueError("Unknown scaling method: {}".format(method))


def dataloader_for_variable_batch_size(dataset, 
        microbatch_sample_ids, dataloader_rank, dataloader_num_replicas, dataloader_collate_fn,
        dataloader_num_workers=2, dataloader_pin_memory=False): 

        # equidistantly distribute the microbatches across the replicas in an interleaved fashion.
        sampler = DistributedSampler(
            dataset=microbatch_sample_ids,
            num_replicas=dataloader_num_replicas,
            rank=dataloader_rank,
            shuffle=False,
            drop_last=False,
            )

        # collate function applies wraps user defined collate function to the variable batch data  
        def collate_fn_wrapper(batch_sample_ids, dataset, collate_fn=None):
            # batch is a list of sample ids per microbatch
            assert len(batch_sample_ids)==1, "only 1 element should be returned by the sampler."
            batch_data = [dataset[idx] for idx in batch_sample_ids[0]]
            return collate_fn(batch_data) if collate_fn else batch_data

        collate_fn = lambda b: collate_fn_wrapper(b, dataset, dataloader_collate_fn)

        dataloader = DataLoader(
            dataset=microbatch_sample_ids,
            sampler=sampler,
            num_workers = dataloader_num_workers,
            collate_fn = collate_fn,
            pin_memory=dataloader_pin_memory,)

        deepspeed_io_kwargs = dict(
            dataset=microbatch_sample_ids,
            batch_size=1,
            pin_memory=dataloader_pin_memory,
            data_sampler=sampler,
            collate_fn=collate_fn,
            num_local_io_workers=dataloader_num_workers,)
        
        return dataloader, deepspeed_io_kwargs
        

class StubLRScheduler(LRScheduler):
    """ a stub LR scheduler that does not change the LR, keeps it constant """
    def get_lr(self) -> float:
        return self.base_lrs

def lr_scheduler_for_variable_batch_size(
        base_batch_size, batch_sizes, dataloader, batch_metrics,
        lr_scaling_method='linear', optimizer=None, lr_scheduler_class=None,
        **lr_scheduler_kwargs):
    """
    returns a class that provides an LR scheduler that scales learning rate at every
    epoch taking into account the batch size of each epoch.
    If learning rate is constant, ie no LR scheduler, then `optimizer` must be provided.
    Otherwise, the base `LRScheduler` must be provided as  `lr_scheduler_class`.

    Arguments:
    - `base_batch_size`: the batch size that the base LR in the optimizer or scheduler refers to;
    - `lr_scaling_method`: method to use to scale LR - see `scale_lr()`;
    - `batch_sizes`: the effective batch size of each batch in the dataloader;
    - `optimizer` and `lr_scheduler_class`: the base LR scheduler. It not provided,
       will use the constant LRs from the optimizer's param groups instead. If provided,
       the initialization of the scheduler will be done with `lr_scheduler_kwargs`.

    Returns the new LRScheduler
    """

    class VariableBatchSizeLR(lr_scheduler_class or StubLRScheduler):

        def __init__(self, optimizer, **lr_scheduler_kwargs):
            self.batch_sizes = batch_sizes
            self.batch_metrics = batch_metrics
            self.base_batch_size = base_batch_size
            self.lr_scaling_method = lr_scaling_method
            self.dataloader = dataloader
            self._last_lr = [p['lr'] for p in optimizer.param_groups]
            super().__init__(optimizer=optimizer, **lr_scheduler_kwargs)
        
        def state_dict(self):
            return {
                'base': super().state_dict(),
                'base_batch_size': self.base_batch_size,
                'lr_scaling_method': self.lr_scaling_method,
                'batch_sizes': self.batch_sizes,
                }

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict['base'])
            self.base_batch_size = state_dict['base_batch_size']
            self.lr_scaling_method = state_dict['lr_scaling_method']
            self.batch_sizes = state_dict['batch_sizes']

        def get_lr(self):
            return [group['lr'] for group in self.optimizer.param_groups]

        def step(self, epoch=None):
            # call the base scheduler's step method to get LR for next epoch
            # Note: optimizer.step preceeds lr_scheduler.step(), so the stepping workflow is:
            # init: lr_scheduler.step(0) --> set LR for epoch 0
            # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
            # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

            # reset unscaled LRs (to the original scheduler's one) for the current epoch
            # Note: epoch==0: reset LR scheduler; epoch==None: scale LR for next epoch;
            unscaled_lrs = self.base_lrs if epoch==0 else self._last_lr
            for group, lr in zip(self.optimizer.param_groups, unscaled_lrs):
                group['lr'] = lr 

            super().step(epoch) # set unscaled lr, _step_count, last_epoch, _last_lr for new epoch

            # scale the learning rate for next epoch for each parameter group.
            batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
            for group in self.optimizer.param_groups:
                group['lr'] = scale_lr(self.base_batch_size, batch_size, group['lr'], lr_scaling_method)

            if self.verbose:
                print(f"Batch id {self.last_epoch}, unscaled LR: {unscaled_lrs}, scaled LR: {self.get_lr()}")


    #### main loop: double check arguments and returns correctly-instantiated LR scheduler

    if lr_scheduler_class is None:
        assert optimizer is not None, "optimizer must be provided if lr_scheduler_class is not"
    else:
        assert issubclass(lr_scheduler_class, LRScheduler), "lr_scheduler should be a LRScheduler"

    if optimizer is None:
        assert lr_scheduler_class is not None, "lr_scheduler_class must be provided if optimizer is not"
        optimizer = lr_scheduler_kwargs['optimizer']

    return VariableBatchSizeLR(optimizer=optimizer, **lr_scheduler_kwargs)


def get_dataloader_and_lr_scheduler_for_variable_batch_size(
        dataset,
        dataset_metric_values,
        max_metric_value_per_batch,
        base_batch_size,
        sample_ids=None,
        lr_scaling_method="linear",
        min_batch_size=1,
        max_batch_size=None,
        shuffle_metric_values=False,
        order_by_metric_value=False,
        gradient_accumulation_steps=1,
        pipeline_parallelism=False,
        dataloader_rank=0,
        dataloader_num_replicas=1,
        dataloader_num_workers=0,
        dataloader_collate_fn=None,
        dataloader_pin_memory=False,
        optimizer=None,
        lr_scheduler_class=None,
        lr_scheduler_kwargs={'verbose':False},
        verbose=False,
):

        # batch_by_size returns the effective batch size and the sample ids for each microbatch.
        # We will use the sample ids to retrieve the batches from the dataset, 
        # and the effective batch size to retrieve the scaled learning rate for each batch
        # Note: pipelining in DeepSpeed takes the first micro-batch activation shape as reference.
        # So we need to make sure batch size remains contant across all microbatches in a batch.
        microbatch_sample_ids, batch_sizes, batch_metrics = batch_by_size(
            metric_values=dataset_metric_values,
            max_metric_value_per_batch=max_metric_value_per_batch,
            sample_ids=sample_ids,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            shuffle_metric_values=shuffle_metric_values,
            order_by_metric_value=order_by_metric_value,
            dataloader_num_replicas=dataloader_num_replicas,
            gradient_accumulation_steps=gradient_accumulation_steps,
            required_microbatches_of_same_size=pipeline_parallelism,
            verbose=verbose,
        )

        dataloader, deepspeed_io_kwargs = dataloader_for_variable_batch_size(
            dataset=dataset,
            microbatch_sample_ids=microbatch_sample_ids,
            dataloader_rank=dataloader_rank,
            dataloader_num_replicas=dataloader_num_replicas,
            dataloader_collate_fn=dataloader_collate_fn,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
        )
            
        lr_scheduler = lr_scheduler_for_variable_batch_size(
            base_batch_size=base_batch_size,
            batch_sizes=batch_sizes,
            batch_metrics=batch_metrics,
            lr_scaling_method=lr_scaling_method,
            optimizer=optimizer,
            dataloader=dataloader,
            lr_scheduler_class=lr_scheduler_class,
            **lr_scheduler_kwargs)

        return dataloader, lr_scheduler, deepspeed_io_kwargs




########## Main includes few examples on how to use this module ###############

if __name__ == "__main__":

    class TestData(torch.utils.data.Dataset):
        """ A test dataset with sequences of random length, and their sum as the target"""
        def __init__(self, seq_count, min_seq_len=1, max_seq_len=21, seed=0):
            data_random = random.Random(seed) 
            self.seqs = [ torch.ones(data_random.randrange(min_seq_len,max_seq_len)) for _ in range(seq_count) ]
        
        __len__ = lambda self: len(self.seqs)
        __getitem__ = lambda self, idx: [self.seqs[idx], self.seqs[idx].sum()]

        # collate_fn merges sequences, padded to the max length, or trimmed/paded to a value
        @staticmethod
        def collate_fn(batch, max_seq_len=None):
            # if max_seq_len in enforces, trim/pad them to the max_len specified
            if max_seq_len is not None:
                for i, (seq, _) in enumerate(batch):
                    batch[i][0] = torch.nn.ConstantPad1d((0, max_seq_len - seq.shape[0]), 0)(seq)
            seqs, labels = zip(*batch)
            padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
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


    dataloader_rank=int(os.environ.get('RANK',0))
    dataloader_num_replicas=int(os.environ.get('WORLD_SIZE',1))
    device_id=int(os.environ.get('LOCAL_RANK',0))
    device = f"cuda:{device_id}"
    max_seq_len=15
    max_metric_value_per_batch=40
    base_batch_size = 8
    base_lr=1e-3
    gradient_accumulation_steps=base_batch_size//dataloader_num_replicas
    pipeline_parallelism=True
    order_by_metric_value=True #enable for curriculum

    dist.init_process_group(backend='nccl')
    model = TestFeedForward().to(device)
    dataset = TestData(seq_count=300, min_seq_len=5, max_seq_len=max_seq_len)
    model_ddp = DDP(model, device_ids=[device])
    optimizer = torch.optim.Adam(model_ddp.parameters(), lr=1e-3)

    metric_values = [ len(s[0]) for s in dataset] # difficulty = input sequence length
    dataloader, lr_scheduler, deepspeed_io_kwargs = get_dataloader_and_lr_scheduler_for_variable_batch_size(
        dataset=dataset,
        dataset_metric_values=metric_values,
        base_batch_size=base_batch_size,
        max_metric_value_per_batch=max_metric_value_per_batch,
        dataloader_rank=dataloader_rank,
        dataloader_num_replicas=dataloader_num_replicas,
        pipeline_parallelism=pipeline_parallelism,
        lr_scaling_method="linear",
        order_by_metric_value=order_by_metric_value,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,
        dataloader_collate_fn=lambda b : TestData.collate_fn(b, max_seq_len=max_seq_len),
        optimizer=optimizer,
        # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
        # lr_scheduler_kwargs=dict(optimizer=optimizer, step_size=1, gamma=0.1),
    )
 
    # PyTorch example iterating whole dataset in one epoch
    for epoch in range(2):
        for sample_idx, (inputs, labels) in enumerate(dataloader):
            batch_id = sample_idx // gradient_accumulation_steps
            batch_id = sample_idx % gradient_accumulation_steps
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ddp(inputs)
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            if (batch_id+1) % gradient_accumulation_steps == 0:
                if dataloader_rank==0:
                    print(f"rank {dataloader_rank}, batch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}")
                optimizer.step()
                optimizer.zero_grad()  
                lr_scheduler.step()

    dist.destroy_process_group()
            
    # DeepSpeed example
    config = {
        "train_batch_size": base_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": { "type": "Adam", "params": { "lr": base_lr } },
    }

    engine, optimizer, _, lr_scheduler = deepspeed.initialize(config=config,
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    # engine.training_dataloader = dataloader #use this or the deepspeed_io()
    engine.training_dataloader = engine.deepspeed_io(**deepspeed_io_kwargs)

    lr_scheduler.step(0) # reset LR scheduler
    for epoch in range(2):
        for sample_idx, (inputs, labels) in enumerate(dataloader):
            batch_id = sample_idx // gradient_accumulation_steps
            batch_id = sample_idx % gradient_accumulation_steps
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = engine(inputs)
            loss = F.mse_loss(outputs, labels)
            engine.backward(loss)
            if dataloader_rank==0:
                print(f"rank {dataloader_rank}, batch {batch_id},  microbatch {batch_id}, loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}")
            engine.step()

    # Deepspeed example for pipeline parallelism
    if pipeline_parallelism:
        model = PipelineModule(layers=model.to_layers(), num_stages=2)
        engine, optimizer, _, lr_scheduler = deepspeed.initialize(config=config,
            model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        # engine.training_dataloader = dataloader #use this or the deepspeed_io()
        engine.training_dataloader = engine.deepspeed_io(**deepspeed_io_kwargs)
        
        dataloader_it = iter(dataloader) # reset dataloader
        lr_scheduler.step(0) # reset LR scheduler
        for epoch in range(2):
            for batch_id in range(len(dataloader)//gradient_accumulation_steps):
                loss = engine.train_batch(data_iter=dataloader_it)
                if dataloader_rank==0:
                    print(f"rank {dataloader_rank}, batch {batch_id},  loss {loss.item()}, LRs {lr_scheduler.get_lr()}, epoch {epoch}")
