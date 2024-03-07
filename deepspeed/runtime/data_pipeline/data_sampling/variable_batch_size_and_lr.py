import random
import torch
from deepspeed.utils import logger
from torch.utils.data import DistributedSampler
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import deepspeed


# see https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/data/data_utils.py#L282
# see how to set new batch size here:
# https://github.com/microsoft/DeepSpeed/issues/2798#issuecomment-1435475061
# engine.set_train_micro_batch_size and set_train_batch_size (only changes grad acc steps) in
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/engine.py
# TODO we need same batch size per GPU per grad step!


def batch_by_size(
    metric_values,
    max_metric_value_per_batch,
    sample_ids=None,
    min_batch_size=1,
    max_batch_size=None,
    shuffle_metric_values=False,
    order_by_metric_value=False,
    batch_size_multiple=1,
    required_microbatches_of_same_size=False,
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
    - `batch_size_multiple`: total batch count should divide the final number of batches, with 
       remaining batches being dropped.
       Useful for data parallelism (where `batch_size_multiple`=`num_data_loaders`) and gradient
       accumulation (where `batch_size_multiple`=`num_data_loaders`*`gradient_accumulation_steps`).
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
        random.shuffle(metrics)
    if order_by_metric_value:
        metrics = sorted(metrics)

    # go through metrics and warn user and filter samples that alone exceed the max batch threshold
    long_ids = [ idx for val, idx in metrics if val>max_metric_value_per_batch ]
    if len(long_ids)>0:
        logger.warning(f"Data indices {long_ids} ignored as metrics exceed {max_metric_value_per_batch}.")
        logger.info(f"Original dataset length: {len(metrics)}. New dataset length: {len(long_ids)}")
        metrics = [ m for m in metrics if m[1] not in long_ids ]

    def is_microbatch_valid(metrics):
        if len(metrics) < min_batch_size: return False # insufficient sample count
        if max_batch_size and len(metrics)>max_batch_size: return False # too many samples
        if sum([m[0] for m in metrics]) > max_metric_value_per_batch: return False # exceeds max
        return True
    
    # go through all samples and pack then in microbatches of metric sums below the threshold
    # `required_microbatches_of_same_size` means all minibatches in a batch must be of equal size
    equal_size_multiple = batch_size_multiple if required_microbatches_of_same_size else 1
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

    # make sure we give the same number of batches to each dataloader by trimming the dataset
    microbatches = microbatches[:len(microbatches) - len(microbatches) % batch_size_multiple]

    #compute the effective batch size for each microbatch.
    effective_batch_sizes, sample_ids = [], []
    for rank in range(0, len(microbatches), batch_size_multiple):
        microbatch = microbatches[rank: rank+batch_size_multiple]
        batch_size = sum([len(mb) for mb in microbatch])
        effective_batch_sizes += [batch_size]*len(microbatch)
        sample_ids += [ [m[1] for m in metrics] for metrics in microbatch]

    # return the sample ids of each microbatch, and their effective batch size 
    assert len(effective_batch_sizes) == len(sample_ids)
    return sample_ids, effective_batch_sizes
        

def scale_lr(effective_batch_size, batch_size, base_lr=1, method="linear"):
    """ given a reference lr and batch_size, compute the new LR for a given batch size """
    if method == "linear":
        # Linear Scaling Rule: "When the minibatch size is multiplied by k, multiply the learning,
        # rate by k" (Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, Goyal et al)
        return base_lr * batch_size / effective_batch_size
    if method == "sqrt":
        # Square Root scaling: "when multiplying the batch size by k, multiply the learning rate
        # by √k, to keep the variance in the gradient expectation constant" 
        # (A. Krizhevsky. One weird trick for parallelizing convolutional neural networks)
        return base_lr * torch.sqrt(batch_size / effective_batch_size)
    raise ValueError("Unknown scaling method: {}".format(method))


def dataloader_for_variable_batch_size(dataset, 
        microbatch_sample_ids, dataloader_rank, dataloader_num_replicas, dataloader_collate_fn,
        dataloader_num_workers=2, dataloader_pin_memory=False, deepspeed_engine=None): 

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
        if deepspeed_engine is None:
            return DataLoader(
                dataset=microbatch_sample_ids,
                sampler=sampler,
                num_workers = dataloader_num_workers,
                collate_fn = collate_fn,
                pin_memory=dataloader_pin_memory,
            )
        else:
            deepspeed_engine.deepspeed_io(dataset,
                                batch_size=1,
                                pin_memory=dataloader_pin_memory,
                                data_sampler=sampler,
                                collate_fn=collate_fn,
                                num_local_io_workers=dataloader_num_workers)
        


def lr_scheduler_for_variable_batch_size(
        effective_batch_size, batch_size_per_epoch_fn, lr_scaling_method='linear',
        lr_scheduler_class=LRScheduler, **lr_scheduler_kwargs):
    """
    returns a class that inherits from `lr_scheduler_class` and provides a scaled
    learning rate for batches of different sizes.

    Arguments:
    - `effective_batch_size`: the batch size that the base_LR refers to;
    - `lr_scaling_method`: method to use to scale LR - see `scale_lr()`;
    - `batch_size_per_epoch_fn`: a function that returns the batch size for a given epoch;
    - `lr_scheduler_class`: the class to inherit from (default: `LRScheduler`). It not provided,
       will use the constant LR `optimizer.lr` as the LR value instead ;
    returns:
    - the class that inherits from `lr_scheduler_class`.
    """
    assert issubclass(lr_scheduler_class, LRScheduler), \
        "lr_scheduler should be a subclass of LRScheduler"

    class VariableBatchSizeLR(lr_scheduler_class):

        def __init__(self, **lr_scheduler_kwargs):
            super().__init__(**lr_scheduler_kwargs)
            self.batch_size_per_epoch_fn = batch_size_per_epoch_fn
            self.effective_batch_size = effective_batch_size
            self.lr_scaling_method = lr_scaling_method
            self.unscaled_lrs = self.get_last_lr()[:] # first epoch LRs, cloned

        def state_dict(self):
            return {'base': super().state_dict(),
                    'effective_batch_size': self.effective_batch_size,
                    'lr_scaling_method': self.lr_scaling_method,
                    'unscaled_lrs': self.unscaled_lrs,
                    }


        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict['base'])
            self.effective_batch_size = state_dict['effective_batch_size']
            self.lr_scaling_method = state_dict['lr_scaling_method']
            self.unscaled_lrs = state_dict['unscaled_lrs']


        def step(self, epoch=None):
            
            # call the base scheduler's step method to get LR for next epoch
            # note: optimizer.step preceeds lr_scheduler.step(), so the stepping workflow is:
            # init: lr_scheduler.step(0) --> set LR for epoch 0
            # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
            # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

            if lr_scheduler_class!=LRScheduler: #use LR scheduler
                
                # reset unscaled LRs (to the original scheduler's one) for the current epoch
                for param_group, lr in zip(self.optimizer.param_groups, self.unscaled_lrs):
                    param_group['lr'] = lr

                super().step(epoch) # set lr, _step_count and last_epoch (for next epoch), _last_lr
                self.unscaled_lrs = self.get_last_lr()[:] # backup next epoch LRs, cloned

            else:
                
                # replicate step(): set LR (constant), _step_count, last_epoch and _last_lr
                for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = lr

                self._step_count += 1
                self.last_epoch = self.last_epoch+1 if epoch is None else epoch
                self._last_lr = [lr]*len(self.optimizer.param_groups)
                
            # scale the learning rate for next epoch for each parameter group
            batch_size = self.batch_size_per_epoch_fn(self.last_epoch)
            lr_multiplier = scale_lr(self.effective_batch_size, batch_size, lr_scaling_method=lr_scaling_method)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_multiplier

    return VariableBatchSizeLR(**lr_scheduler_kwargs)


def get_dataloader_and_lr_scheduler_for_variable_batch_size(
        dataset,
        dataset_metric_values,
        max_metric_value_per_batch,
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
        lr_scheduler_class=None,
        lr_scheduler_kwargs={},
        deepspeed_engine=None,
):
        # pipelining in DeepSpeed takes the first micro-batch activation shape as reference.
        # So we need to make sure batch size remains contant across all microbatches in a batch.
        required_microbatches_of_same_size = pipeline_parallelism
        effective_batch_size = dataloader_num_replicas*gradient_accumulation_steps

        # batch_by_size returns the effective batch size and the sample ids for each microbatch.
        # We will use the sample ids to retrieve the batches from the dataset, and
        # the effective batch size to retrive the scaled learning rate for each batch
        microbatch_sample_ids, microbatch_batch_sizes = batch_by_size(
            metric_values=dataset_metric_values,
            max_metric_value_per_batch=max_metric_value_per_batch,
            sample_ids=sample_ids,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            shuffle_metric_values=shuffle_metric_values,
            order_by_metric_value=order_by_metric_value,
            batch_size_multiple=effective_batch_size,
            required_microbatches_of_same_size=required_microbatches_of_same_size,
        )

        dataloader = dataloader_for_variable_batch_size(
            dataset=dataset,
            microbatch_sample_ids=microbatch_sample_ids,
            dataloader_rank=dataloader_rank,
            dataloader_num_replicas=dataloader_num_replicas,
            dataloader_collate_fn=dataloader_collate_fn,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            deepspeed_engine=deepspeed_engine,
        )
            
        if lr_scheduler_class is None:
            return dataloader

        lr_scheduler = lr_scheduler_for_variable_batch_size(
            effective_batch_size=effective_batch_size,
            batch_size_per_epoch_fn=lambda epoch: microbatch_batch_sizes[epoch],
            lr_scaling_method=lr_scaling_method,
            lr_scheduler_class=lr_scheduler_class, **lr_scheduler_kwargs)

        return dataloader, lr_scheduler


if __name__ == "__main__":

    # A small example/test on how to use this module

    from torch.utils.data import Dataset
    class TestData(Dataset):
        """ A test dataset with sequences of random length, and their sum as the target"""
        def __init__(self, seq_count, min_seq_len=1, max_seq_len=21):
            self.seqs = [ torch.ones(random.randrange(min_seq_len,max_seq_len)) for _ in range(seq_count) ]
        
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


    import torch.nn as nn
    import torch.nn.functional as F 
    class TestFeedForward(nn.Module):

        def __init__(self):
            super(TestFeedForward, self).__init__()
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(max_seq_len, 128)
            self.fc2 = nn.Linear(128, 128)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return x.sum(dim=1)


    max_seq_len=20
    dataset = TestData(seq_count=30, min_seq_len=5, max_seq_len=max_seq_len)
    max_metric_value_per_batch=50
    dataloader_num_workers=2
    gradient_accumulation_steps=2
    effective_batch_size=dataloader_num_workers*gradient_accumulation_steps
    base_lr=1,
    metric_values = [ len(s) for s in dataset]
    gradient_accumulation_steps=2

    model = TestFeedForward()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    criterion = torch.nn.MSELoss() 
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    dataloader = get_dataloader_and_lr_scheduler_for_variable_batch_size(
            dataset=dataset,
            dataset_metric_values=metric_values,
            max_metric_value_per_batch=max_metric_value_per_batch,
            dataloader_rank=0,
            dataloader_num_replicas=1,
            sample_ids=None,
            pipeline_parallelism=False,
            lr_scaling_method="linear",
            min_batch_size=1,
            max_batch_size=None,
            shuffle_metric_values=False,
            order_by_metric_value=False,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=0,
            dataloader_collate_fn=lambda b : TestData.collate_fn(b, max_seq_len=max_seq_len),
            dataloader_pin_memory=False,
            lr_scheduler_class=None,
            lr_scheduler_kwargs={},
    )
 
    # test with PyTorch
    for epoch in range(2):
        with torch.set_grad_enabled(True):
            for minibatch_id in range(len(dataloader)//gradient_accumulation_steps):
                for microbatch_id in range(gradient_accumulation_steps):
                    inputs, label = next(iter(dataloader))
                    outputs = model(inputs)
                    loss = criterion(outputs, label)
                    loss.backward()
                    print(f"Epoch {epoch}, minibatch {minibatch_id}, microbatch {microbatch_id}, batch size {len(inputs)}, loss {loss.item()}, LRs {lr_scheduler.get_last_lr()}")
                optimizer.step()
                optimizer.zero_grad()   

    # Test with DeepSpeed
    engine, optimizer, _, _ = deepspeed.initialize (
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
    engine.training_dataloader = dataloader
    # engine.training_dataloader = engine.deepspeed_io()

    for epoch in range(2):
        for minibatch_id in range(len(dataloader)//gradient_accumulation_steps):
            inputs, label = next(iter(dataloader))
            loss = engine(inputs)
            engine.backward(loss)
            engine.step()
            print(f"Epoch {epoch}, minibatch {minibatch_id}, microbatch {microbatch_id}, batch size {len(inputs)}, loss {loss.item()}, LRs {lr_scheduler.get_last_lr()}")

            