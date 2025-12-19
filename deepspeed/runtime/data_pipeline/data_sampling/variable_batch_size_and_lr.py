# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# support/questions/maintenance: github user @brunomaga or @deepspeedai/deepspeed

import random
import os
import math
import numpy as np
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from deepspeed.utils import logger
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.data_pipeline.constants import *
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import MMapIndexedDataset
from deepspeed.runtime.data_pipeline.data_sampling.data_analyzer import DistributedDataAnalyzer
import pathlib


def batch_by_seqlens(
    seqlens,
    max_tokens,
    sequence_ids_per_mb=None,
    min_batch_size=1,
    max_batch_size=None,
    sequence_picking_order="dataloader",
    effective_batch_size=1,
    required_microbatches_of_same_size=False,
    verbose=False,
    seed=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain sequences of different lengths.
    Similar to "Attention is all you need", Section 5.1:
    "sequence pairs were batched together by approximate sequence length. Each training batch
    contained a set of sequence pairs containing approximately X source tokens and X target tokens"

    Arguments:
    - `seqlens`: a list of difficulties (metric values) for every sample in the dataset;
    - `max_tokens`: maximum cap in total difficulty in a batch;
    - `min_batch_size`: smallest allowed size of a batch;
    - `min_batch_size`: largest allowed size of a batch;
    - `sequence_picking_order`: order in which to process samples: "dataloader" (default), "random" or "seqlen" (ascending)
    - `effective_batch_size`: effective batch size;
    - `required_microbatches_of_same_size`: enable if each mini-batch (in a total of `batch_size_multiple`
       micro-batches per batch), should have all micro-batches with the same batch size ie the same
       number of sequences.
    - `verbose`: print debug information;
    - `seed`: random seed for reproducibility;

    Returns:
    - `microbatch_ids`: list of tuple of batch id and samples ids per microbatch
    - `batch_sizes`: the effective batch size of each batch, used for to compute the scaled LR
    - `batch_max_seqlens`: the max seqlen across all microbatches in a batch
    """

    assert sequence_picking_order in ["random", "seqlen", "dataloader"]
    if sequence_ids_per_mb is None:
        metrics = list(zip(seqlens, range(len(seqlens))))  # use all samples
    else:
        metrics = list(zip(np.array(seqlens)[sequence_ids_per_mb], sequence_ids_per_mb))

    if sequence_picking_order == 'random':
        metric_random = random.Random(seed)
        metric_random.shuffle(metrics)
    if sequence_picking_order == 'seqlen':
        metrics = sorted(metrics)

    # go through metrics, warn user, and filter samples that alone exceed the max batch threshold
    long_ids = [idx for val, idx in metrics if val > max_tokens]
    if len(long_ids) > 0:
        logger.warning(f"Data indices {long_ids} ignored as metrics exceed {max_tokens}.")
        logger.info(f"Original dataset length: {len(metrics)}. New dataset length: {len(long_ids)}")
        metrics = [m for m in metrics if m[1] not in long_ids]

    def is_microbatch_valid(metrics):
        if min_batch_size and len(metrics) < min_batch_size: return False  # insufficient sample count
        if max_batch_size and len(metrics) > max_batch_size: return False  # too many samples
        if sum([m[0] for m in metrics]) > max_tokens: return False  # exceeds max
        return True

    # go through all samples and pack then in microbatches of metric sums below the threshold
    # `required_microbatches_of_same_size` means all minibatches in a batch must be of equal size
    equal_size_multiple = effective_batch_size if required_microbatches_of_same_size else 1
    microbatches = []
    batch_init = 0
    while batch_init < len(metrics):

        # we iterate over possible effective batch sizes (groups of microbatches of same size)
        valid_batch_end = batch_init
        for batch_end in range(batch_init + equal_size_multiple, len(metrics), equal_size_multiple):

            # attempt effective batch
            batch = metrics[batch_init:batch_end]

            # pick interleaved samples for each microbatch to help with load balancing
            # (in the ordered use case), and to replicate what the distributed sampler does.
            mbs = [batch[b::equal_size_multiple] for b in range(equal_size_multiple)]

            # if they are all valid micro-batches, keep them until you find longer mbatches, if any
            is_batch_valid = all([is_microbatch_valid(mb) for mb in mbs])
            if is_batch_valid:
                valid_batch_end = batch_end

        if batch_init == valid_batch_end: break  # last batch is not valid (size zero), so we are done
        batch = metrics[batch_init:valid_batch_end]
        mbs = [batch[b::equal_size_multiple] for b in range(equal_size_multiple)]
        batch_init += sum([len(l) for l in mbs])
        microbatches += mbs

    # make sure we give the same number of (micro-)batches to each dataloader by trimming the dataset
    assert len(microbatches) >= effective_batch_size, "not enough datapoints to create a single sample per dataloader"
    microbatches = microbatches[:len(microbatches) - len(microbatches) % effective_batch_size]

    #compute the effective batch size for each microbatch.
    batch_sizes, batch_max_seqlens, microbatch_ids = [], [], []
    for rank in range(0, len(microbatches), effective_batch_size):
        batch_id = rank // effective_batch_size
        mbs = microbatches[rank:rank + effective_batch_size]
        # compute the number of samples (not tokens) in this batch (not microbatch)
        n_sequences = sum([len(mb) for mb in mbs])
        # compute the longest sequence (as number of tokens) in this batch (not microbatch)
        sequence_ids_per_mb = [[m[1] for m in metrics] for metrics in mbs]
        sequence_lens_per_mb = [[m[0] for m in metrics] for metrics in mbs]
        batch_max_seqlen = max([max(seqlens) for seqlens in sequence_lens_per_mb])
        batch_and_mb_ids = zip([batch_id] * effective_batch_size, sequence_ids_per_mb)
        batch_sizes.append(n_sequences)
        batch_max_seqlens.append(batch_max_seqlen)
        microbatch_ids += batch_and_mb_ids
        if verbose:
            n_tokens_per_mb = [sum([m[0] for m in mb]) for mb in mbs]
            n_sequences_per_mb = [len(mb) for mb in mbs]
            assert all([n <= max_tokens for n in n_tokens_per_mb]), "size of microbatch exceeds max tokens"
            logger.info(
                f"Batch id {batch_id} contains in total {len(mbs)} microbatches or {n_sequences} sequences. "\
                f"n_sequences per microbatch {n_sequences_per_mb}. "\
                f"n_tokens per microbatch {n_tokens_per_mb}. "\
                f"sequence ids per microbatch: {sequence_ids_per_mb}. "\
                f"sequence lengths per microbatch: {sequence_lens_per_mb}.")

    # return the sample ids of each microbatch, and the batch sizes
    assert len(batch_sizes) == len(microbatch_ids) // effective_batch_size
    return microbatch_ids, batch_sizes, batch_max_seqlens


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
        return base_lr * math.sqrt(batch_size / base_batch_size)
    elif method == None or method.upper() == "NONE":
        return base_lr
    raise ValueError("Unknown scaling method: {}".format(method))


def dataloader_for_variable_batch_size(
    dataset,
    microbatch_ids,
    batch_max_seqlens,
    dataloader_rank=0,
    dataloader_batch_size=1,
    dataloader_num_replicas=1,
    dataloader_collate_fn=None,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    required_microbatches_of_same_seqlen=False,
    sample_padding_fn=None,
):

    # equidistantly distribute the microbatches across the replicas in an interleaved fashion.
    sampler = DistributedSampler(
        dataset=microbatch_ids,
        num_replicas=dataloader_num_replicas,
        rank=dataloader_rank,
        shuffle=False,
        drop_last=False,
    )

    # collate function wraps user-defined collate function to the variable batch data
    def collate_fn_wrapper(list_microbatch_ids):
        # each batch is a list of sample ids that fill up to the max tokens per batch
        # we return the collated batch of all dataset samples of all input batches.
        batch = []
        for batch_id, microbatch_ids in list_microbatch_ids:
            batch_data = [dataset[idx] for idx in microbatch_ids]
            if required_microbatches_of_same_seqlen:
                assert sample_padding_fn is not None, \
                    "padding dataloader_padding_fn must be provided if required_microbatches_of_same_seqlen is True"
                max_seqlen = batch_max_seqlens[batch_id]
                assert all([len(sample) <= max_seqlen for sample in batch_data]), \
                    "some samples are longer than the computed max seqlen for the batch those samples belong to"
                batch_data = [sample_padding_fn(sample, max_seqlen) for sample in batch_data]
            batch += batch_data
        return dataloader_collate_fn(batch) if dataloader_collate_fn else batch

    dataloader = DataLoader(
        dataset=microbatch_ids,
        batch_size=dataloader_batch_size,
        sampler=sampler,
        num_workers=dataloader_num_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=dataloader_pin_memory,
    )

    deepspeed_io_kwargs = dict(
        dataset=microbatch_ids,
        batch_size=dataloader_batch_size,
        pin_memory=dataloader_pin_memory,
        data_sampler=sampler,
        collate_fn=collate_fn_wrapper,
        num_local_io_workers=dataloader_num_workers,
    )

    return dataloader, deepspeed_io_kwargs


class VariableBatchSizeLR(LRScheduler):
    """ an LR scheduler that scales the LR of a given scheduler's LR """

    @property
    def optimizer(self):
        return self.base_lr_scheduler.optimizer

    def __init__(self,
                 lr_scheduler,
                 base_batch_size,
                 batch_sizes,
                 dataloader,
                 lr_scaling_method="linear",
                 last_epoch=-1,
                 verbose=False):
        self.batch_sizes = batch_sizes
        self.base_batch_size = base_batch_size
        self.lr_scaling_method = lr_scaling_method
        self.dataloader = dataloader
        self.base_lr_scheduler = lr_scheduler
        # the following exist in LRScheduler but not in DeepSpeed's LRScheduler so we redefine them here
        self.base_lrs = self.base_lr_scheduler.get_lr()
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.step(0)  # scale LR for first sample in the dataloader

    def state_dict(self):
        return {
            'base_lr_scheduler': self.base_lr_scheduler.state_dict()
        } | {
            'base_batch_size': self.base_batch_size,
            'lr_scaling_method': self.lr_scaling_method,
            'batch_sizes': self.batch_sizes,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            'verbose': self.verbose,
        }

    def load_state_dict(self, state_dict):
        self.base_lr_scheduler.load_state_dict(state_dict['base_lr_scheduler'])
        self.base_batch_size = state_dict['base_batch_size']
        self.lr_scaling_method = state_dict['lr_scaling_method']
        self.batch_sizes = state_dict['batch_sizes']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.verbose = state_dict['verbose']

    def get_last_lr(self):
        return self.base_lr_scheduler._last_lr

    def get_lr(self):
        return [group['lr'] for group in self.base_lr_scheduler.optimizer.param_groups]

    def step(self, epoch=None):
        # call the base scheduler's step method to get LR for next epoch
        # Note: optimizer.step precedes lr_scheduler.step(), so the stepping workflow is:
        # init: lr_scheduler.step(0) --> set LR for epoch 0
        # epoch 0: optimizer.step(); lr_scheduler.step(1) --> set LR for epoch 1
        # epoch 1: optimizer.step(); lr_scheduler.step(2) --> set LR for epoch 2

        # reset unscaled LRs (to the original scheduler's one) to be able to step the base LR scheduler
        # Note: epoch==0: reset LR scheduler; epoch==None: scale LR for next epoch;
        unscaled_lrs = self.base_lrs if epoch == 0 else self.get_last_lr()
        for group, lr in zip(self.base_lr_scheduler.optimizer.param_groups, unscaled_lrs):
            group['lr'] = lr

        self.base_lr_scheduler.step(epoch)  # set unscaled lr, _step_count, last_epoch, _last_lr for new epoch

        # scale the learning rate for the the next iteration for each parameter group.
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        # batch sizes are precomputed and stored in batch_sizes se we loop around to get the next one
        batch_size = self.batch_sizes[self.last_epoch % len(self.batch_sizes)]
        for group in self.base_lr_scheduler.optimizer.param_groups:
            group['lr'] = scale_lr(self.base_batch_size, batch_size, group['lr'], self.lr_scaling_method)

        if self.verbose:
            logger.info(
                f"Next batch id {self.last_epoch}. "\
                f"Reference batch_size {self.base_batch_size} and lr {unscaled_lrs}. "\
                f"Scaled batch_size {batch_size} and lr {self.get_lr()}.")


def lr_scheduler_for_variable_batch_size(base_batch_size,
                                         batch_sizes,
                                         dataloader,
                                         lr_scheduler_or_optimizer,
                                         lr_scaling_method='linear',
                                         verbose=False):
    """
    returns a class that provides an LR scheduler that scales the learning rate at every
    iteration taking into account the batch size of that iteration.
    If learning rate is constant, ie no LR scheduler, then the base LR will be taken from the
    constant LR values in the optimizer param groups. Otherwise from the scheduler's LR.

    Arguments:
    - `base_batch_size`: the batch size that the base LR in the optimizer or scheduler refers to;
    - `lr_scaling_method`: method to use to scale LR - see `scale_lr()`;
    - `lr_scheduler_or_optimizer`: one instance of `LRScheduler` or `Optimizer` to be used as base;
    - `batch_sizes`: the effective batch size of each batch in the dataloader;

    Returns the new LRScheduler
    """

    class StubLRScheduler(LRScheduler):
        """ a stub LR scheduler that does not change the LR, keeps it constant """

        def get_lr(self) -> float:
            return self.base_lrs

    if isinstance(lr_scheduler_or_optimizer, Optimizer):
        lr_scheduler = StubLRScheduler(lr_scheduler_or_optimizer)
    elif hasattr(lr_scheduler_or_optimizer, 'optimizer'):  #LRScheduler or DeepSpeed 'object' schedulers
        assert isinstance(lr_scheduler_or_optimizer.optimizer, Optimizer)
        lr_scheduler = lr_scheduler_or_optimizer
    else:
        raise ValueError("Unknown type for lr_scheduler_or_optimizer: {}".format(type(lr_scheduler_or_optimizer)))

    return VariableBatchSizeLR(lr_scheduler=lr_scheduler,
                               base_batch_size=base_batch_size,
                               batch_sizes=batch_sizes,
                               dataloader=dataloader,
                               lr_scaling_method=lr_scaling_method,
                               verbose=verbose)


def get_dataloader_and_lr_scheduler_for_variable_batch_size_deepspeed(dataset,
                                                                      engine,
                                                                      dataset_seqlens=None,
                                                                      dataset_filter_ids=None,
                                                                      dataloader_collate_fn=None,
                                                                      sample_padding_fn=None,
                                                                      batch_seqlens_fn=None):
    """
    a simplified call to get_dataloader_and_lr_scheduler_for_variable_batch_size for the deepspeed runtime.
    Needs the seqlens of every sample. It will try three alternatives:
    - if `dataset_seqlens` is provided by user, use that.
    - otherwise, looks for the seqlen metric path (in the connfig) that contains the output of the Data Analyzer
    - otherwise, use the user-provided function `batch_seqlens_fn` and call Data Analyzer to output seqlen metric
    See `batch_by_seqlens()` for arguments and more documentation.
    """
    data_efficiency_config = engine._config.data_efficiency_config
    data_sampling_config = data_efficiency_config[DATA_SAMPLING]
    batching_config = data_sampling_config[DYNAMIC_BATCHING]
    assert batching_config[DYNAMIC_BATCHING_ENABLED], "Dynamic batching is not enabled in the config"

    if dataset_seqlens is None:
        # In seqlen provided by user, look for the seqlen metric that was output by the Data Analyzer
        # (see the main in deepspeed/runtime/data_pipeline/data_sampling/data_analyzer.py for an example)
        metrics_path = batching_config[DYNAMIC_BATCHING_METRICS_PATH]
        sample_to_seqlen_path = os.path.join(metrics_path, "seqlen/seqlen_sample_to_metric")
        if not (os.path.exists(f"{sample_to_seqlen_path}.bin") and os.path.exists(f"{sample_to_seqlen_path}.idx")):
            # if the metric files are not found, we run the DataAnalyzer to write the metric files
            msg = f"Cannot find metric files for sequence length in {sample_to_seqlen_path}.idx or *.bin."
            msg += " We will run data analyzer to generated them..."
            logger.warning(msg)

            if batch_seqlens_fn is None:
                raise ValueError("sample_seqlen_fn must be provided if dataset_seqlens is not provided")

            DistributedDataAnalyzer(
                dataset=dataset,
                metric_functions=[batch_seqlens_fn],
                collate_fn=dataloader_collate_fn,
                batch_size=2**10,  # batch size for map-reduce, not training
                num_workers=engine.world_size,
                worker_id=engine.global_rank,
                save_path=pathlib.Path(metrics_path),
                metric_types=['single_value_per_sample'],
                metric_names=["seqlen"],
                device=engine.device,
            ).run_map_reduce()

        dataset_seqlens = MMapIndexedDataset(sample_to_seqlen_path, skip_warmup=True)
        assert len(dataset_seqlens) == len(dataset), \
            "Seqlens size does not match the input dataset size. If you changed the dataset, delete the metrics_path folder."

        # TODO we are copying all seqlens into memory, we should adapt the code to use an iterative streamer
        # and use the other files output by DataAnalyzer that returns an ordered dictionary of seqlen to sample ids
        dataset_seqlens = np.array(list(dataset_seqlens), dtype=np.int64).flatten()  # from Nx1 to N

    dataloader, lr_scheduler, deepspeed_io_kwargs = get_dataloader_and_lr_scheduler_for_variable_batch_size(
        dataset=dataset,
        dataset_filter_ids=dataset_filter_ids,
        dataset_seqlens=dataset_seqlens,
        effective_batch_size=engine.train_batch_size(),
        max_tokens=batching_config[DYNAMIC_BATCHING_MAX_TOKENS],
        lr_scaling_method=batching_config[DYNAMIC_BATCHING_LR_SCALING_METHOD],
        sequence_picking_order=batching_config[DYNAMIC_BATCHING_SEQUENCE_PICKING_ORDER],
        min_batch_size=batching_config[DYNAMIC_BATCHING_MIN_BATCH_SIZE],
        max_batch_size=batching_config[DYNAMIC_BATCHING_MAX_BATCH_SIZE],
        dataloader_batch_size=engine.train_micro_batch_size_per_gpu(),
        dataloader_rank=engine.data_parallel_group.rank(),
        dataloader_num_replicas=engine.data_parallel_group.size(),
        dataloader_num_workers=data_sampling_config[DATA_SAMPLING_NUM_WORKERS],
        dataloader_collate_fn=dataloader_collate_fn,
        dataloader_pin_memory=data_sampling_config[DATA_SAMPLING_PIN_MEMORY],
        sample_padding_fn=sample_padding_fn,
        lr_scheduler_or_optimizer=engine.lr_scheduler or engine.optimizer,
        required_microbatches_of_same_size=isinstance(engine, PipelineEngine),
        required_microbatches_of_same_seqlen=isinstance(engine, PipelineEngine),
        verbose=batching_config[DYNAMIC_BATCHING_VERBOSE],
        seed=data_efficiency_config[DATA_EFFICIENCY_SEED],
    )
    return dataloader, lr_scheduler, deepspeed_io_kwargs


def get_dataloader_and_lr_scheduler_for_variable_batch_size(
    dataset,
    dataset_seqlens,
    max_tokens,
    effective_batch_size,
    dataset_filter_ids=None,
    lr_scaling_method="linear",
    min_batch_size=1,
    max_batch_size=None,
    sequence_picking_order="dataloader",
    dataloader_batch_size=1,
    dataloader_rank=0,
    dataloader_num_replicas=1,
    dataloader_num_workers=0,
    dataloader_collate_fn=None,
    dataloader_pin_memory=False,
    lr_scheduler_or_optimizer=None,
    required_microbatches_of_same_size=False,
    required_microbatches_of_same_seqlen=False,
    sample_padding_fn=None,
    verbose=False,
    seed=None,
):
    """ returns a dataloader and LR scheduler for the variable batch size. see `batch_by_seqlens()` for details. """

    # effective_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of dataloaders
    microbatch_ids, batch_sizes, batch_max_seqlens = batch_by_seqlens(
        seqlens=dataset_seqlens,
        max_tokens=max_tokens,
        sequence_ids_per_mb=dataset_filter_ids,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        sequence_picking_order=sequence_picking_order,
        effective_batch_size=effective_batch_size,
        required_microbatches_of_same_size=required_microbatches_of_same_size,
        verbose=verbose,
        seed=seed,
    )

    dataloader, deepspeed_io_kwargs = dataloader_for_variable_batch_size(
        dataset=dataset,
        microbatch_ids=microbatch_ids,
        batch_max_seqlens=batch_max_seqlens,
        dataloader_rank=dataloader_rank,
        dataloader_num_replicas=dataloader_num_replicas,
        dataloader_batch_size=dataloader_batch_size,
        dataloader_collate_fn=dataloader_collate_fn,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        required_microbatches_of_same_seqlen=required_microbatches_of_same_seqlen,
        sample_padding_fn=sample_padding_fn,
    )

    lr_scheduler = lr_scheduler_for_variable_batch_size(base_batch_size=effective_batch_size,
                                                        batch_sizes=batch_sizes,
                                                        lr_scaling_method=lr_scaling_method,
                                                        lr_scheduler_or_optimizer=lr_scheduler_or_optimizer,
                                                        dataloader=dataloader,
                                                        verbose=verbose)

    return dataloader, lr_scheduler, deepspeed_io_kwargs
