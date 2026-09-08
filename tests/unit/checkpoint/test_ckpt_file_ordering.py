# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
from types import SimpleNamespace

import torch

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from deepspeed.runtime.state_dict_factory import SDLoaderFactory

# The rank field in a checkpoint file name is padded to two digits, so this is the
# smallest interesting degree: the field overflows and lexicographic order diverges
# from rank order.
MP_DEGREE_OVER_PAD = 128
MP_DEGREE_UNDER_PAD = 8


class PipelineModuleStub:
    """Supplies only the attributes the two checkpoint path methods read off ``self``."""

    ckpt_layer_path = PipelineModule.ckpt_layer_path
    ckpt_layer_path_list = PipelineModule.ckpt_layer_path_list

    def __init__(self, topo, global_rank=0):
        self._local_start = 0
        self.global_rank = global_rank
        self._grid = SimpleNamespace(_topo=topo)


class EngineStub:
    """Supplies only the attributes the two checkpoint name methods read off ``self``."""

    _get_ckpt_name = DeepSpeedEngine._get_ckpt_name
    _get_all_ckpt_names = DeepSpeedEngine._get_all_ckpt_names

    def __init__(self, checkpoint_mp_rank=0):
        self.checkpoint_mp_rank = checkpoint_mp_rank

    def zero_optimization_partition_weights(self):
        return False

    def load_universal_checkpoint(self):
        return False


def write_pipeline_layer_shards(ckpt_dir, mp_degree):
    topo = PipeModelDataParallelTopology(num_pp=1, num_mp=mp_degree, num_dp=1)
    for rank in range(mp_degree):
        torch.save({'rank': rank}, PipelineModuleStub(topo, rank).ckpt_layer_path(ckpt_dir, 0))
    return topo


def test_pipeline_layer_shards_load_by_numeric_rank(tmpdir):
    ckpt_dir = str(tmpdir)
    topo = write_pipeline_layer_shards(ckpt_dir, MP_DEGREE_OVER_PAD)

    ckpt_list = PipelineModuleStub(topo).ckpt_layer_path_list(ckpt_dir, 0)
    assert len(ckpt_list) == MP_DEGREE_OVER_PAD

    sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, version=2.0, checkpoint_engine=None)
    for mp_rank in range(MP_DEGREE_OVER_PAD):
        _, sd, _ = sd_loader.load(MP_DEGREE_OVER_PAD, mp_rank, module_key=None, is_pipe_parallel=True)
        assert sd['rank'] == mp_rank


def test_engine_checkpoint_names_ordered_by_numeric_rank(tmpdir):
    ckpt_dir, tag = str(tmpdir), 'global_step100'
    os.makedirs(os.path.join(ckpt_dir, tag))
    for rank in range(MP_DEGREE_OVER_PAD):
        torch.save({'rank': rank}, EngineStub(rank)._get_ckpt_name(ckpt_dir, tag))

    ckpt_files = EngineStub()._get_all_ckpt_names(ckpt_dir, tag)
    assert len(ckpt_files) == MP_DEGREE_OVER_PAD

    loaded = [torch.load(f, weights_only=True)['rank'] for f in ckpt_files]
    assert loaded == list(range(MP_DEGREE_OVER_PAD))


def test_shard_order_below_pad_width_matches_lexicographic(tmpdir):
    # Every checkpoint written before this change has a rank field of at most two
    # digits, where the two orderings agree, so none of them is reordered.
    ckpt_dir = str(tmpdir)
    topo = write_pipeline_layer_shards(ckpt_dir, MP_DEGREE_UNDER_PAD)

    ckpt_list = PipelineModuleStub(topo).ckpt_layer_path_list(ckpt_dir, 0)
    loaded = [torch.load(p, weights_only=True)['rank'] for p in ckpt_list]
    assert loaded == list(range(MP_DEGREE_UNDER_PAD))
    assert ckpt_list == sorted(ckpt_list)
