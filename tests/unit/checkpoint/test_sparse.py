# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed

from unit.common import DistributedTest
from unit.simple_model import *

import pytest


class TestSparseCheckpoint(DistributedTest):
    world_size = 2

    @pytest.mark.parametrize(["to_save_model_has_embedding", "to_save_model_sparse"], [
        [False, False],
        [True, False],
        [True, True],
    ])
    @pytest.mark.parametrize(["destination_has_embedding", "destination_sparse"], [
        [False, False],
        [True, False],
        [True, True],
    ])
    @pytest.mark.parametrize('compile_mode', [True, False])
    def test_non_strict_load_sparse(self, tmpdir, to_save_model_has_embedding, to_save_model_sparse,
                                    destination_has_embedding, destination_sparse, compile_mode):

        class ModelNoEmbedding(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 1)

            def forward(self, x):
                return self.linear(x)

        class ModelEmbedding(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(10, 3)
                self.linear = torch.nn.Linear(3, 1)

            def forward(self, x, offsets):
                return self.linear(self.emb(x, offsets))

        if to_save_model_has_embedding:
            model_to_save = ModelEmbedding()
        else:
            model_to_save = ModelNoEmbedding()
        if destination_has_embedding:
            model_destination = ModelEmbedding()
        else:
            model_destination = ModelNoEmbedding()

        engine_to_save, _, _, _ = deepspeed.initialize(model=model_to_save,
                                                       config={
                                                           "train_batch_size": 2,
                                                           "sparse_gradients": to_save_model_sparse
                                                       })
        engine_destination, _, _, _ = deepspeed.initialize(model=model_destination,
                                                           config={
                                                               "train_batch_size": 2,
                                                               "sparse_gradients": destination_sparse
                                                           })

        if compile_mode:
            engine_to_save.compile()
            engine_destination.compile()

        save_folder = os.path.join(tmpdir, 'saved_checkpoint')
        save_tag = '1'

        engine_to_save.save_checkpoint(save_folder, tag=save_tag)

        is_sparse_destination = isinstance(model_destination, ModelEmbedding) and destination_sparse
        if isinstance(model_destination, ModelEmbedding) and model_destination.emb.sparse:
            assert "emb.weight" in engine_destination.sparse_tensor_module_names
        engine_destination.load_checkpoint(save_folder,
                                           tag=save_tag,
                                           load_module_strict=False,
                                           load_optimizer_states=False,
                                           load_lr_scheduler_states=False,
                                           load_module_only=False)
        if isinstance(model_destination, ModelEmbedding) and isinstance(model_to_save, ModelEmbedding):
            assert engine_destination.sparse_tensor_module_names == engine_to_save.sparse_tensor_module_names
        elif isinstance(model_destination, ModelEmbedding):
            assert not is_sparse_destination or "emb.weight" in engine_destination.sparse_tensor_module_names
        else:
            assert len(engine_destination.sparse_tensor_module_names) == 0
