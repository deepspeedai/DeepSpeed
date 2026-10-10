# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pickle

import numpy as np
import torch

from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import (MMapIndexedDataset,
                                                                           MMapIndexedDatasetBuilder, data_file_path,
                                                                           index_file_path)


def test_mmap_indexed_dataset_pickle_round_trip(tmpdir):
    prefix = str(tmpdir.join("dataset"))
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.int64)
    builder.add_item(torch.tensor([1, 2, 3]))
    builder.add_item(torch.tensor([4, 5]))
    builder.end_document()
    builder.finalize(index_file_path(prefix))

    dataset = MMapIndexedDataset(prefix, skip_warmup=True)
    restored = pickle.loads(pickle.dumps(dataset))

    assert len(restored) == 2
    assert restored[0].tolist() == [1, 2, 3]
    assert restored[1].tolist() == [4, 5]
