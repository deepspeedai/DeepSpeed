# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import numpy as np
import pytest
import torch

from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import make_builder, make_dataset


@pytest.mark.parametrize("impl", ["lazy", "cached", "mmap"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64, np.uint32, np.uint64])
def test_numpy_indices_through_subset_dataloader(tmp_path, impl, index_dtype):
    prefix = str(tmp_path / "samples")
    builder = make_builder(prefix + ".bin", impl=impl)
    rows = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
    for row in rows:
        builder.add_item(row)
    builder.finalize(prefix + ".idx")
    dataset = make_dataset(prefix, impl=impl, skip_warmup=True)
    indices = np.array([2, 0], dtype=index_dtype)
    if dataset.supports_prefetch:
        dataset.prefetch([0, 2])
    subset = torch.utils.data.Subset(dataset, indices)
    batch = next(iter(torch.utils.data.DataLoader(subset, batch_size=2)))
    torch.testing.assert_close(batch.long(), torch.stack([rows[2], rows[0]]))
