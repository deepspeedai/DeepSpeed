# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.utils import RepeatingLoader
import pickle
import numpy as np
import torch
import pytest
import deepspeed
from deepspeed.accelerator import get_accelerator
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, random_dataset
from deepspeed.runtime.data_pipeline.data_sampling import indexed_dataset
from deepspeed.runtime.data_pipeline.data_sampling.indexed_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder


@pytest.mark.parametrize("dtype", [np.int32, np.uint16])
@pytest.mark.parametrize("skip_warmup", [False, True])
def test_mmap_dataset_pickle_round_trip(tmp_path, monkeypatch, dtype, skip_warmup):
    path = str(tmp_path / "dataset")
    builder = MMapIndexedDatasetBuilder(path + ".bin", dtype=dtype)
    for i in range(3):
        builder.add_item_numpy(np.arange(i + 1, dtype=dtype))
    builder.end_document()
    builder.finalize(path + ".idx")
    dataset = MMapIndexedDataset(path, skip_warmup=skip_warmup)
    warmup_calls = []
    warmup = indexed_dataset._warmup_mmap_file

    def record_warmup(filename):
        warmup_calls.append(filename)
        return warmup(filename)

    monkeypatch.setattr(indexed_dataset, "_warmup_mmap_file", record_warmup)
    restored = pickle.loads(pickle.dumps(dataset))
    assert warmup_calls == ([] if skip_warmup else [path + ".idx", path + ".bin"])

    assert len(restored) == len(dataset)
    assert restored.dtype == dataset.dtype
    np.testing.assert_array_equal(restored.sizes, dataset.sizes)
    np.testing.assert_array_equal(restored.doc_idx, dataset.doc_idx)
    for i in range(len(dataset)):
        np.testing.assert_array_equal(restored[i], dataset[i])

    legacy = MMapIndexedDataset.__new__(MMapIndexedDataset)
    legacy.__setstate__(path)
    np.testing.assert_array_equal(legacy[0], dataset[0])


def test_mmap_dataset_spawn_dataloader(tmp_path):
    path = str(tmp_path / "dataset")
    expected = torch.arange(12).reshape(4, 3)
    builder = MMapIndexedDatasetBuilder(path + ".bin")
    for row in expected:
        builder.add_item(row)
    builder.end_document()
    builder.finalize(path + ".idx")
    dataset = MMapIndexedDataset(path, skip_warmup=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=1, multiprocessing_context="spawn")

    torch.testing.assert_close(torch.cat(list(loader)), expected)


def test_repeating_loader():
    loader = [1, 2, 3]
    loader = RepeatingLoader(loader)

    for idx in range(50):
        assert next(loader) == 1
        assert next(loader) == 2
        assert next(loader) == 3


@pytest.mark.parametrize('train_batch_size, drop_last', [(1, True), (4, True), (1, False), (4, False)])
class TestDataLoaderDropLast(DistributedTest):
    world_size = 1

    def test(self, train_batch_size, drop_last):
        config_dict = {"train_batch_size": train_batch_size, "dataloader_drop_last": drop_last, "steps_per_print": 1}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        # TODO: no way to set DeepSpeedEngine.deepspeed_io params, need to use
        # pin_memory=False for cuda device
        train_dataset = random_dataset(total_samples=50,
                                       hidden_dim=hidden_dim,
                                       device=torch.device('cpu'),
                                       dtype=torch.float32)
        model, _, training_dataloader, _ = deepspeed.initialize(config=config_dict,
                                                                model=model,
                                                                training_data=train_dataset,
                                                                optimizer=optimizer)
        training_dataloader.num_local_io_workers = 0  # We can't do nested mp.pool
        for n, batch in enumerate(training_dataloader):
            x = batch[0].to(get_accelerator().current_device_name())
            y = batch[1].to(get_accelerator().current_device_name())
            loss = model(x, y)
            model.backward(loss)
            model.step()
