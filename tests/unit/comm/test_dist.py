# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import importlib
import os
import torch
import deepspeed.comm as dist
import deepspeed

from unit.common import DistributedTest, DistributedFixture, get_master_port
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator

import pytest
from deepspeed.ops.op_builder import FusedAdamBuilder

if not deepspeed.ops.__compatible_ops__[FusedAdamBuilder.NAME]:
    pytest.skip("This op had not been implemented on this system.", allow_module_level=True)


class TestInit(DistributedTest):
    world_size = 3

    def test(self):
        assert dist.is_initialized()
        assert dist.get_world_size() == 3
        assert dist.get_rank() < 3


# Demonstration of pytest's parameterization and fixtures
@pytest.fixture(params=["hello"])
def greeting(request):
    return request.param


@pytest.mark.parametrize("number,color", [(1138, "purple")])
class TestDistArgs(DistributedTest):
    world_size = 2
    """ Classes that use DistributedTest class must define a test* method """

    @pytest.mark.parametrize("shape", ["icosahedron"])
    def test(self, number, color, shape, greeting):
        """Ensure that we can parse args to DistributedTest methods. """
        assert dist.get_world_size() == 2
        assert number == 1138
        assert color == "purple"
        assert shape == "icosahedron"
        assert greeting == "hello"


# Demonstration of distributed tests grouped in single class
@pytest.mark.parametrize("number", [1138])
class TestGroupedDistTest(DistributedTest):
    world_size = 2

    def test_one(self, number):
        assert dist.get_world_size() == 2
        assert number == 1138

    def test_two(self, number, color="purple"):
        assert dist.get_world_size() == 2
        assert number == 1138
        assert color == "purple"


# Demonstration of world_size override
class TestWorldSizeOverrideDistTest(DistributedTest):
    world_size = 2

    def test_world_size_2(self):
        assert dist.get_world_size() == 2

    @pytest.mark.world_size(1)
    def test_world_size_1(self):
        assert dist.get_world_size() == 1


# Demonstration of the DistributedFixture class
@pytest.fixture(params=[2, 4])
def val1(request):
    return request.param


@pytest.fixture(params=[16, 32])
def val2(request):
    return request.param


class distributed_fixture(DistributedFixture):
    world_size = 2

    def run(self, class_tmpdir, val1, val2):
        assert int(os.environ["WORLD_SIZE"]) == self.world_size
        local_rank = os.environ["LOCAL_RANK"]
        file_path = os.path.join(class_tmpdir, f"checkpoint-{local_rank}.pt")
        with open(file_path, "w") as f:
            f.write(f"{local_rank},{val1},{val2}")


class TestDistributedFixture(DistributedTest):
    world_size = 1

    def test(self, distributed_fixture, class_tmpdir, val1, val2):
        for rank in range(2):
            file_path = os.path.join(class_tmpdir, f"checkpoint-{rank}.pt")
            with open(file_path, "r") as f:
                chkpt = f.read()
            assert chkpt == f"{rank},{val1},{val2}"
        assert int(os.environ["WORLD_SIZE"]) == 1


@pytest.mark.parametrize("num_elements", [128, 3])
class TestDistAllReduce(DistributedTest):
    device_count = get_accelerator().device_count()
    if device_count >= 4:
        world_size = [1, 2, 4]
    elif device_count >= 2:
        world_size = [1, 2]
    else:
        world_size = [1]

    def test(self, num_elements):
        x = torch.ones(1, num_elements).to(get_accelerator().device_name()) * (dist.get_rank() + 1)
        sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
        result = torch.ones(1, num_elements).to(get_accelerator().device_name()) * sum_of_ranks
        dist.all_reduce(x)
        assert torch.all(x == result)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_elements", [128, 3])
class TestDistInferenceAllReduce(DistributedTest):
    device_count = get_accelerator().device_count()
    if device_count >= 4:
        world_size = [1, 2, 4]
    elif device_count >= 2:
        world_size = [1, 2]
    else:
        world_size = [1]

    def test(self, dtype, num_elements):
        x = torch.ones(1, num_elements).to(get_accelerator().device_name()) * (dist.get_rank() + 1)
        sum_of_ranks = (dist.get_world_size() * (dist.get_world_size() + 1)) // 2
        result = torch.ones(1, num_elements).to(get_accelerator().device_name()) * sum_of_ranks
        result = result.to(dtype)
        x = x.to(dtype)
        dist.inference_all_reduce(x)
        assert torch.all(x == result)


@pytest.mark.parametrize("dist_init_required", [True, False, None])
class TestDistInit(DistributedTest):
    init_distributed = False

    def test_already_init(self, dist_init_required):
        torch.distributed.init_process_group(get_accelerator().communication_backend_name())
        deepspeed.init_distributed(get_accelerator().communication_backend_name(),
                                   dist_init_required=dist_init_required)

    def test_no_init(self, dist_init_required):
        if dist_init_required or dist_init_required is None:
            deepspeed.init_distributed(get_accelerator().communication_backend_name(),
                                       dist_init_required=dist_init_required)
        else:
            # torch.dist is not done and for some reason the user says they don't want it done
            with pytest.raises(Exception):
                deepspeed.init_distributed(get_accelerator().communication_backend_name(),
                                           dist_init_required=dist_init_required)


class TestDistInitNoEnv(DistributedTest):
    world_size = 1
    init_distributed = False
    set_dist_env = False

    def test(self):
        torch.distributed.init_process_group(backend=get_accelerator().communication_backend_name(),
                                             init_method=f"tcp://127.0.0.1:{get_master_port()}",
                                             world_size=1,
                                             rank=0)
        assert torch.distributed.is_initialized()
        deepspeed.init_distributed(get_accelerator().communication_backend_name(), auto_mpi_discovery=True)


@pytest.mark.parametrize("dist_init_required", [True, False])
class TestDistInitWithModel(DistributedTest):
    init_distributed = False

    def test_already_init(self, dist_init_required):
        torch.distributed.init_process_group(get_accelerator().communication_backend_name())
        model = SimpleModel(4)
        config_dict = {"train_micro_batch_size_per_gpu": 1, "optimizer": {"type": "Adam", "params": {}}}
        engine, *_ = deepspeed.initialize(model=model,
                                          config=config_dict,
                                          model_parameters=model.parameters(),
                                          dist_init_required=dist_init_required)

    def test_no_init(self, dist_init_required):
        model = SimpleModel(4)
        config_dict = {"train_micro_batch_size_per_gpu": 1, "optimizer": {"type": "Adam", "params": {}}}
        if dist_init_required:
            engine, *_ = deepspeed.initialize(model=model,
                                              config=config_dict,
                                              model_parameters=model.parameters(),
                                              dist_init_required=dist_init_required)
        else:
            # torch.dist is not done and for some reason the user says they don't want it done
            with pytest.raises(Exception):
                engine, *_ = deepspeed.initialize(model=model,
                                                  config=config_dict,
                                                  model_parameters=model.parameters(),
                                                  dist_init_required=dist_init_required)


# `deepspeed.comm.torch` is shadowed by the real torch module in the `deepspeed.comm` namespace.
ds_comm_torch = importlib.import_module("deepspeed.comm.torch")

# Spelled out rather than imported, so the end-to-end checks also run against unpatched DeepSpeed.
SET_DEVICE_ID_ENV = "DEEPSPEED_SET_DEVICE_ID"


class FakeAccelerator:
    """Stands in for a real accelerator so the device_id policy can be exercised on any host."""

    def __init__(self, name='cuda', device_count=8):
        self._name = name
        self._device_count = device_count

    def device_name(self, device_index=None):
        if device_index is None:
            return self._name
        return f'{self._name}:{device_index}'

    def device_count(self):
        return self._device_count

    def device(self, device_index=None):
        return torch.device(self._name, device_index)


def resolve_device_id(monkeypatch, world_size, local_rank=0, device_count=8, override=None, name='cuda'):
    """Run the device_id policy against a pretend host, independent of the real accelerator."""
    accelerator = FakeAccelerator(name=name, device_count=device_count)
    monkeypatch.setattr(ds_comm_torch, 'get_accelerator', lambda: accelerator)
    monkeypatch.setenv('LOCAL_RANK', str(local_rank))
    monkeypatch.setenv('WORLD_SIZE', str(world_size))
    monkeypatch.delenv(SET_DEVICE_ID_ENV, raising=False)
    if override is not None:
        monkeypatch.setenv(SET_DEVICE_ID_ENV, override)
    return ds_comm_torch.get_init_process_group_device_id(world_size)


def test_device_id_skipped_for_single_rank(monkeypatch):
    # No peer to connect to, so eager init is pure failure surface. This is the case in #8248.
    assert resolve_device_id(monkeypatch, world_size=1) is None


def test_device_id_set_for_multi_rank(monkeypatch):
    assert resolve_device_id(monkeypatch, world_size=2, local_rank=1) == torch.device('cuda', 1)


@pytest.mark.parametrize("override", ["0", "false", "NO", "off", " 0 "])
def test_device_id_disabled_by_env(monkeypatch, override):
    assert resolve_device_id(monkeypatch, world_size=4, override=override) is None


@pytest.mark.parametrize("override", ["1", "true", "YES", "on"])
def test_device_id_forced_by_env_for_single_rank(monkeypatch, override):
    assert resolve_device_id(monkeypatch, world_size=1, override=override) == torch.device('cuda', 0)


@pytest.mark.parametrize("override", ["fasle", "banana", "2"])
def test_unrecognized_env_value_falls_back_to_the_default(monkeypatch, override):
    # A misspelt "false" must not silently mean "true"; the world_size default still decides.
    assert resolve_device_id(monkeypatch, world_size=1, override=override) is None
    assert resolve_device_id(monkeypatch, world_size=2, override=override) == torch.device('cuda', 0)


def test_device_id_skipped_when_local_rank_is_not_visible(monkeypatch):
    # CUDA_VISIBLE_DEVICES can be narrowed independently of the launcher's rank numbering.
    assert resolve_device_id(monkeypatch, world_size=8, local_rank=3, device_count=1) is None


def test_device_id_skipped_for_non_cuda_accelerator(monkeypatch):
    assert resolve_device_id(monkeypatch, world_size=2, name='xpu') is None


def test_env_var_name_matches_source():
    assert ds_comm_torch.DS_SET_DEVICE_ID == SET_DEVICE_ID_ENV


def test_world_size_read_from_env_when_not_supplied(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '4')
    assert ds_comm_torch.resolve_world_size(-1) == 4


def test_world_size_defaults_to_one_when_unknown(monkeypatch):
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    assert ds_comm_torch.resolve_world_size(-1) == 1


def assert_device_binding(override, expect_bound):
    """Initialize the process group under an override and check what the binding does downstream."""
    if get_accelerator().communication_backend_name() != 'nccl':
        pytest.skip("device_id is only bound for the nccl backend")

    if override is None:
        os.environ.pop(SET_DEVICE_ID_ENV, None)
    else:
        os.environ[SET_DEVICE_ID_ENV] = override

    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)

    default_pg = torch.distributed.distributed_c10d._get_default_group()
    assert (default_pg.bound_device_id is not None) == expect_bound

    # The eager split new_group() makes when a device is bound is the call that fails in #8248.
    device = get_accelerator().device(int(os.environ["LOCAL_RANK"]))
    default_backend = default_pg._get_backend(device)
    if hasattr(default_backend, "comm_split_count"):
        splits_before = default_backend.comm_split_count()
        torch.distributed.new_group(ranks=list(range(dist.get_world_size())))
        splits_after = default_backend.comm_split_count()
        assert (splits_after > splits_before) == expect_bound


@pytest.mark.parametrize("override,expect_bound", [(None, False), ("0", False), ("1", True)])
class TestSingleRankDeviceId(DistributedTest):
    world_size = 1
    init_distributed = False

    def test(self, override, expect_bound):
        assert_device_binding(override, expect_bound)


@pytest.mark.parametrize("override,expect_bound", [(None, True), ("0", False)])
class TestMultiRankDeviceId(DistributedTest):
    world_size = 2
    init_distributed = False

    def test(self, override, expect_bound):
        assert_device_binding(override, expect_bound)
