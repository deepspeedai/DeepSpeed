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


@pytest.mark.parametrize("world_size,env,expected", [
    (2, None, 2),
    (1, None, 1),
    (-1, "4", 4),
    (-1, None, None),
    (-1, "", None),
])
def test_known_world_size(monkeypatch, world_size, env, expected):
    # -1 with nothing in the environment means the size is genuinely unknown: it may be carried
    # in the init_method URL. Returning None keeps the caller from skipping the device binding
    # on an assumed size.
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if env is not None:
        monkeypatch.setenv("WORLD_SIZE", env)
    assert ds_comm_torch.known_world_size(world_size) == expected


def assert_device_binding(expect_bound):
    """A device is bound for multi-rank jobs, and not for a single-rank one (#8248)."""
    if get_accelerator().communication_backend_name() != 'nccl':
        pytest.skip("device_id is only bound for the nccl backend")

    deepspeed.init_distributed(dist_backend='nccl', auto_mpi_discovery=False)
    default_pg = torch.distributed.distributed_c10d._get_default_group()
    assert (default_pg.bound_device_id is not None) == expect_bound

    # The eager split new_group() makes when a device is bound is the call that fails in #8248.
    device = get_accelerator().device(int(os.environ["LOCAL_RANK"]))
    default_backend = default_pg._get_backend(device)
    if hasattr(default_backend, "comm_split_count"):
        splits_before = default_backend.comm_split_count()
        torch.distributed.new_group(ranks=list(range(dist.get_world_size())))
        assert (default_backend.comm_split_count() > splits_before) == expect_bound


class TestSingleRankDeviceId(DistributedTest):
    world_size = 1
    init_distributed = False

    def test(self):
        assert_device_binding(expect_bound=False)


class TestMultiRankDeviceId(DistributedTest):
    world_size = 2
    init_distributed = False

    def test(self):
        assert_device_binding(expect_bound=True)
