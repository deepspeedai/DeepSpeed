# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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


class FakeP2PWork:
    """Stands in for a torch.distributed Work so the staging wrapper can be tested single-process."""

    def __init__(self, fill=None):
        self.fill = fill

    def wait(self):
        if self.fill is not None:
            self.fill()
        return True


@pytest.mark.skipif(get_accelerator().device_name() != 'mps', reason="covers the MPS CPU-staging path")
class TestMpsStagedP2P:

    def test_irecv_defers_copy_back_to_wait(self, monkeypatch):
        from deepspeed.comm.torch import TorchBackend, StagedWork
        captured = {}

        def fake_irecv(tensor, src=None, group=None, tag=0):
            captured['staged'] = tensor
            return FakeP2PWork(fill=lambda: tensor.copy_(torch.arange(16, dtype=torch.float32)))

        monkeypatch.setattr(torch.distributed, 'irecv', fake_irecv)
        backend = TorchBackend.__new__(TorchBackend)
        received = torch.zeros(16, dtype=torch.float32, device='mps')

        handle = backend.irecv(received, src=0)

        # gloo must see a CPU tensor, and the MPS tensor must stay untouched until wait().
        assert isinstance(handle, StagedWork)
        assert captured['staged'].device.type == 'cpu'
        assert received.abs().sum().item() == 0
        assert handle.wait() is True
        assert torch.equal(received.cpu(), torch.arange(16, dtype=torch.float32))

    def test_isend_stages_payload_on_cpu(self, monkeypatch):
        from deepspeed.comm.torch import TorchBackend, StagedWork
        captured = {}

        def fake_isend(tensor, dst=None, group=None, tag=0):
            captured['staged'] = tensor
            return FakeP2PWork()

        monkeypatch.setattr(torch.distributed, 'isend', fake_isend)
        backend = TorchBackend.__new__(TorchBackend)
        payload = torch.arange(16, dtype=torch.float32, device='mps')

        handle = backend.isend(payload, dst=0)

        assert isinstance(handle, StagedWork)
        assert captured['staged'].device.type == 'cpu'
        assert torch.equal(captured['staged'], torch.arange(16, dtype=torch.float32))
        assert handle.wait() is True


class TestDistIsendIrecv(DistributedTest):
    world_size = 2

    def _launch_procs(self, num_procs, init_method):
        # Two gloo ranks do not need two devices, but the base class gates process count on
        # device_count(), which reports sockets on CPU and would skip this test on CI. Bypass
        # the gate there and pin gloo so oneCCL bindings are not silently picked up.
        if get_accelerator().device_name() != 'cpu':
            return super()._launch_procs(num_procs, init_method)
        self.backend = 'gloo'
        torch.multiprocessing.set_start_method('forkserver', force=True)
        self._launch_daemonic_procs(num_procs, init_method)

    def test(self):
        rank = dist.get_rank()
        device = get_accelerator().device_name()
        length = 4096
        if rank == 0:
            payload = torch.arange(length, dtype=torch.float32).to(device)
            handle = dist.isend(payload, dst=1)
        else:
            received = torch.zeros(length, dtype=torch.float32).to(device)
            handle = dist.irecv(received, src=0)
        # isend/irecv are asynchronous by contract: they must hand back a waitable handle,
        # and the received buffer is only valid after wait() completes.
        assert hasattr(handle, 'wait')
        handle.wait()
        if rank == 1:
            expected = torch.arange(length, dtype=torch.float32).to(device)
            assert torch.equal(received, expected)


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
