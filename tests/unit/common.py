# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
import inspect
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.multiprocessing as mp
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from torch.multiprocessing import Process

import pytest
from _pytest.outcomes import Skipped
from _pytest.fixtures import FixtureLookupError, FixtureFunctionMarker

# Worker timeout *after* the first worker has completed.
DEEPSPEED_UNIT_WORKER_TIMEOUT = 120

# Worker timeout for tests that hang
DEEPSPEED_TEST_TIMEOUT = 600


def get_xdist_worker_id():
    xdist_worker = os.environ.get('PYTEST_XDIST_WORKER', None)
    if xdist_worker is not None:
        xdist_worker_id = xdist_worker.replace('gw', '')
        return int(xdist_worker_id)
    return None


def get_master_port():
    master_port = os.environ.get('DS_TEST_PORT', '29503')
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is not None:
        master_port = str(int(master_port) + xdist_worker_id)
    return master_port


def set_accelerator_visible():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it using accelerator specific command instead
        import subprocess
        if get_accelerator().device_name() == 'cuda':
            is_rocm_pytorch = hasattr(torch.version, 'hip') and torch.version.hip is not None
            if is_rocm_pytorch:
                rocm_smi = subprocess.check_output(['rocm-smi', '--showid'])
                gpu_ids = filter(lambda s: 'GPU' in s, rocm_smi.decode('utf-8').strip().split('\n'))
                num_accelerators = len(list(gpu_ids))
            else:
                nvidia_smi = subprocess.check_output(['nvidia-smi', '--list-gpus'])
                num_accelerators = len(nvidia_smi.decode('utf-8').strip().split('\n'))
        elif get_accelerator().device_name() == 'xpu':
            import re
            clinfo = subprocess.check_output(['clinfo'])
            lines = clinfo.decode('utf-8').strip().split('\n')
            num_accelerators = 0
            for line in lines:
                match = re.search('Device Type.*GPU', line)
                if match:
                    num_accelerators += 1
        else:
            assert get_accelerator().device_name() == 'cpu'
            cpu_sockets = int(
                subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
            num_accelerators = cpu_sockets

        cuda_visible = ",".join(map(str, range(num_accelerators)))

    # rotate list based on xdist worker id, example below
    # wid=0 -> ['0', '1', '2', '3']
    # wid=1 -> ['1', '2', '3', '0']
    # wid=2 -> ['2', '3', '0', '1']
    # wid=3 -> ['3', '0', '1', '2']
    dev_id_list = cuda_visible.split(",")
    dev_id_list = dev_id_list[xdist_worker_id:] + dev_id_list[:xdist_worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dev_id_list)


class DistributedExec(ABC):
    """
    Base class for distributed execution of functions/methods. Contains common
    methods needed for DistributedTest and DistributedFixture.
    """
    world_size = 2
    backend = get_accelerator().communication_backend_name()
    init_distributed = True
    set_dist_env = True
    requires_cuda_env = True

    @abstractmethod
    def run(self):
        ...

    def __call__(self, request=None):
        self._fixture_kwargs = self._get_fixture_kwargs(request, self.run)
        world_size = self.world_size
        if self.requires_cuda_env and not get_accelerator().is_available():
            pytest.skip("only supported in accelerator environments.")

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs)
            time.sleep(0.5)

    def _get_fixture_kwargs(self, request, func):
        if not request:
            return {}
        # Grab fixture / parametrize kwargs from pytest request object
        fixture_kwargs = {}
        params = inspect.getfullargspec(func).args
        params.remove("self")
        for p in params:
            try:
                fixture_kwargs[p] = request.getfixturevalue(p)
            except FixtureLookupError:
                pass  # test methods can have kwargs that are not fixtures
        return fixture_kwargs

    def _launch_procs(self, num_procs):
        if get_accelerator().is_available() and get_accelerator().device_count() < num_procs:
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {num_procs} required, {get_accelerator().device_count()} available"
            )
        mp.set_start_method('forkserver', force=True)
        skip_msg = mp.Queue()  # Allows forked processes to share pytest.skip reason
        processes = []
        for local_rank in range(num_procs):
            p = Process(target=self._dist_init, args=(local_rank, num_procs, skip_msg))
            p.start()
            processes.append(p)

        # Now loop and wait for a test to complete. The spin-wait here isn't a big
        # deal because the number of processes will be O(#GPUs) << O(#CPUs).
        any_done = False
        start = time.time()
        while (not any_done) and ((time.time() - start) < DEEPSPEED_TEST_TIMEOUT):
            for p in processes:
                if not p.is_alive():
                    any_done = True
                    break
            time.sleep(.1)  # So we don't hog CPU

        # If we hit the timeout, then presume a test is hanged
        if not any_done:
            for p in processes:
                p.terminate()
            pytest.exit("Test hanged, exiting", returncode=0)

        # Wait for all other processes to complete
        for p in processes:
            p.join(DEEPSPEED_UNIT_WORKER_TIMEOUT)

        failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
        for rank, p in failed:
            # If it still hasn't terminated, kill it because it hung.
            if p.exitcode is None:
                p.terminate()
                pytest.fail(f'Worker {rank} hung.', pytrace=False)
            if p.exitcode < 0:
                pytest.fail(f'Worker {rank} killed by signal {-p.exitcode}', pytrace=False)
            if p.exitcode > 0:
                pytest.fail(f'Worker {rank} exited with code {p.exitcode}', pytrace=False)

        if not skip_msg.empty():
            # This assumed all skip messages are the same, it may be useful to
            # add a check here to assert all exit messages are equal
            pytest.skip(skip_msg.get())

    def _dist_init(self, local_rank, num_procs, skip_msg):
        """Initialize deepspeed.comm and execute the user function. """
        if self.set_dist_env:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = get_master_port()
            os.environ['LOCAL_RANK'] = str(local_rank)
            # NOTE: unit tests don't support multi-node so local_rank == global rank
            os.environ['RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(num_procs)

        # turn off NCCL logging if set
        os.environ.pop('NCCL_DEBUG', None)

        if get_accelerator().is_available():
            set_accelerator_visible()

        if self.init_distributed:
            deepspeed.init_distributed(dist_backend=self.backend)
            dist.barrier()

        if get_accelerator().is_available():
            get_accelerator().set_device(local_rank)

        try:
            self.run(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                skip_msg.put(e.msg)
            else:
                raise e

        if self.init_distributed or dist.is_initialized():
            # make sure all ranks finish at the same time
            dist.barrier()
            # tear down after test completes
            dist.destroy_process_group()


class DistributedFixture(DistributedExec):
    """
    Implementation that extends @pytest.fixture to allow for distributed execution.
    This is primarily meant to be used when a test requires executing two pieces of
    code with different world sizes.

    There are 2 parameters that can be modified:
        - world_size: int = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

    Features:
        - able to call pytest.skip() inside fixture
        - can be reused by multiple tests
        - can accept other fixtures as input

    Limitations:
        - cannot use @pytest.mark.parametrize
        - world_size cannot be modified after definition and only one world_size value is accepted
        - any fixtures used must also be used in the test that uses this fixture (see example below)
        - return values cannot be returned. Passing values to a DistributedTest
          object can be achieved using class_tmpdir and writing to file (see example below)

    Usage:
        - must implement a run(self, ...) method
        - fixture can be used by making the class name input to a test function

    Example:
        @pytest.fixture(params=[10,20])
        def regular_pytest_fixture(request):
            return request.param

        class distributed_fixture_example(DistributedFixture):
            world_size = 4

            def run(self, regular_pytest_fixture, class_tmpdir):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                local_rank = os.environ["LOCAL_RANK"]
                print(f"Rank {local_rank} with value {regular_pytest_fixture}")
                with open(os.path.join(class_tmpdir, f"{local_rank}.txt"), "w") as f:
                    f.write(f"{local_rank},{regular_pytest_fixture}")

        class TestExample(DistributedTest):
            world_size = 1

            def test(self, distributed_fixture_example, regular_pytest_fixture, class_tmpdir):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                for rank in range(4):
                    with open(os.path.join(class_tmpdir, f"{rank}.txt"), "r") as f:
                        assert f.read() == f"{rank},{regular_pytest_fixture}"
    """
    is_dist_fixture = True

    # These values are just placeholders so that pytest recognizes this as a fixture
    _pytestfixturefunction = FixtureFunctionMarker(scope="function", params=None)
    __name__ = ""

    def __init__(self):
        assert isinstance(self.world_size, int), "Only one world size is allowed for distributed fixtures"
        self.__name__ = type(self).__name__
        _pytestfixturefunction = FixtureFunctionMarker(scope="function", params=None, name=self.__name__)


class DistributedTest(DistributedExec):
    """
    Implementation for running pytest with distributed execution.

    There are 2 parameters that can be modified:
        - world_size: Union[int,List[int]] = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

    Features:
        - able to call pytest.skip() inside tests
        - works with pytest fixtures, parametrize, mark, etc.
        - can contain multiple tests (each of which can be parametrized separately)
        - class methods can be fixtures (usable by tests in this class only)
        - world_size can be changed for individual tests using @pytest.mark.world_size(world_size)
        - class_tmpdir is a fixture that can be used to get a tmpdir shared among
          all tests (including DistributedFixture)

    Usage:
        - class name must start with "Test"
        - must implement one or more test*(self, ...) methods

    Example:
        @pytest.fixture(params=[10,20])
        def val1(request):
            return request.param

        @pytest.mark.fast
        @pytest.mark.parametrize("val2", [30,40])
        class TestExample(DistributedTest):
            world_size = 2

            @pytest.fixture(params=[50,60])
            def val3(self, request):
                return request.param

            def test_1(self, val1, val2, str1="hello world"):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                assert all(val1, val2, str1)

            @pytest.mark.world_size(1)
            @pytest.mark.parametrize("val4", [70,80])
            def test_2(self, val1, val2, val3, val4):
                assert int(os.environ["WORLD_SIZE"]) == 1
                assert all(val1, val2, val3, val4)
    """
    is_dist_test = True

    # Temporary directory that is shared among test methods in a class
    @pytest.fixture(autouse=True, scope="class")
    def class_tmpdir(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp(self.__class__.__name__)
        return fn

    def run(self, **fixture_kwargs):
        self._current_test(**fixture_kwargs)

    def __call__(self, request):
        self._current_test = self._get_current_test_func(request)
        self._fixture_kwargs = self._get_fixture_kwargs(request, self._current_test)

        if self.requires_cuda_env and not get_accelerator().is_available():
            pytest.skip("only supported in accelerator environments.")

        # Catch world_size override pytest mark
        for mark in getattr(request.function, "pytestmark", []):
            if mark.name == "world_size":
                world_size = mark.args[0]
                break
        else:
            world_size = self.world_size

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs)
            time.sleep(0.5)

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)


def get_test_path(filename):
    curr_path = Path(__file__).parent
    return str(curr_path.joinpath(filename))
