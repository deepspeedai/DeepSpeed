# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import time
import inspect
import socket
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
import random
import numpy as np
from typing import Callable, Any

import torch
import torch.multiprocessing as mp
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist

from .util import torch_assert_close

import pytest
from _pytest.outcomes import Skipped
from _pytest.fixtures import FixtureLookupError, FixtureFunctionMarker

# Worker timeout for tests that hang
DEEPSPEED_TEST_TIMEOUT = int(os.environ.get('DS_UNITTEST_TIMEOUT', '600'))


def is_rocm_pytorch():
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


def get_xdist_worker_id():
    xdist_worker = os.environ.get('PYTEST_XDIST_WORKER', None)
    if xdist_worker is not None:
        xdist_worker_id = xdist_worker.replace('gw', '')
        return int(xdist_worker_id)
    return None


def get_master_port(base_port=29500, port_range_size=1000):
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is not None:
        # Make xdist workers use different port ranges to avoid race conditions
        base_port += port_range_size * xdist_worker_id

    # Select first open port in range
    port = base_port
    max_port = base_port + port_range_size
    sock = socket.socket()
    while port < max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return str(port)
        except OSError:
            port += 1
    raise IOError('no free ports')


def _get_cpu_socket_count():
    import shlex
    p1 = subprocess.Popen(shlex.split("cat /proc/cpuinfo"), stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "physical id"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    p3 = subprocess.Popen(shlex.split("sort -u"), stdin=p2.stdout, stdout=subprocess.PIPE)
    p2.stdout.close()
    p4 = subprocess.Popen(shlex.split("wc -l"), stdin=p3.stdout, stdout=subprocess.PIPE)
    p3.stdout.close()
    r = int(p4.communicate()[0])
    p4.stdout.close()
    return r


def set_accelerator_visible():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it using accelerator specific command instead
        if get_accelerator().device_name() == 'cuda':
            if is_rocm_pytorch():
                rocm_smi = subprocess.check_output(['rocm-smi', '--showid'])
                gpu_ids = filter(lambda s: 'GPU' in s, rocm_smi.decode('utf-8').strip().split('\n'))
                num_accelerators = len(list(gpu_ids))
            else:
                nvidia_smi = subprocess.check_output(['nvidia-smi', '--list-gpus'])
                num_accelerators = len(nvidia_smi.decode('utf-8').strip().split('\n'))
        elif get_accelerator().device_name() == 'xpu':
            clinfo = subprocess.check_output(['clinfo'])
            lines = clinfo.decode('utf-8').strip().split('\n')
            num_accelerators = 0
            for line in lines:
                match = re.search('Device Type.*GPU', line)
                if match:
                    num_accelerators += 1
        elif get_accelerator().device_name() == 'hpu':
            try:
                hl_smi = subprocess.check_output(['hl-smi', "-L"])
                num_accelerators = re.findall(r"Module ID\s+:\s+(\d+)", hl_smi.decode())
            except FileNotFoundError:
                sim_list = subprocess.check_output(['ls', '-1', '/dev/accel'])
                num_accelerators = re.findall(r"accel(\d+)", sim_list.decode())
            num_accelerators = sorted(num_accelerators, key=int)
            os.environ["HABANA_VISIBLE_MODULES"] = ",".join(num_accelerators)
        elif get_accelerator().device_name() == 'npu':
            npu_smi = subprocess.check_output(['npu-smi', 'info', '-l'])
            num_accelerators = int(npu_smi.decode('utf-8').strip().split('\n')[0].split(':')[1].strip())
        else:
            assert get_accelerator().device_name() == 'cpu'
            num_accelerators = _get_cpu_socket_count()

        if isinstance(num_accelerators, list):
            cuda_visible = ",".join(num_accelerators)
        else:
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
    reuse_dist_env = False
    non_daemonic_procs = False
    _pool_cache = {}
    exec_timeout = DEEPSPEED_TEST_TIMEOUT

    @abstractmethod
    def run(self):
        ...

    def __call__(self, request):
        self._fixture_kwargs = self._get_fixture_kwargs(request, self.run)
        world_size = self.world_size
        if self.requires_cuda_env and not get_accelerator().is_available():
            pytest.skip("only supported in accelerator environments.")

        self._launch_with_file_store(request, world_size)

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

    def _launch_daemonic_procs(self, num_procs, init_method):
        # Create process pool or use cached one
        master_port = None

        if get_accelerator().device_name() == 'hpu':
            if self.reuse_dist_env:
                print("Ignoring reuse_dist_env for hpu")
                self.reuse_dist_env = False

        if self.reuse_dist_env:
            if num_procs not in self._pool_cache:
                self._pool_cache[num_procs] = mp.Pool(processes=num_procs)
                master_port = get_master_port()
            pool = self._pool_cache[num_procs]
        else:
            pool = mp.Pool(processes=num_procs)
            master_port = get_master_port()

        # Run the test
        args = [(local_rank, num_procs, master_port, init_method) for local_rank in range(num_procs)]
        skip_msgs_async = pool.starmap_async(self._dist_run, args)

        try:
            skip_msgs = skip_msgs_async.get(self.exec_timeout)
        except mp.TimeoutError:
            # Shortcut to exit pytest in the case of a hanged test. This
            # usually means an environment error and the rest of tests will
            # hang (causing super long unit test runtimes)
            pytest.exit("Test hanged, exiting", returncode=1)
        finally:
            # Regardless of the outcome, ensure proper teardown
            # Tear down distributed environment and close process pools
            self._close_pool(pool, num_procs)

        # If we skipped a test, propagate that to this process
        if any(skip_msgs):
            assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
            pytest.skip(skip_msgs[0])

    def _launch_non_daemonic_procs(self, num_procs, init_method):
        assert not self.reuse_dist_env, "Cannot reuse distributed environment with non-daemonic processes"

        master_port = get_master_port()
        skip_msg = mp.Queue()  # Allows forked processes to share pytest.skip reason
        processes = []
        prev_start_method = mp.get_start_method()
        mp.set_start_method('spawn', force=True)
        for local_rank in range(num_procs):
            p = mp.Process(target=self._dist_run, args=(local_rank, num_procs, master_port, init_method, skip_msg))
            p.start()
            processes.append(p)
        mp.set_start_method(prev_start_method, force=True)

        # Now loop and wait for a test to complete. The spin-wait here isn't a big
        # deal because the number of processes will be O(#GPUs) << O(#CPUs).
        any_done = False
        start = time.time()
        while (not any_done) and ((time.time() - start) < self.exec_timeout):
            for p in processes:
                if not p.is_alive():
                    any_done = True
                    break
            time.sleep(.1)  # So we don't hog CPU

        # If we hit the timeout, then presume a test is hanged
        if not any_done:
            for p in processes:
                p.terminate()
            pytest.exit("Test hanged, exiting", returncode=1)

        # Wait for all other processes to complete
        for p in processes:
            p.join(self.exec_timeout)

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

    def _launch_procs(self, num_procs, init_method):
        # Verify we have enough accelerator devices to run this test
        if get_accelerator().is_available() and get_accelerator().device_count() < num_procs:
            pytest.skip(
                f"Skipping test because not enough GPUs are available: {num_procs} required, {get_accelerator().device_count()} available"
            )

        if get_accelerator().device_name() == 'xpu':
            self.non_daemonic_procs = True
            self.reuse_dist_env = False

        # Set start method to `forkserver` (or `fork`)
        mp.set_start_method('forkserver', force=True)

        if self.non_daemonic_procs:
            self._launch_non_daemonic_procs(num_procs, init_method)
        else:
            self._launch_daemonic_procs(num_procs, init_method)

    def _dist_run(self, local_rank, num_procs, master_port, init_method, skip_msg=""):
        if dist.is_initialized():
            if get_accelerator().is_available():
                # local_rank might not match the rank in the previous run if you are reusing the environment
                get_accelerator().set_device(dist.get_rank())
        else:
            """ Initialize deepspeed.comm and execute the user function. """
            if self.set_dist_env:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = str(master_port)
                os.environ['LOCAL_RANK'] = str(local_rank)
                # NOTE: unit tests don't support multi-node so local_rank == global rank
                os.environ['RANK'] = str(local_rank)
                # In case of multiprocess launching LOCAL_SIZE should be same as WORLD_SIZE
                # DeepSpeed single node launcher would also set LOCAL_SIZE accordingly
                os.environ['LOCAL_SIZE'] = str(num_procs)
                os.environ['WORLD_SIZE'] = str(num_procs)

            # turn off NCCL logging if set
            os.environ.pop('NCCL_DEBUG', None)

            if get_accelerator().is_available():
                set_accelerator_visible()

            if get_accelerator().is_available():
                get_accelerator().set_device(local_rank)

            if self.init_distributed:
                deepspeed.init_distributed(dist_backend=self.backend,
                                           init_method=init_method,
                                           rank=local_rank,
                                           world_size=num_procs)
                dist.barrier()

        try:
            self.run(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                if self.non_daemonic_procs:
                    skip_msg.put(e.msg)
                else:
                    skip_msg = e.msg
            else:
                raise e

        return skip_msg

    def _launch_with_file_store(self, request, world_size):
        tmpdir = request.getfixturevalue("tmpdir")
        dist_file_store = tmpdir.join("dist_file_store")
        assert not os.path.exists(dist_file_store)
        init_method = f"file://{dist_file_store}"

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            try:
                self._launch_procs(procs, init_method)
            finally:
                if os.path.exists(dist_file_store):
                    os.remove(dist_file_store)
            time.sleep(0.5)

    def _dist_destroy(self):
        if (dist is not None) and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _close_pool(self, pool, num_procs, force=False):
        if force or not self.reuse_dist_env:
            msg = pool.starmap(self._dist_destroy, [() for _ in range(num_procs)])
            pool.close()
            pool.join()


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
            world_size = self._fixture_kwargs.get("world_size", self.world_size)

        self._launch_with_file_store(request, world_size)

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)


def get_test_path(filename):
    curr_path = Path(__file__).parent
    return str(curr_path.joinpath(filename))


# bf16 > fp16 > fp32
def preferred_dtype():
    if get_accelerator().is_bf16_supported():
        return torch.bfloat16
    elif get_accelerator().is_fp16_supported():
        return torch.float16
    else:
        return torch.float32


class EnableDeterminism:

    def __init__(self, seed: int):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        self.seed = seed + local_rank
        self.saved_random_state = None
        self.saved_np_random_state = None
        self.saved_cuda_launch_blocking = None
        self.saved_cublas_workspace_config = None
        self.saved_deterministic_algorithms = None

    def __enter__(self):
        self.saved_random_state = random.getstate()
        self.saved_np_random_state = np.random.get_state()
        self.saved_acc_rng_state = get_accelerator().get_rng_state()
        self.saved_cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "")
        self.saved_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        self.saved_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

        random.seed(self.seed)
        np.random.seed(self.seed)
        get_accelerator().manual_seed(self.seed)
        get_accelerator().manual_seed_all(self.seed)

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

    def __exit__(self, type, value, traceback):
        random.setstate(self.saved_random_state)
        np.random.set_state(self.saved_np_random_state)
        get_accelerator().set_rng_state(self.saved_acc_rng_state)
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.saved_cuda_launch_blocking
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.saved_cublas_workspace_config
        torch.use_deterministic_algorithms(self.saved_deterministic_algorithms)


def enable_determinism(seed: int):

    def decorator(func: Callable) -> Callable:

        def wrapper(*args: Any, **kwargs: Any):
            with EnableDeterminism(seed):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def reduce_boolean_flags(flag: bool, op=all) -> bool:
    if not dist.is_initialized():
        return flag
    device = get_accelerator().current_device()
    tensor_flag = torch.tensor(1 if flag else 0, dtype=torch.int, device=device)
    world_size = dist.get_world_size()
    tensor_flag_buf = torch.zeros(world_size, dtype=torch.int, device=device)
    dist.all_gather_into_tensor(tensor_flag_buf, tensor_flag)
    list_flags = [bool(f) for f in tensor_flag_buf.tolist()]
    return op(list_flags)


def allclose_on_all_ranks(actual, expected, assert_message=None, **kwargs) -> None:
    """
    Compare two tensors across all ranks.
    We want to make sure that all ranks succeed or fail together.
    """
    allclose_local = False
    allclose_global = False
    mismatch_msg = ""
    try:
        torch_assert_close(actual, expected, **kwargs)
        allclose_local = True
        allclose_global = reduce_boolean_flags(allclose_local, all)
    except AssertionError:
        allclose_global = reduce_boolean_flags(allclose_local, all)
        mismatch_msg = f"Tensors are not close: {actual=}, {expected=} {kwargs=}"

    if not allclose_global:
        message = "Tensors are not close on all ranks." if assert_message is None else assert_message
        raise AssertionError(f"{message} {mismatch_msg}")
