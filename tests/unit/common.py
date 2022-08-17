import os
import time
import inspect
from abc import ABC
from pathlib import Path

import torch
import torch.multiprocessing as mp
import deepspeed
import deepspeed.comm as dist
from torch.multiprocessing import Process

import pytest
from _pytest.outcomes import Skipped
from _pytest.fixtures import FixtureLookupError

# Worker timeout *after* the first worker has completed.
DEEPSPEED_UNIT_WORKER_TIMEOUT = 120


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


def set_cuda_visibile():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is None:
        xdist_worker_id = 0
    if cuda_visible is None:
        # CUDA_VISIBLE_DEVICES is not set, discover it from nvidia-smi instead
        import subprocess
        is_rocm_pytorch = hasattr(torch.version, 'hip') and torch.version.hip is not None
        if is_rocm_pytorch:
            rocm_smi = subprocess.check_output(['rocm-smi', '--showid'])
            gpu_ids = filter(lambda s: 'GPU' in s,
                             rocm_smi.decode('utf-8').strip().split('\n'))
            num_gpus = len(list(gpu_ids))
        else:
            nvidia_smi = subprocess.check_output(['nvidia-smi', '--list-gpus'])
            num_gpus = len(nvidia_smi.decode('utf-8').strip().split('\n'))
        cuda_visible = ",".join(map(str, range(num_gpus)))

    # rotate list based on xdist worker id, example below
    # wid=0 -> ['0', '1', '2', '3']
    # wid=1 -> ['1', '2', '3', '0']
    # wid=2 -> ['2', '3', '0', '1']
    # wid=3 -> ['3', '0', '1', '2']
    dev_id_list = cuda_visible.split(",")
    dev_id_list = dev_id_list[xdist_worker_id:] + dev_id_list[:xdist_worker_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(dev_id_list)


class DistributedTest(ABC):
    is_dist_test = True
    world_size = 2
    backend = "nccl"
    init_distributed = True
    set_dist_env = True

    # Temporary directory that is shared among test methods in a class
    @pytest.fixture(autouse=True, scope="class")
    def class_tmpdir(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp(self.__class__.__name__)
        return fn

    def _run_test(self, request):
        self.current_test = self._get_current_test_func(request)
        self.test_kwargs = self._get_test_kwargs(request)

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

    def _get_test_kwargs(self, request):
        # Grab fixture / parametrize kwargs from pytest request object
        test_kwargs = {}
        params = inspect.getfullargspec(self.current_test).args
        params.remove("self")
        for p in params:
            try:
                test_kwargs[p] = request.getfixturevalue(p)
            except FixtureLookupError:
                pass  # test methods can have kwargs that are not fixtures
        return test_kwargs

    def _launch_procs(self, num_procs):
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
        while not any_done:
            for p in processes:
                if not p.is_alive():
                    any_done = True
                    break

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
                pytest.fail(f'Worker {rank} killed by signal {-p.exitcode}',
                            pytrace=False)
            if p.exitcode > 0:
                pytest.fail(f'Worker {rank} exited with code {p.exitcode}',
                            pytrace=False)

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

        set_cuda_visibile()

        if self.init_distributed:
            deepspeed.init_distributed(dist_backend=self.backend)
            dist.barrier()

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        try:
            self.current_test(**self.test_kwargs)
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


def distributed_test(world_size=2, backend='nccl'):
    """A decorator for executing a function (e.g., a unit test) in a distributed manner.
    This decorator manages the spawning and joining of processes, initialization of
    deepspeed.comm, and catching of errors.

    Usage example:
        @distributed_test(worker_size=[2,3])
        def my_test():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert(rank < world_size)

    Arguments:
        world_size (int or list): number of ranks to spawn. Can be a list to spawn
        multiple tests.
    """
    def dist_wrap(run_func):
        """Second-level decorator for dist_test. This actually wraps the function. """
        def dist_init(local_rank, num_procs, *func_args, **func_kwargs):
            """Initialize deepspeed.comm and execute the user function. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = get_master_port()
            os.environ['LOCAL_RANK'] = str(local_rank)
            # NOTE: unit tests don't support multi-node so local_rank == global rank
            os.environ['RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(num_procs)

            # turn off NCCL logging if set
            os.environ.pop('NCCL_DEBUG', None)

            set_cuda_visibile()

            deepspeed.init_distributed(dist_backend=backend)
            #dist.init_process_group(backend=backend)
            dist.barrier()

            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            run_func(*func_args, **func_kwargs)

            # make sure all ranks finish at the same time
            dist.barrier()
            # tear down after test completes
            dist.destroy_process_group()

        def dist_launcher(num_procs, *func_args, **func_kwargs):
            """Launch processes and gracefully handle failures. """

            # Spawn all workers on subprocesses.
            processes = []
            for local_rank in range(num_procs):
                p = Process(target=dist_init,
                            args=(local_rank,
                                  num_procs,
                                  *func_args),
                            kwargs=func_kwargs)
                p.start()
                processes.append(p)

            # Now loop and wait for a test to complete. The spin-wait here isn't a big
            # deal because the number of processes will be O(#GPUs) << O(#CPUs).
            any_done = False
            while not any_done:
                for p in processes:
                    if not p.is_alive():
                        any_done = True
                        break

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
                    pytest.fail(f'Worker {rank} killed by signal {-p.exitcode}',
                                pytrace=False)
                if p.exitcode > 0:
                    pytest.fail(f'Worker {rank} exited with code {p.exitcode}',
                                pytrace=False)

        def run_func_decorator(*func_args, **func_kwargs):
            """Entry point for @distributed_test(). """

            if isinstance(world_size, int):
                dist_launcher(world_size, *func_args, **func_kwargs)
            elif isinstance(world_size, list):
                for procs in world_size:
                    dist_launcher(procs, *func_args, **func_kwargs)
                    time.sleep(0.5)
            else:
                raise TypeError(f'world_size must be an integer or a list of integers.')

        return run_func_decorator

    return dist_wrap


def get_test_path(filename):
    curr_path = Path(__file__).parent
    return str(curr_path.joinpath(filename))
