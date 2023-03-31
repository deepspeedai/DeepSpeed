# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from torch.distributed.elastic.agent.server.api import log, _get_socket_with_port
from torch.distributed.elastic.metrics import put_metric
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerGroup,
    WorkerSpec,
    WorkerState,
)
from torch.distributed import Store
import time
import os
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.utils import macros
import shutil
import copy
from contextlib import closing
import subprocess


class DSElasticAgent(LocalElasticAgent):

    def __init__(
        self,
        spec: WorkerSpec,
        env: Dict,
        start_method="spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ):
        super().__init__(spec, start_method, exit_barrier_timeout, log_dir)
        self.ds_env = env

    @staticmethod
    def _set_master_addr_port(store: Store, master_addr: Optional[str], master_port: Optional[int]):
        if master_port is None:
            sock = _get_socket_with_port()
            with closing(sock):
                master_port = sock.getsockname()[1]

        if master_addr is None:
            # master_addr = _get_fq_hostname()
            result = subprocess.check_output("hostname -I", shell=True)
            master_addr = result.decode('utf-8').split()[0]

        store.set("MASTER_ADDR", master_addr.encode(encoding="UTF-8"))
        store.set("MASTER_PORT", str(master_port).encode(encoding="UTF-8"))

    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"

        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank

            worker_env_ds = copy.deepcopy(self.ds_env)
            worker_env_elastic = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": os.getenv("NCCL_ASYNC_ERROR_HANDLING", str(1)),
            }
            worker_env_ds.update(worker_env_elastic)
            if "OMP_NUM_THREADS" in os.environ:
                worker_env_ds["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            envs[local_rank] = worker_env_ds
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        # scaling events do not count towards restarts (gets same attempt #)
        # remove existing log dir if this restart is due to a scaling event
        attempt_log_dir = os.path.join(self._log_dir, f"attempt_{restart_count}")
        shutil.rmtree(attempt_log_dir, ignore_errors=True)
        os.makedirs(attempt_log_dir)

        assert spec.entrypoint is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
        )

        return self._pcontext.pids()

    def _invoke_run(self, role: str = "default") -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        log.info(f"[{role}] starting workers for entrypoint: {spec.get_entrypoint_name()}")

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        participants = rdzv_handler._state_holder.state.participants

        while True:
            assert self._worker_group.state != WorkerState.INIT
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            expire_time = datetime.utcnow() - (rdzv_handler._settings.keep_alive_interval *
                                               rdzv_handler._settings.keep_alive_max_attempt)
            _dead_nodes = [
                node for node, last_heartbeat in rdzv_handler._state_holder.state.last_heartbeats.items()
                if last_heartbeat < expire_time
            ]

            put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
            put_metric(f"workers.{role}.{state.name.lower()}", 1)

            if state == WorkerState.SUCCEEDED:
                log.info(f"[{role}] worker group successfully finished."
                         f" Waiting {self._exit_barrier_timeout} seconds for other agents to finish.")
                self._exit_barrier()
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED
                           } or len(participants) > len(rdzv_handler._state_holder.state.participants):
                if self._remaining_restarts > 0:
                    log.info(f"[{role}] Worker group {state.name}. "
                             f"{self._remaining_restarts}/{spec.max_restarts} attempts left;"
                             f" will restart worker group")
                    self._remaining_restarts -= 1
                    # rdzv_handler._state_holder.state.restart = False
                    self._restart_workers(self._worker_group)
                    participants = rdzv_handler._state_holder.state.participants

                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    self._exit_barrier()
                    return run_result
            elif state == WorkerState.HEALTHY:
                # membership changes do not count as retries
                num_nodes_waiting = rdzv_handler.num_nodes_waiting()
                group_rank = self._worker_group.group_rank
                if num_nodes_waiting > 0:
                    log.info(f"[{role}] Detected {num_nodes_waiting} "
                             f"new nodes from group_rank={group_rank}; "
                             f"will restart worker group")
                    self._restart_workers(self._worker_group)
                    participants = rdzv_handler._state_holder.state.participants
            else:
                raise Exception(f"[{role}] Worker group in {state.name} state")
