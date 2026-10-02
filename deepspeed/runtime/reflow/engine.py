# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Engine-side wiring for Reflow, kept out of deepspeed/runtime/engine.py.

Each function takes the engine, so the engine only carries one call per decision point.
"""

import functools
from typing import TYPE_CHECKING

import torch

from deepspeed.runtime.zero.utils import ZeRORuntimeException
from deepspeed.utils import log_dist

if TYPE_CHECKING:
    from deepspeed.runtime.engine import DeepSpeedEngine


def reflow_enabled(engine: "DeepSpeedEngine") -> bool:
    """Whether this run uses Reflow: the `reflow` config block is set and ZenFlow is not."""
    if engine._config.zero_config.reflow is None:
        return False
    # The throughput timer is built before _configure_zenflow runs, so read the flag defensively.
    return not getattr(engine, "zenflow", False)


def reflow_num_threads(engine: "DeepSpeedEngine"):
    return engine._config.zero_config.reflow.num_threads


def stage3_optimizer_class(engine: "DeepSpeedEngine"):
    """Reflow's ZeRO-3 subclass with its config bound, so the engine's Stage-3 call is the shared one."""
    from deepspeed.runtime.reflow.reflow_stage3 import ReflowOptimizer_Stage3
    return functools.partial(ReflowOptimizer_Stage3, reflow_config=engine._config.zero_config.reflow)


def cpu_adam_class():
    """Reflow's CPU-Adam subclass, with the async / cpu_conversion / bucketwise step kernels."""
    from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
    return ReflowCPUAdam


def build_cpu_lion(engine: "DeepSpeedEngine", model_parameters, optimizer_parameters: dict):
    """Reflow's CPU-Lion subclass, built over the client's Lion parameters."""
    from deepspeed.runtime.reflow.reflow_cpu_lion import ReflowCPULion
    return ReflowCPULion(model_parameters, **optimizer_parameters, num_threads=reflow_num_threads(engine))


def maybe_remap_client_optimizer(engine: "DeepSpeedEngine", optimizer):
    """Reflow's ZeRO-3 optimizer requires a Reflow CPU optimizer subclass (ReflowCPUAdam or
    ReflowCPULion). If a client passed a plain DeepSpeedCPUAdam / torch Adam(W) or a
    DeepSpeedCPULion (the usual ZeRO-Offload pattern) instead, rebuild an equivalent Reflow
    optimizer over the same param groups and hyperparameters so existing scripts work unchanged.
    Returns the optimizer unchanged if it is already a Reflow CPU optimizer."""
    from deepspeed.runtime.reflow.reflow_cpu_adam import ReflowCPUAdam
    from deepspeed.runtime.reflow.reflow_cpu_lion import ReflowCPULion
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from deepspeed.ops.lion import DeepSpeedCPULion
    from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
    if not getattr(optimizer, "fp32_optimizer_states", True):
        raise ZeRORuntimeException("Reflow requires FP32 optimizer states; fp32_optimizer_states=False "
                                   "is not supported.")
    if isinstance(optimizer, (ReflowCPUAdam, ReflowCPULion)):
        return optimizer
    if isinstance(optimizer, MuonWithAuxAdam):
        # Its class name contains "Adam", so the remap below would rebuild it as ReflowCPUAdam and silently train
        # every parameter with Adam. Supporting Muon in Reflow is future work.
        raise ZeRORuntimeException("Reflow does not support the Muon optimizer yet; remove the 'reflow' block "
                                   "from zero_optimization to use Muon with ZeRO-Offload")
    optimizer_type = type(optimizer)
    name = optimizer_type.__name__
    defaults = getattr(optimizer, "defaults", {})
    is_lion = optimizer_type is DeepSpeedCPULion
    if optimizer_type not in (DeepSpeedCPUAdam, DeepSpeedCPULion, torch.optim.Adam, torch.optim.AdamW):
        raise ZeRORuntimeException(
            f"Reflow cannot remap the client optimizer {name} without changing its update rule. "
            "Pass a DeepSpeedCPUAdam / torch AdamW / DeepSpeedCPULion, or set the optimizer in the "
            "DeepSpeed config.")
    scheduler = engine.client_lr_scheduler
    if scheduler is not None and getattr(scheduler, "optimizer", None) is optimizer:
        raise ZeRORuntimeException("Reflow cannot remap a client optimizer already bound to an LR scheduler. "
                                   "Pass a scheduler factory or set the scheduler in the DeepSpeed config.")
    if is_lion:
        keep = ("lr", "betas", "weight_decay")
    else:
        keep = ("lr", "betas", "eps", "weight_decay", "bias_correction", "amsgrad", "maximize")
    param_groups = [{"params": g["params"], **{k: g[k] for k in keep if k in g}} for g in optimizer.param_groups]
    if is_lion:
        log_dist(f"Reflow enabled: remapping client optimizer {name} to ReflowCPULion", ranks=[0])
        return ReflowCPULion(param_groups,
                             lr=defaults.get("lr", 1e-3),
                             betas=defaults.get("betas", (0.9, 0.999)),
                             weight_decay=defaults.get("weight_decay", 0.0),
                             num_threads=reflow_num_threads(engine))
    adamw_mode = getattr(optimizer, "adam_w_mode", None)
    if adamw_mode is None:
        adamw_mode = optimizer_type is torch.optim.AdamW
    log_dist(f"Reflow enabled: remapping client optimizer {name} to ReflowCPUAdam", ranks=[0])
    return ReflowCPUAdam(param_groups,
                         lr=defaults.get("lr", 1e-3),
                         betas=defaults.get("betas", (0.9, 0.999)),
                         eps=defaults.get("eps", 1e-8),
                         weight_decay=defaults.get("weight_decay", 0.0),
                         bias_correction=defaults.get("bias_correction", True),
                         amsgrad=defaults.get("amsgrad", False),
                         maximize=defaults.get("maximize", False),
                         adamw_mode=bool(adamw_mode),
                         num_threads=reflow_num_threads(engine))
