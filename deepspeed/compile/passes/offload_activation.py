# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except ImportError:
    # Unsupported torch version
    pass

from ..fx import get_output_node
from ..graph_param import DSGraphParamManager
from ..util import get_no_copy_ops
from .contract import PassContract

NAME = "offload_activation"
# Moves tensors that the forward graph saves for the backward pass. It neither reads nor rewrites
# what the other passes produce, so it has no capability requirements.
CONTRACT = PassContract()

# Share of device memory left to the allocator and to whatever lives outside the compiled graph.
MARGIN = 0.1

# Below this size a tensor costs more in copy launches and event bookkeeping than the memory it
# returns. Same threshold the free_activation path uses.
MIN_OFFLOAD_SIZE = 10 * 1024 * 1024

# Used only if the bandwidth measurement below fails.
DEFAULT_H2D_BYTES_PER_SEC = 10e9

# The C++ ops that move a tensor to pinned host memory and bring it back.
_ACTIVATION_OFFLOAD_OPS = ("offload_tensor", "wait_offload", "reload_tensor", "wait_reload")

_activation_ops_lib = None

# Activations chosen while rewriting each forward graph, keyed by graph id and then by node name.
# The backward graph receives those tensors as placeholders carrying the same names, which is how
# the two halves of the pass find each other.
_offload_plans: Dict[int, "OrderedDict[str, Tuple[int, int]]"] = {}

# Value ids identify a host buffer inside the C++ executor. They never repeat, so a buffer holding
# a tensor of one shape is never reused for another.
_next_value_id = 0

_h2d_bytes_per_sec = None

# Nodes inserted so far. A run whose forward graph carries offload nodes but whose backward graph
# carries no reload nodes has silently lost its activations, so tests assert on both.
_stats = {"offload_nodes": 0, "reload_nodes": 0}


def get_offload_activation_stats():
    return dict(_stats)


def reset_offload_activation_stats():
    for key in _stats:
        _stats[key] = 0


def print_rank_0(message):
    # Straight to stdout, like the optimizer-state offload pass: these lines are the only record of
    # what the pass decided, and the DeepSpeed logger is turned down in many training harnesses.
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(message)


def register_activation_offload_ops():
    """Give the activation-offload ops a Meta kernel so the rewritten graph can be traced.

    The compiled extension registers CPU and CUDA kernels for these ops but no Meta kernel, and
    without one the compiler cannot trace a graph that contains them.
    """
    global _activation_ops_lib
    if _activation_ops_lib is not None:
        return

    # FRAGMENT, not DEF: the compiled extension creates the "dc" namespace with TORCH_LIBRARY, and
    # a namespace may only be created once per process. A fragment extends it without claiming it.
    lib = torch.library.Library("dc", "FRAGMENT")
    for name in _ACTIVATION_OFFLOAD_OPS:
        if torch._C._dispatch_has_kernel_for_dispatch_key(f"dc::{name}", "Meta"):
            continue
        lib.impl(name, lambda tensor, graph_id, value_id: torch.empty_like(tensor), "Meta")

    # The ops deregister if the library object is garbage collected.
    _activation_ops_lib = lib


def _new_value_id() -> int:
    global _next_value_id
    _next_value_id += 1
    return _next_value_id


def _memory_budget() -> float:
    budget_override = os.environ.get("DS_DC_OFFLOAD_ACT_BUDGET_GB")
    if budget_override is not None:
        # Test hook: pretend the device has this much memory, to force or suppress offloading.
        return float(budget_override) * 1e9

    accelerator = get_accelerator()
    budget = accelerator.total_memory() * (1 - MARGIN)
    # Ranks run the same graph, so they must reach the same plan. The smallest device decides.
    vals_to_bcast = torch.tensor([budget], device=torch.device(accelerator.current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    return vals_to_bcast[0].item()


def _min_offload_size() -> int:
    size_override = os.environ.get("DS_DC_OFFLOAD_ACT_MIN_SIZE_MB")
    if size_override is not None:
        # Test hook: the models used in tests have activations far below the real threshold.
        return int(float(size_override) * 1024 * 1024)
    return MIN_OFFLOAD_SIZE


def _h2d_bandwidth() -> float:
    """Measure how fast this device reads pinned host memory, in bytes per second.

    Platforms differ by more than an order of magnitude here, and the measurement decides how far
    ahead of its first use each reload has to start.
    """
    global _h2d_bytes_per_sec
    if _h2d_bytes_per_sec is not None:
        return _h2d_bytes_per_sec

    accelerator = get_accelerator()
    _h2d_bytes_per_sec = DEFAULT_H2D_BYTES_PER_SEC
    try:
        with unset_fake_temporarily():
            num_bytes = 32 * 1024 * 1024
            host_buffer = accelerator.pin_memory(torch.empty(num_bytes, dtype=torch.uint8, device="cpu"))
            device_buffer = torch.empty(num_bytes, dtype=torch.uint8, device=accelerator.current_device_name())

            # The first copy pays for pinning bookkeeping and stream setup; leave it out.
            device_buffer.copy_(host_buffer, non_blocking=True)
            accelerator.synchronize()

            iterations = 5
            start = time.perf_counter()
            for _ in range(iterations):
                device_buffer.copy_(host_buffer, non_blocking=True)
            accelerator.synchronize()
            elapsed = time.perf_counter() - start

        if elapsed > 0:
            _h2d_bytes_per_sec = num_bytes * iterations / elapsed
    except Exception as e:
        print_rank_0(f"offload_activation could not measure host-to-device bandwidth ({e}); "
                     f"assuming {DEFAULT_H2D_BYTES_PER_SEC / 1e9:.1f}GB/s")

    print_rank_0(f"offload_activation host-to-device bandwidth {_h2d_bytes_per_sec / 1e9:.2f}GB/s")
    return _h2d_bytes_per_sec


def _static_tensor_size(node: Node) -> Optional[int]:
    """Size of the tensor a node produces, or None if it is not a tensor of known size.

    A host buffer is allocated once per value id and reused every step, so a tensor whose shape is
    symbolic (it can change between steps) is not a candidate.
    """
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None
    # A value already on the host has nothing to move, and the copy op holds its input through
    # record_stream, which only exists for device tensors.
    if val.device.type == "cpu":
        return None
    if any(not isinstance(dim, int) for dim in val.shape):
        return None
    return val.numel() * val.element_size()


def _is_floating_point(node: Node) -> bool:
    val = node.meta.get("val")
    return isinstance(val, torch.Tensor) and val.is_floating_point()


def _copy_tensor_meta(src: Node, dst: Node) -> None:
    for key in ("val", "tensor_meta"):
        if key in src.meta:
            dst.meta[key] = src.meta[key]


def _insertion_point_after_last_use(nodes: List[Node], node: Node) -> Node:
    """Return the node to insert the copy before: the first node after the tensor's last use.

    Copying earlier would read a tensor still being written; copying later than the last use keeps
    the device copy alive for no reason, because nothing frees it until the copy op releases it.
    """
    last_use_index = nodes.index(node)
    for index, candidate in enumerate(nodes):
        if candidate.op == "output":
            continue
        if node in candidate.all_input_nodes:
            last_use_index = index

    insert_index = last_use_index + 1
    # Every placeholder has to stay at the head of the graph, so a tensor that is a graph input and
    # is never read again starts its copy at the first node that follows the placeholders.
    while nodes[insert_index].op == "placeholder":
        insert_index += 1
    return nodes[insert_index]


def _select_activations(graph: Graph, graph_id: int, profile, param_manager) -> List[Tuple[Node, int]]:
    """Choose which saved activations to move, largest first, until the profiled peak fits."""
    output_node = get_output_node(graph)
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        print_rank_0(f"offload_activation graph_id={graph_id} unexpected output format; skipping")
        return []

    # The partitioner puts the values the caller receives first and the values saved for the
    # backward pass after them. Only the saved ones live until the backward pass, and returning a
    # host tensor to the caller would change what the model outputs.
    num_fwd_outputs = profile.num_fwd_outputs
    if num_fwd_outputs is None:
        print_rank_0(f"offload_activation graph_id={graph_id} has no partition information; skipping")
        return []

    returned_to_caller = set(node for node in outputs[:num_fwd_outputs] if isinstance(node, Node))
    param_names = set(param_manager[graph_id].param_names) if graph_id in param_manager else set()
    min_size = _min_offload_size()
    no_copy_ops = get_no_copy_ops()

    saved_nodes = [node for node in outputs[num_fwd_outputs:] if isinstance(node, Node)]

    candidates = []
    seen = set()
    for node in saved_nodes:
        if node in seen:
            continue
        seen.add(node)
        # A value saved twice reaches the backward graph as two placeholders under two names, and
        # only the one named after this node would be reloaded. Leave it alone.
        if saved_nodes.count(node) > 1:
            continue
        # A parameter is already managed by ZeRO, and a value the caller also receives has to stay
        # on the device.
        if node in returned_to_caller or node.name in param_names:
            continue
        # A value that only aliases another tensor shares its storage, so copying it out frees
        # nothing while the tensor it aliases is still live.
        if node.target in no_copy_ops:
            continue
        # Only floating-point values are activations. The rest are bookkeeping the backward pass
        # needs -- indices, masks, and the random-number state that attention saves. That state is
        # the reason this test cannot be a device check: it lives on the host, but an op's traced
        # metadata takes its device from the op's inputs, so it claims to be on the accelerator.
        if not _is_floating_point(node):
            continue
        size = _static_tensor_size(node)
        if size is None or size < min_size:
            continue
        candidates.append((node, size))

    if not candidates:
        return []

    if not profile.fwd_mem:
        print_rank_0(f"offload_activation graph_id={graph_id} incomplete profiling data; skipping")
        return []

    budget = _memory_budget()
    peak = max(peak for _, _, _, peak in profile.fwd_mem)

    # Largest first: the fewest copies for the memory returned.
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)

    selected = []
    offloaded_bytes = 0
    for node, size in candidates:
        if peak - offloaded_bytes <= budget:
            break
        selected.append((node, size))
        offloaded_bytes += size

    print_rank_0(f"offload_activation graph_id={graph_id} peak={peak} budget={budget} "
                 f"candidates={len(candidates)} selected={len(selected)} selected_bytes={offloaded_bytes}")
    return selected


def _offload_activation_fwd(gm: GraphModule, graph_id: int, profiling_results, param_manager) -> Optional[GraphModule]:
    graph = gm.graph
    # A later compile phase plans again from the original graph, so drop any earlier plan first.
    _offload_plans[graph_id] = OrderedDict()

    selected = _select_activations(graph, graph_id, profiling_results[graph_id], param_manager)
    if not selected:
        return None

    output_node = get_output_node(graph)
    for node, size in selected:
        value_id = _new_value_id()
        # The graph is re-read for every tensor because each insertion changes it.
        insert_before = _insertion_point_after_last_use(list(graph.nodes), node)

        with graph.inserting_before(insert_before):
            offload_node = graph.create_node('call_function',
                                             torch.ops.dc.offload_tensor.default, (node, graph_id, value_id), {},
                                             name=f"offload_{node.name}")
        _copy_tensor_meta(node, offload_node)

        # The wait sits immediately after the copy, which makes the copy synchronous and costs the
        # overlap. It is the only correct placement under inductor: inductor's liveness knows
        # nothing about streams, so the only way to stop it writing into a buffer whose copy is
        # still in flight is to mark that buffer as never reused -- and that keeps the buffer alive
        # for the whole forward pass, which is the memory this pass exists to release. Waiting here
        # ends the copy before anything else runs, so the buffer is genuinely dead afterwards and
        # inductor frees and recycles it as usual.
        with graph.inserting_after(offload_node):
            wait_node = graph.create_node('call_function',
                                          torch.ops.dc.wait_offload.default, (offload_node, graph_id, value_id), {},
                                          name=f"wait_offload_{node.name}")
        _copy_tensor_meta(node, wait_node)

        output_node.replace_input_with(node, wait_node)
        _offload_plans[graph_id][node.name] = (value_id, size)
        _stats["offload_nodes"] += 1

    graph.lint()
    gm.recompile()
    # Returning None skips the caller's re-profiling replay. Nothing later in this schedule reads
    # that profile, and replaying a graph that is already at the memory wall is precisely what this
    # pass exists to avoid: the replay ran out of memory on one rank, whose profiler then stopped
    # taking part in the per-node collectives while the other ranks kept waiting, and the job died
    # half an hour later on a collective timeout.
    return None


def _reload_activation_bwd(gm: GraphModule, graph_id: int, profiling_results) -> Optional[GraphModule]:
    plan = _offload_plans.get(graph_id)
    if not plan:
        return None

    graph = gm.graph
    profile = profiling_results[graph_id]
    node_time_ms = {name: device_time for name, device_time, _ in profile.bwd_time}
    peak_mem = {name: peak for name, _, _, peak in profile.bwd_mem}
    budget = _memory_budget()
    bandwidth = _h2d_bandwidth()

    nodes = list(graph.nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    placeholders = {node.name: node for node in nodes if node.op == "placeholder"}

    targets = []
    for name, (value_id, size) in plan.items():
        node = placeholders.get(name)
        if node is None:
            print_rank_0(f"offload_activation graph_id={graph_id} offloaded {name} never reaches this backward graph")
            continue
        users = [user for user in node.users if user.op != "output"]
        if not users:
            continue
        first_user = min(users, key=lambda user: node_index[user])
        targets.append((node_index[first_user], node, first_user, value_id, size))

    # Backward reads the activations roughly in reverse order of the forward pass that produced
    # them, so starting the copies in order of first use is also the order the copy stream drains.
    targets.sort(key=lambda target: target[0])

    reloaded_bytes = 0
    for first_user_index, node, first_user, value_id, size in targets:
        copy_time_ms = size / bandwidth * 1000
        insert_before = first_user
        elapsed_ms = 0.0
        for index in range(first_user_index - 1, -1, -1):
            candidate = nodes[index]
            if candidate.op == "placeholder":
                break
            # Starting earlier only helps while the memory the tensor takes back still fits.
            if peak_mem.get(candidate.name, 0) + reloaded_bytes + size > budget:
                break
            elapsed_ms += node_time_ms.get(candidate.name, 0.0)
            insert_before = candidate
            if elapsed_ms >= copy_time_ms:
                break

        with graph.inserting_before(insert_before):
            reload_node = graph.create_node('call_function',
                                            torch.ops.dc.reload_tensor.default, (node, graph_id, value_id), {},
                                            name=f"reload_{node.name}")
        _copy_tensor_meta(node, reload_node)

        with graph.inserting_before(first_user):
            wait_node = graph.create_node('call_function',
                                          torch.ops.dc.wait_reload.default, (reload_node, graph_id, value_id), {},
                                          name=f"wait_reload_{node.name}")
        _copy_tensor_meta(node, wait_node)

        for user in list(node.users):
            if user is not reload_node:
                user.replace_input_with(node, wait_node)

        reloaded_bytes += size
        _stats["reload_nodes"] += 1

    if not targets:
        return None

    graph.lint()
    gm.recompile()
    # Same reason as the forward half: no consumer for the profile, and the replay is the risk.
    return None


def offload_activation(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                       create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                       bwd: bool) -> Optional[GraphModule]:
    """Move activations saved for the backward pass to pinned host memory and bring them back.

    The forward half copies a chosen tensor out right after its last use and hands the host buffer
    to the backward pass instead of the device tensor, which is what frees the memory. The backward
    half starts the copy back far enough ahead of the tensor's first use for the transfer to hide
    behind the compute in between.
    """
    register_activation_offload_ops()

    if bwd:
        return _reload_activation_bwd(gm, graph_id, profiling_results)
    return _offload_activation_fwd(gm, graph_id, profiling_results, param_manager)
