# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import functools
import os
import time
from collections import OrderedDict, defaultdict
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
from ..profilers.graph_profile import _get_mem_usage_out_of_torch
from ..util import get_no_copy_ops
from .contract import PassContract

NAME = "offload_activation"
FLOOR_NAME = "offload_activation_floor"
# Moves tensors that the forward graph saves for the backward pass. It neither reads nor rewrites
# what the other passes produce, so it has no capability requirements.
CONTRACT = PassContract()

# Fallback share of device memory left to the allocator, used only when the real overhead cannot be
# read. The measured value is normally far smaller: with expandable_segments the allocator wastes
# little, and every point of margin here is memory the planner may not give back to the device.
MARGIN = 0.1

# reserved-minus-allocated is read before the memory-heavy phase, so it is near zero and the
# floor decides in practice. The floor is the calibrated value: what a margin has to cover is
# peak-time fragmentation plus floor-profile error, both far larger than the reading. The
# ceiling stays above the floor so an unusually fat allocator can still raise the reserve.
MIN_MEASURED_MARGIN = 0.1
MAX_MEASURED_MARGIN = 0.25

# Below this size a tensor costs more in copy launches and event bookkeeping than the memory it
# returns. The floor is ignored once the fallback below fires: at seq4096 the tensors above 10MB
# were not enough to fit the run, while moving every one of them was.
MIN_OFFLOAD_SIZE = 5 * 1024 * 1024

# Used only if the bandwidth measurement below fails.
DEFAULT_H2D_BYTES_PER_SEC = 10e9

# The C++ ops that move a tensor to pinned host memory and bring it back.
_ACTIVATION_OFFLOAD_OPS = ("offload_tensor", "wait_offload", "reload_tensor", "wait_reload")

_activation_ops_lib = None

# Activations chosen while rewriting each forward graph, keyed by graph id and then by node name.
# The backward graph receives those tensors as placeholders carrying the same names, which is how
# the two halves of the pass find each other.
_offload_plans: Dict[int, "OrderedDict[str, Tuple[int, int]]"] = {}

# Bytes each planned value keeps allocated if it stays on the device, keyed the same way. This is
# not the size of the copy: a saved view holds its whole base allocation alive, so a 4KB row of a
# 4MB tensor costs 4MB of residency and 4KB of copy. The planner spends headroom in residency
# bytes, while the backward pass schedules its copies in copy bytes.
_resident_bytes: Dict[int, Dict[str, int]] = {}

# Value ids identify a host buffer inside the C++ executor. They never repeat, so a buffer holding
# a tensor of one shape is never reused for another.
_next_value_id = 0

_h2d_bytes_per_sec = None

# Nodes inserted so far. A run whose forward graph carries offload nodes but whose backward graph
# carries no reload nodes has silently lost its activations, so tests assert on both.
# offload_nodes/reload_nodes count what was ever built. planned_offloads is what the last plan
# actually left on the host, so a rebuild that quietly dropped every offload shows up as 0
# instead of hiding behind a stale build counter.
_stats = {"offload_nodes": 0, "reload_nodes": 0, "planned_offloads": 0}


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


def _measured_margin(accelerator) -> float:
    """The share of the device the allocator holds without using, clamped to a calibrated range."""
    margin_override = os.environ.get("DS_DC_OFFLOAD_ACT_MARGIN")
    if margin_override is not None:
        return float(margin_override)

    total = accelerator.total_memory()
    if not total:
        return MARGIN

    overhead = max(0, accelerator.memory_reserved() - accelerator.memory_allocated())
    margin = overhead / total
    return min(max(margin, MIN_MEASURED_MARGIN), MAX_MEASURED_MARGIN)


def _memory_budget() -> float:
    budget_override = os.environ.get("DS_DC_OFFLOAD_ACT_BUDGET_GB")
    if budget_override is not None:
        # Test hook: pretend the device has this much memory, to force or suppress offloading.
        return float(budget_override) * 1e9

    accelerator = get_accelerator()
    margin = _measured_margin(accelerator)
    budget = accelerator.total_memory() * (1 - margin)
    if not dist.is_initialized():
        return budget

    # Ranks run the same graph, so they must reach the same plan. The smallest device decides.
    vals_to_bcast = torch.tensor([budget], device=torch.device(accelerator.current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    return vals_to_bcast[0].item()


def _agree_on_floor_peak(floor_peak: int) -> float:
    """Make every rank plan against the same floor.

    This number is measured locally and varies across ranks. Ranks that disagree hand back
    different sets, and the rank reading the highest floor keeps the most while being least able
    to afford it. The busiest rank decides, mirroring the budget's MIN.
    """
    if not dist.is_initialized():
        return float(floor_peak)

    vals_to_bcast = torch.tensor([float(floor_peak)], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MAX)
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


_skipped = defaultdict(int)


def _skip_bytes(node) -> int:
    """Bytes a rejected candidate would have carried, for the skip breakdown."""
    val = node.meta.get("val", None) if hasattr(node, "meta") else None
    if val is None or not hasattr(val, "numel") or not hasattr(val, "element_size"):
        return 0
    try:
        return int(val.numel()) * int(val.element_size())
    except Exception:
        return 0


@functools.lru_cache
def _aliasing_ops():
    """Ops whose output shares storage with an input, for the purpose of tracking that storage.

    get_no_copy_ops() reads the aten schemas, and aten._unsafe_view declares a fresh tensor return
    even though it hands back a view -- that declaration is the entire point of the op. It shares
    storage all the same, and AOTAutograd emits it after nearly every matmul, so this pass has to
    know about it. It is added here rather than in get_no_copy_ops() because that set also decides
    where the ZeRO-3 passes release parameters, and this pass has no business changing that.
    """
    return frozenset(get_no_copy_ops() | {torch.ops.aten._unsafe_view.default})


def _zero3_gathered_param_ops():
    """The op that produces a ZeRO-3 gathered parameter buffer, empty if DeepCompile is not built."""
    try:
        return {torch.ops.dc.allgather_param.default}
    except (AttributeError, RuntimeError):
        return set()


def _alias_root(node: Node, no_copy_ops, cache: Dict[Node, Node]) -> Node:
    """The node that allocated the storage `node` reads.

    An aliasing op returns a tensor that shares the storage of its first tensor input, so following
    that input back through the aliasing ops reaches the node that allocated the memory. A node
    whose op is not an aliasing one owns its storage and is its own root. Ops that alias an input
    other than the first are rare enough that they are not tracked; such a node is treated as
    owning its storage, which only costs a copy that frees nothing.
    """
    chain = []
    current = node
    while current not in cache:
        if current.target not in no_copy_ops:
            break
        base = next((arg for arg in current.all_input_nodes if isinstance(arg.meta.get("val"), torch.Tensor)), None)
        if base is None or base in chain:
            break
        chain.append(current)
        current = base

    root = cache.get(current, current)
    for aliasing_node in chain:
        cache[aliasing_node] = root
    cache[current] = root
    return root


def _storage_keeper_counts(graph: Graph, saved_nodes: List[Node], returned_to_caller, no_copy_ops,
                           cache: Dict[Node, Node]) -> Dict[Node, int]:
    """How many values that outlive the forward pass hold each storage.

    Moving a saved value to the host releases its memory only if nothing else still points at that
    storage. Four kinds of value still do:

    - the graph's own inputs, which the caller owns;
    - get_attr nodes, whose tensor the GraphModule holds for the life of the process (attention
      masks, rotary embedding tables, and the other constants inductor bakes in);
    - ZeRO-3 gathered parameters, whose buffer belongs to ZeRO's own registry and is released by
      release_param, not by this graph. The forward graph reaches one of these through
      dc.wait_allgather, which is an aliasing op, so without this the gathered weight looks like
      an ordinary activation with nothing else holding it;
    - the values the graph returns to the caller, and every other saved value.
    """
    counts = defaultdict(int)
    gather_ops = _zero3_gathered_param_ops()
    keepers = [node for node in graph.nodes if node.op in ("placeholder", "get_attr") or node.target in gather_ops]
    keepers.extend(returned_to_caller)
    keepers.extend(saved_nodes)
    for keeper in keepers:
        counts[_alias_root(keeper, no_copy_ops, cache)] += 1
    return counts


def _has_non_overlapping_storage(node: Node) -> bool:
    """Whether the tensor's elements each occupy their own place in storage.

    Both the host buffer and the reloaded device tensor are made with `empty_like`, which allocates
    one element per logical element. A tensor whose elements overlap holds fewer elements of
    storage than it has entries -- `expand` is the ordinary case, repeating a row with a stride of
    zero -- so the round trip would copy and allocate several times what the value actually keeps
    alive. An expanded row of 1000 floats seen as 4x1000 copies 16KB each way to release 4KB.

    Strides that are merely non-contiguous are fine and are deliberately allowed. `empty_like` does
    return a contiguous tensor for a strided slice or one piece of a split, so the reload hands the
    backward pass different strides than the traced metadata promises. That was measured on torch
    2.6.0+cu124 with torch._inductor.config.size_asserts off, as init_z3.py sets it: an opaque op
    whose meta claims stride (576, 8, 72, 1) while returning (192, 8, 24, 1), consumed by matmul,
    batched matmul, linear and reductions under torch.compile, matched eager exactly in every case.
    """
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return False
    try:
        shape, strides = val.shape, val.stride()
    except (RuntimeError, NotImplementedError):
        # Sparse, nested and other non-strided layouts have no strides to compare.
        return False
    # Symbolic sizes or strides cannot be checked here; the static-size rule rejects them anyway.
    if any(not isinstance(dim, int) for dim in (*shape, *strides)):
        return False

    # How many elements of storage the tensor spans, against how many entries it has.
    span = 1 + sum((size - 1) * abs(stride) for size, stride in zip(shape, strides))
    return val.numel() <= span


def _eligible_activations(graph: Graph, graph_id: int, num_fwd_outputs, param_manager) -> List[Tuple[Node, int, int]]:
    """Every saved activation this pass is allowed to move, largest first.

    Each entry is (node, bytes copied, bytes kept allocated if the value stays resident). The two
    sizes differ for a view: the copy carries the view's own bytes while residency holds the whole
    allocation the view points into.
    """
    output_node = get_output_node(graph)
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        print_rank_0(f"offload_activation graph_id={graph_id} unexpected output format; skipping")
        return []

    # The partitioner puts the values the caller receives first and the values saved for the
    # backward pass after them. Only the saved ones live until the backward pass, and returning a
    # host tensor to the caller would change what the model outputs.
    if num_fwd_outputs is None:
        print_rank_0(f"offload_activation graph_id={graph_id} has no partition information; skipping")
        return []

    returned_to_caller = set(node for node in outputs[:num_fwd_outputs] if isinstance(node, Node))
    param_names = set(param_manager[graph_id].param_names) if graph_id in param_manager else set()
    no_copy_ops = _aliasing_ops()

    # No profile means no peak to plan against, and the usual reason it is missing is that profiling
    # itself ran out of memory -- which is evidence of exactly the pressure this pass relieves. Take
    # everything in that case, size floor included: measured at seq4096, moving only the tensors
    # above the floor still ran out of memory, while moving all of them completed the run.
    min_size = _min_offload_size()
    _skipped.clear()

    saved_nodes = [node for node in outputs[num_fwd_outputs:] if isinstance(node, Node)]
    alias_roots: Dict[Node, Node] = {}
    storage_keepers = _storage_keeper_counts(graph, saved_nodes, returned_to_caller, no_copy_ops, alias_roots)

    candidates = []
    seen = set()
    for node in saved_nodes:
        if node in seen:
            continue
        seen.add(node)
        # A value saved twice reaches the backward graph as two placeholders under two names, and
        # only the one named after this node would be reloaded. Leave it alone.
        if saved_nodes.count(node) > 1:
            _skipped["duplicate"] += _skip_bytes(node)
            continue
        # A parameter is already managed by ZeRO, and a value the caller also receives has to stay
        # on the device.
        if node in returned_to_caller or node.name in param_names:
            _skipped["param_or_output"] += _skip_bytes(node)
            continue
        # A value that only aliases another tensor shares its storage, so copying it out frees
        # nothing while another value that outlives the forward pass still points at that storage.
        # When no other value does -- AOTAutograd routinely saves the view and not the tensor it
        # came from -- this view is the last holder, and moving it releases the whole allocation.
        root = _alias_root(node, no_copy_ops, alias_roots)
        # root is node for an aliasing op only when the walk could not find the tensor it aliases,
        # and an unknown base is not a base this pass may assume is dead.
        if node.target in no_copy_ops and (root is node or storage_keepers[root] > 1):
            _skipped["alias"] += _skip_bytes(node)
            continue
        # Checked for every candidate, not only the ones the rule above let through: a piece of a
        # split reaches here through operator.getitem, which is not an aliasing op, so the alias
        # rule never sees it even though the tensor is a view.
        if not _has_non_overlapping_storage(node):
            _skipped["overlapping"] += _skip_bytes(node)
            continue
        # Only floating-point values are activations. The rest are bookkeeping the backward pass
        # needs -- indices, masks, and the random-number state that attention saves. That state is
        # the reason this test cannot be a device check: it lives on the host, but an op's traced
        # metadata takes its device from the op's inputs, so it claims to be on the accelerator.
        if not _is_floating_point(node):
            _skipped["not_float"] += _skip_bytes(node)
            continue
        size = _static_tensor_size(node)
        if size is None or size < min_size:
            _skipped["too_small" if size is not None else "no_static_size"] += _skip_bytes(node)
            continue
        # Keeping this value resident holds its whole allocation, which for a view is the base's.
        # The alias rule above guarantees this view is the only saved value pointing there, so no
        # two entries ever charge the planner for the same bytes.
        resident = _static_tensor_size(root) if root is not node else size
        candidates.append((node, size, resident if resident is not None else size))

    if _skipped:
        breakdown = " ".join(f"{k}={v}" for k, v in sorted(_skipped.items()))
        print_rank_0(f"offload_activation graph_id={graph_id} skipped bytes by rule: {breakdown}")

    # Largest first, so bringing tensors back later returns the most memory per copy avoided.
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    return candidates


def _report_partitioner_split(graph: Graph, graph_id: int, num_fwd_outputs: int) -> None:
    """Say how much of the forward AOTAutograd kept, and how much it already recomputes.

    Only the saved set is this pass's to move. Without this, a small plan looks like a timid
    planner when the partitioner may simply have discarded most of the forward already.
    """
    output_node = get_output_node(graph)
    outputs = output_node.args[0]
    saved = outputs[num_fwd_outputs:] if isinstance(outputs, (list, tuple)) else []

    def _bytes(node) -> int:
        if not hasattr(node, "meta"):
            return 0
        val = node.meta.get("val", None)
        if val is None or not hasattr(val, "numel") or not hasattr(val, "element_size"):
            return 0
        try:
            return int(val.numel()) * int(val.element_size())
        except Exception:
            return 0

    produced_bytes = 0
    produced_count = 0
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        size = _bytes(node)
        if size:
            produced_bytes += size
            produced_count += 1

    saved_bytes = sum(_bytes(n) for n in saved if hasattr(n, "meta"))
    dropped_bytes = max(0, produced_bytes - saved_bytes)
    share = (100.0 * saved_bytes / produced_bytes) if produced_bytes else 0.0
    print_rank_0(f"offload_activation graph_id={graph_id} partitioner split: forward produces "
                 f"{produced_count} tensors / {produced_bytes} bytes; saved for backward "
                 f"{len(saved)} / {saved_bytes} bytes ({share:.1f}%); not saved (recomputed or dead) "
                 f"{dropped_bytes} bytes")


def _offload_everything_fwd(gm: GraphModule, graph_id: int, profiling_results, param_manager):
    """Move every eligible activation out, so the profile taken next measures the floor.

    Profiling the graph as written measures the memory this pass exists to reduce, and that replay
    is the first thing to run out of memory where the pass is needed. `move_opt_states` reaches its
    floor the same way, by emptying states before profiling.
    """
    graph = gm.graph
    # A later compile phase plans again from the original graph, so drop any earlier plan first.
    _offload_plans[graph_id] = OrderedDict()
    _resident_bytes[graph_id] = {}

    _report_partitioner_split(graph, graph_id, profiling_results[graph_id].num_fwd_outputs)

    selected = _eligible_activations(graph, graph_id, profiling_results[graph_id].num_fwd_outputs, param_manager)
    if not selected:
        return None

    output_node = get_output_node(graph)
    for node, size, resident in selected:
        value_id = _new_value_id()
        # The graph is re-read for every tensor because each insertion changes it.
        insert_before = _insertion_point_after_last_use(list(graph.nodes), node)

        with graph.inserting_before(insert_before):
            offload_node = graph.create_node('call_function',
                                             torch.ops.dc.offload_tensor.default, (node, graph_id, value_id), {},
                                             name=f"offload_{node.name}")
        _copy_tensor_meta(node, offload_node)

        # Waiting here makes the copy synchronous and costs the overlap, but it is the only
        # correct placement: inductor's liveness is stream-unaware, so the alternative is marking
        # the buffer never-reused, which keeps it alive for the whole forward -- the very memory
        # this pass releases.
        with graph.inserting_after(offload_node):
            wait_node = graph.create_node('call_function',
                                          torch.ops.dc.wait_offload.default, (offload_node, graph_id, value_id), {},
                                          name=f"wait_offload_{node.name}")
        _copy_tensor_meta(node, wait_node)

        output_node.replace_input_with(node, wait_node)
        _offload_plans[graph_id][node.name] = (value_id, size)
        _resident_bytes.setdefault(graph_id, {})[node.name] = resident
        _stats["offload_nodes"] += 1

    graph.lint()
    print_rank_0(f"offload_activation graph_id={graph_id} floor: moved all {len(selected)} eligible "
                 f"activations ({sum(size for _, size, _ in selected) / 1e9:.1f}GB copied, "
                 f"{sum(resident for _, _, resident in selected) / 1e9:.1f}GB released) before profiling")
    # Returned, not None: the caller profiles what it gets back, and that profile is the floor the
    # planner needs.
    return gm


def _bring_back(graph: Graph, name: str) -> None:
    """Undo one activation's move: the graph keeps it resident again."""
    by_name = {node.name: node for node in graph.nodes}
    original = by_name.get(name)
    offload_node = by_name.get(f"offload_{name}")
    wait_node = by_name.get(f"wait_offload_{name}")
    if original is None or offload_node is None or wait_node is None:
        return

    # Whoever reads the host buffer goes back to reading the tensor itself, and the two copy nodes
    # are erased users-first, since the wait reads the copy.
    for user in list(wait_node.users):
        user.replace_input_with(wait_node, original)
    graph.erase_node(wait_node)
    graph.erase_node(offload_node)


def _remove_reload(graph: Graph, name: str) -> None:
    """Undo one activation's copy back: the backward reads the tensor directly again.

    Mirrors _bring_back. Once an activation stays resident the backward placeholder carries the
    tensor, so a leftover reload would copy from a host buffer nothing wrote.
    """
    by_name = {node.name: node for node in graph.nodes}
    placeholder = by_name.get(name)
    reload_node = by_name.get(f"reload_{name}")
    wait_node = by_name.get(f"wait_reload_{name}")
    if placeholder is None or reload_node is None or wait_node is None:
        return

    for user in list(wait_node.users):
        user.replace_input_with(wait_node, placeholder)
    graph.erase_node(wait_node)
    graph.erase_node(reload_node)


def _drop_reloads_for_resident_bwd(gm: GraphModule, graph_id: int) -> None:
    """Remove the copies back for whatever the forward half decided to keep on the device."""
    plan = _offload_plans.get(graph_id) or {}
    graph = gm.graph
    removed = 0
    for node in list(graph.nodes):
        if not node.name.startswith("wait_reload_"):
            continue
        name = node.name[len("wait_reload_"):]
        if name in plan:
            continue
        _remove_reload(graph, name)
        removed += 1
        _stats["reload_nodes"] -= 1

    if removed:
        print_rank_0(f"offload_activation graph_id={graph_id} dropped {removed} reloads for "
                     f"activations the planner kept resident")
        graph.lint()
        gm.recompile()


def _plan_against_floor_fwd(gm: GraphModule, graph_id: int, profiling_results) -> Optional[GraphModule]:
    """Bring activations back while the floor says they fit.

    The floor pass moved everything, so headroom between that floor and the budget is what can be
    kept resident. Largest first: most bytes not copied per unit of headroom spent.
    """
    plan = _offload_plans.get(graph_id)
    if not plan:
        return None

    profile = profiling_results[graph_id]
    if not profile.fwd_mem:
        # Even the floor could not be profiled. Keeping everything on the host is the only safe
        # reading of that, and it is what the run needs to have any chance of starting.
        print_rank_0(f"offload_activation graph_id={graph_id} floor could not be profiled either; "
                     f"keeping all {len(plan)} activations on the host")
        return None

    # Prefer what the run actually reached over what the profile predicts. Several steps have
    # already executed the fully-offloaded configuration by now, so their peak IS the floor,
    # measured for this graph with the tiled loss included. The profile estimates the same
    # quantity badly in both directions -- it cannot see work outside the compiled graph, and
    # taken here it also counts whatever else is live. See activation-offload-learnings.md.
    profiled_floor = _agree_on_floor_peak(max(peak for _, _, _, peak in profile.fwd_mem))
    observed_floor = _agree_on_floor_peak(get_accelerator().max_memory_allocated() + _get_mem_usage_out_of_torch())
    if observed_floor > 0:
        floor_peak = observed_floor
        source = "observed"
    else:
        floor_peak = profiled_floor
        source = "profiled"
    print_rank_0(f"offload_activation graph_id={graph_id} floor from {source}: "
                 f"observed={observed_floor} profiled={profiled_floor}")
    budget = _memory_budget()
    headroom = budget - floor_peak
    # Report where the budget came from. DS_DC_OFFLOAD_ACT_BUDGET_GB bypasses the margin entirely,
    # so printing a margin next to an overridden budget invites a reader to believe the margin
    # produced it -- the split-sweep logs of 2026-08-13 all read margin=0.0100 while their budgets
    # came from the hook.
    if os.environ.get("DS_DC_OFFLOAD_ACT_BUDGET_GB") is not None:
        margin_note = "budget=overridden(margin unused)"
    else:
        margin_note = f"margin={_measured_margin(get_accelerator()):.4f}"
    print_rank_0(f"offload_activation graph_id={graph_id} {margin_note} "
                 f"floor_peak={floor_peak} budget={budget} headroom={headroom}")

    # Largest first: each one returned buys back the most memory per copy avoided. What it costs
    # is residency, which for a saved view is the whole allocation the view points into, not the
    # view's own bytes. Charging the copy size here would let a handful of small views of large
    # tensors retain many times the headroom the planner thinks it spent.
    resident_bytes = _resident_bytes.get(graph_id, {})
    by_size = sorted(plan.items(), key=lambda item: item[1][1], reverse=True)
    kept_resident = 0
    for name, (_, size) in by_size:
        cost = resident_bytes.get(name, size)
        if cost > headroom:
            continue
        _bring_back(gm.graph, name)
        del plan[name]
        headroom -= cost
        kept_resident += cost
        _stats["offload_nodes"] -= 1

    moved_bytes = sum(size for _, size in plan.values())
    _stats["planned_offloads"] = len(plan)
    print_rank_0(f"offload_activation graph_id={graph_id} floor_peak={floor_peak} budget={budget} "
                 f"kept_resident={kept_resident} selected={len(plan)} selected_bytes={moved_bytes}")

    gm.graph.lint()
    gm.recompile()
    # None: the caller skips its replay. The graph only shrank in memory terms from the profile
    # just taken, and replaying it again at this pressure is what used to hang the job.
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

    # Without a backward profile every node reads as using no memory, so the headroom check below
    # cannot refuse anything and every copy back is hoisted as early as it will go -- the backward
    # pass then holds everything the forward pass just moved out, and the run dies with the same plan
    # that survives when the copies come back one at a time. A missing profile means profiling ran
    # out of memory, so bring each tensor back at the point it is needed and nowhere sooner.
    reload_just_in_time = not profile.bwd_mem
    if reload_just_in_time:
        print_rank_0(f"offload_activation graph_id={graph_id} no backward profile; bringing each "
                     f"tensor back just before its first use")

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
        search_start = -1 if reload_just_in_time else first_user_index - 1
        for index in range(search_start, -1, -1):
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


def offload_activation_floor(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                             create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                             bwd: bool) -> Optional[GraphModule]:
    """First half: move every eligible activation out so the next profile measures the floor.

    This half has to stand alone. The planner that trims it runs at WARMUP, so the steps before that
    execute exactly what this pass produced -- a forward that moves activations to the host and a
    backward that copies them back. Leaving the backward untouched here would offload without ever
    reloading, and the backward would read a host buffer on a CUDA matmul.
    """
    register_activation_offload_ops()

    if bwd:
        return _reload_activation_bwd(gm, graph_id, profiling_results)
    return _offload_everything_fwd(gm, graph_id, profiling_results, param_manager)


def offload_activation(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                       create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager,
                       bwd: bool) -> Optional[GraphModule]:
    """Second half: keep what fits on the device, and bring the rest back before the backward reads it.

    Schedule this after offload_activation_floor. That pass moves everything and is profiled; this
    one reads the resulting floor and returns to the device whatever the budget has room for. The
    backward graph is rewritten here too, by which point the forward plan is final.
    """
    register_activation_offload_ops()

    if bwd:
        _drop_reloads_for_resident_bwd(gm, graph_id)
        return None
    return _plan_against_floor_fwd(gm, graph_id, profiling_results)
