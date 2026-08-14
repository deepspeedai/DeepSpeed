# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

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

# The measured margin is clamped into this range. The floor is the calibrated value, not a token
# non-zero one, because the measurement and the risk are different quantities. reserved-minus-
# allocated is read while this pass runs, before the memory-heavy phase, so it is near zero and the
# floor decides every time -- it read exactly 0.0100 in all twelve cells of the 2026-08-13 sweep.
# What the margin actually has to cover is peak-time fragmentation plus the error in the floor
# profile, and both are far larger: at seq4096 the run died with 4.5GB of nominal slack and at
# seq3584 with 3.5GB, putting the usable ceiling near 145GB rather than the card's 150.1GB, while
# the floor profile itself was off by up to 3.3GB. A 1% floor covers neither; 10% covers both.
#
# The ceiling stays above the floor so the measurement still does something in the safe direction:
# an allocator genuinely holding more than a tenth of the card can raise the reserve, but nothing
# can lower it below the calibrated value. Collapsing both ends to 0.1 would leave a function that
# reads two counters, divides them, and returns a constant -- the appearance of measurement with
# none of it, which is the failure this comment exists to prevent.
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
    """The share of the device the allocator is holding without using.

    This is what the margin is standing in for. Reading it beats guessing: a flat tenth of an H200
    reserves 14GiB, which at the sequence lengths that need this pass is most of the room the
    planner has to give activations back with.
    """
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

    The profile behind this number is measured locally, and it varies across ranks: the
    out-of-torch term reads NVML for one device, and no two devices carry the same non-torch
    memory. The plan below spends the gap between this and the budget, so ranks that disagree
    here hand back different sets of activations: the rank that measured the highest floor keeps
    the most resident, which is the rank least able to afford it, and it runs out of memory on a
    plan that fit everywhere else. The busiest rank decides, the same way the smallest device
    decides the budget.
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


def _eligible_activations(graph: Graph, graph_id: int, num_fwd_outputs, param_manager) -> List[Tuple[Node, int]]:
    """Every saved activation this pass is allowed to move, largest first."""
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
    no_copy_ops = get_no_copy_ops()

    # No profile means no peak to plan against, and the usual reason it is missing is that profiling
    # itself ran out of memory -- which is evidence of exactly the pressure this pass relieves. Take
    # everything in that case, size floor included: measured at seq4096, moving only the tensors
    # above the floor still ran out of memory, while moving all of them completed the run.
    min_size = _min_offload_size()
    _skipped.clear()

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
            _skipped["duplicate"] += _skip_bytes(node)
            continue
        # A parameter is already managed by ZeRO, and a value the caller also receives has to stay
        # on the device.
        if node in returned_to_caller or node.name in param_names:
            _skipped["param_or_output"] += _skip_bytes(node)
            continue
        # A value that only aliases another tensor shares its storage, so copying it out frees
        # nothing while the tensor it aliases is still live.
        if node.target in no_copy_ops:
            _skipped["alias"] += _skip_bytes(node)
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
        candidates.append((node, size))

    if _skipped:
        breakdown = " ".join(f"{k}={v}" for k, v in sorted(_skipped.items()))
        print_rank_0(f"offload_activation graph_id={graph_id} skipped bytes by rule: {breakdown}")

    # Largest first, so bringing tensors back later returns the most memory per copy avoided.
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)
    return candidates


def _report_partitioner_split(graph: Graph, graph_id: int, num_fwd_outputs: int) -> None:
    """Say how much of the forward AOTAutograd chose to keep, and how much it already recomputes.

    This pass can only move what the partitioner decided to save; anything it dropped is recomputed
    in the backward and was never this pass's to offload. Without this line the two are impossible
    to tell apart in the numbers -- a small offload plan looks like a timid planner when it may
    simply be that the partitioner already threw most of the forward away.
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

    Profiling replays the graph, so profiling the graph as written measures the memory the pass was
    called in to reduce -- and at the sequence lengths that need this pass, that replay is the first
    thing to run out of memory, leaving no numbers to plan with. Moving everything first means the
    replay runs against the low-water mark instead, which fits, and the planner that follows has
    measurements rather than guesses. The optimizer-state pass reaches the same profile by emptying
    its states before profiling; this is the same idea for values the graph itself produces.
    """
    graph = gm.graph
    # A later compile phase plans again from the original graph, so drop any earlier plan first.
    _offload_plans[graph_id] = OrderedDict()

    _report_partitioner_split(graph, graph_id, profiling_results[graph_id].num_fwd_outputs)

    selected = _eligible_activations(graph, graph_id, profiling_results[graph_id].num_fwd_outputs, param_manager)
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
    print_rank_0(f"offload_activation graph_id={graph_id} floor: moved all {len(selected)} eligible "
                 f"activations ({sum(size for _, size in selected) / 1e9:.1f}GB) before profiling")
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

    Mirrors _bring_back on the forward side. Once the planner keeps an activation resident, the
    value reaching the backward graph is the tensor itself rather than a host buffer, so its reload
    is not merely wasted -- it would copy from a host buffer nothing wrote.
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
    """Bring activations back while the floor profile says they fit.

    The pass before this one moved everything, so the profile now describes the graph at its lowest
    memory. Whatever headroom is left between that floor and the budget can be spent keeping
    activations resident, cheapest-to-move first, which is the amount this pass should move rather
    than the everything it starts from.
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

    # The profile covers the compiled forward graph, and work that runs outside it is invisible to
    # the floor. DeepSpeed's tiled loss is the case that bites: it drives a nested autograd.backward
    # from inside its own forward, so none of that memory appears in this profile. Measured on
    # Qwen3-14B, the floor overshoots the run's real peak whenever the loss is untiled -- seq2048
    # +4.4GB, seq3072 +3.1GB, seq3584 +1.7GB, all conservative -- and undershoots as soon as tiling
    # is on: seq3072 goes from +3.1GB to -1.9GB with nothing changed but the loss, and seq4096
    # undershoots by 3.1GB against 2.3GB of real headroom, which is why every plan there died.
    #
    # The profiling replays ran the whole step for real, tiled loss included, so the allocator's
    # high-water mark has already seen what the per-node profile missed. Taking the larger of the
    # two lets the floor rise to meet reality and never fall below it: the planner can only become
    # more conservative, never less.
    # Prefer what the run actually reached over what the profile predicts. By the time this pass
    # runs, several steps have already executed the fully-offloaded configuration, so their peak IS
    # the floor -- measured, for this exact graph, with the tiled loss and everything else included.
    # The profile is an estimate of the same quantity and a poor one in both directions: taken at
    # step 0 it missed the tiled loss's nested backward and read 3.1GB low at seq4096, and taken
    # here it reads the graph replay plus whatever else is live at that instant, which put it 6.5GB
    # high at seq3200 and 13GB high at seq3072 -- enough to leave the planner nothing to spend.
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

    # Largest first: each one returned buys back the most memory per copy avoided.
    by_size = sorted(plan.items(), key=lambda item: item[1][1], reverse=True)
    kept_resident = 0
    for name, (_, size) in by_size:
        if size > headroom:
            continue
        _bring_back(gm.graph, name)
        del plan[name]
        headroom -= size
        kept_resident += size
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
