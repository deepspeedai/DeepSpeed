Affine Shard Transfers
======================

AutoTP and AutoEP both describe the shards they create with affine maps from
``deepspeed.checkpoint.affine``. ``deepspeed.checkpoint.affine_transfer``
plans how to move a parameter from one such sharding to another, and
``deepspeed.checkpoint.affine_transfer_executor`` executes that plan directly
into existing packed target shards without reconstructing the full logical
tensor. Neither depends on which parallelism produced the maps.

Executing affine transfers
--------------------------

Endpoint mappings use transport-group ranks and distinguish source locations
from target locations::

    from deepspeed.checkpoint.affine_transfer import plan_transfer
    from deepspeed.checkpoint.affine_transfer_executor import (
        StagingBudget, TransferEndpoints, execute_transfer)

    plan = plan_transfer(target_map, source_map)
    execute_transfer(plan, local_sources, local_targets,
                     TransferEndpoints(source_bindings, target_bindings),
                     dtype, StagingBudget(bytes_per_peer=8 << 20))

Use ``execute_transfers`` with uniquely keyed ``TransferRequest`` objects to pack
multiple parameters into peer messages. Each process supplies only its own
contiguous buffers; strided plan regions and buffer storage offsets are retained.
All participants must agree on requests, endpoints, dtype and staging budget.
Preflight refusal leaves targets untouched. Return guarantees target readiness;
a transport failure after payload begins can leave partially written targets.
The budget applies to each peer direction, not total device memory. Weights must
remain frozen during transfer; optimizer migration and live AutoEP swaps are outside
this API. Maintenance contact: ``@0z5a``.

Planning a sync with replica policies
-------------------------------------

``plan_transfers`` plans every parameter of one sync and hands one replica policy every request at
once, so a cost-aware policy can balance load over all of it::

    from deepspeed.checkpoint.affine_replica_policy import local_first_least_loaded
    from deepspeed.checkpoint.affine_transfer import plan_transfers

    plans = plan_transfers(pairs,
                           source_processes=source_processes,
                           target_processes=target_processes,
                           element_size=element_size,
                           min_split_bytes=1 << 20,
                           replica_policy=local_first_least_loaded)

``pairs`` maps a parameter name to its ``(target_map, source_map)``. Every pair is validated as in
``plan_transfer``, and ``limits`` applies to each parameter on its own.

The policy receives one ``ReplicaRequest`` per intersection of a target piece with a source
region, in a fixed order, and returns one answer per request. A request names the parameter, the
target location, the region's ``logical_origin``, ``shape``, target offset and strides, and every
validated holder as a ``ReplicaCandidate``. ``source_processes`` and ``target_processes`` map plan
locations to processes, so a policy can tell a local copy from a transfer; without them the
process fields are ``None``. ``element_size`` gives bytes per element for every parameter in
``pairs``, which fills ``bytes`` and is required by ``min_split_bytes``; a missing or non-positive
entry is refused before the policy runs.

An answer is the location of one validated holder, or ``ReplicaPart`` objects that cut the region
along axis 0 and name a holder for each cut. The planner refuses an answer unless there is one per
request, every location is a validated holder of that request, and the parts tile the request
exactly along axis 0; a split is also refused when a part is under ``min_split_bytes``, and a
parameter is refused when its parts exceed ``PlanBudget.max_pair_checks``. Such refusals raise
``InvalidReplicaPolicyError`` before any segment is emitted.

``deepspeed.checkpoint.affine_replica_policy`` provides ``first_holder`` (``plan_transfer``'s
default), ``local_first_least_loaded`` (the default here) and ``per_request``, which runs a
per-request ``replica_selector`` as a batch policy. Every rank must return the same answers, so
cost inputs must come from data every rank shares, such as a link calibration gathered across
ranks, never from a rank-local measurement.

Transport and staging
---------------------

The staging arena and ``StagingBudget`` belong to the current transport, not to
affine transfer itself. Torch point-to-point operations move one contiguous
tensor per peer per message, so a round whose regions are strided or
come from several segments is packed through a per-peer arena; a round that is
one region already contiguous in its buffer is sent or received in place. A
zero-copy transport such as GPUDirect RDMA would instead take each round's
``(offset, length)`` pairs as a scatter/gather list over registered memory. That
removes the arena copy but needs a registration lifetime this API has no place
for, so the wire contract (one packed message per peer per round) is expected to
change when such a path lands.

Calibrating links
-----------------

``calibrate_links`` measures what moving bytes costs between the processes of
a process group, the default group unless ``group`` is given, so that a cost
model or replica policy can price a plan's
messages with numbers from the machine it runs on. It only measures; it does not
plan, choose replicas, or fit anything by itself. Every process calls it and
receives the same raw samples::

    from deepspeed.checkpoint.affine_transfer_calibration import (calibrate_links, fit_links,
                                                                   node_path_of)

    calibration = calibrate_links(path_of=node_path_of(processes_per_node=8), fan_width=[1, 2, 4, 7])
    fits = fit_links(calibration)
    fits['peer_pairs'][(0, 1)]       # the link from process 0 to process 1 alone
    fits['peer_both_ways_pairs'][(0, 1)]  # the pair's completion time while 0 and 1 send to each other
    fits['relay'][('intra_node', 'inter_node')]      # receiving within a node while sending out of it
    fits['fan_in'][(0, (1, 2, 3, 4))]['inter_node']  # per-link cost while 1-4 send to 0 at once
    fits['receive'][(0, (1, 2, 3, 4))]               # process 0's aggregate rate receiving from 1-4

Five patterns are sampled at every size, 4 KiB to 256 MiB by default:

- ``local``: a device-local copy on each process, labeled ``path_of(process, process)``.
- ``peer``: each process pair alone (1 to 1), one way, and both ways at once
  (``both_ways``), as when a process sends and receives together in colocated
  resharding. A two-way exchange gives a sample per direction, labelled with
  that direction's path.
- ``relay``: every process receives from ``process - shift`` while sending to
  ``process + shift``, for every shift, as in a pipelined relay or the
  all-gather stage of one. Each sample is one process's time, labelled with
  both its ``receive_path`` and its ``send_path``.
- ``fan_out``: one root sends to ``width`` peers at once (1 to N).
- ``fan_in``: ``width`` peers send to one root at once (N to 1), measuring
  receiver contention.

``fan_width`` takes one width or a list of them; each fan sample records its
``width``. It defaults to one less than the fewest processes on any node, and at
least 1, so the default fans fit in a node however large the job is. A root's
default peers are taken from ``root + 1`` onwards, those on its node (by host
name) first. ``fan_peers(root, width)`` picks other peers, for example the
senders behind one PCIe switch or across a socket, and must return the same
peers on every process.

Each message is labelled with ``path_of(from_process, to_process)``. The default
labels only ``local`` and ``peer``; ``node_path_of`` splits ``peer`` into
``intra_node`` and ``inter_node``, and any callable can be used, for example one
that tells NVLink from PCIe pairs. ``fit_links`` pools the samples per label and
fits ``alpha + bytes / bandwidth`` to the median at each size, and
``peer_pairs`` fits each ordered pair alone, both from one-way samples;
``peer_both_ways`` and ``peer_both_ways_pairs`` do the same for the two-way
samples. Both directions of a two-way sample carry the pair's completion time,
so ``(a, b)`` and ``(b, a)`` fit the same; comparing them with the slower of the
two one-way ``peer_pairs`` fits shows what a link loses when both directions are
busy. ``relay`` is fitted per
``(receive_path, send_path)``. Fan fits are keyed by
``(root, peers)``, with the peers sorted; the width is ``len(peers)``. ``send``
and ``receive`` fit each root's own time against all the bytes it moved, giving
the root's aggregate rate with those peers, which is what a plan's per-process
sent and received bytes are priced against.

A fan finishes when the busiest resource it crosses does: a peer's link, the
root, or a link the peers share. Behind PCIe switches that depends on *which*
peers share a process more than on how many, which is why fans are not keyed by
width alone.

There, use ``peer_pairs``, label paths by topology, and choose the sender sets
to measure with ``fan_peers``; to compare several sets of one width, calibrate
once per set and concatenate the ``fan_out`` and ``fan_in`` samples before
fitting. On a uniform fabric, fits of one width can be pooled by
``len(peers)``.

Processes here are ranks of ``group``; pass the group the transfer runs on. A
plan's source and target locations are
mapped to processes by ``TransferEndpoints``, so translate through it before
joining a plan's messages with these labels. The pair and relay sweeps make
``(1.5 * world_size + 1) * (world_size - 1)`` barriered samples per size and repeat, and each
process holds up to ``(2 + max(widths)) * max(sizes)`` bytes, 2.25 GiB with the
default width at 8 processes per node and 256 MiB, so pass shorter
``sizes`` or smaller fan widths on large jobs. The numbers describe the
transport used for the measurement (torch point-to-point) on that topology.
